#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.light.light_chats.worlds import (
    LightChatOnboardingWorld, LightChatTaskWorld
)
import parlai.mturk.core.mturk_utils as mturk_utils
import parlai.mturk.tasks.light.light_chats.graph as graph
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.tasks.light.light_chats.task_config import task_config
from parlai.tasks.light_dialog.build import download as download_light
import os
import random
import pickle


class GraphGenerator(object):
    def __init__(self, opt, use_seen):
        download_light(opt)
        self.opt = opt
        self.use_seen = use_seen
        dpath = os.path.join(opt['datapath'], 'light_dialogue')
        env_file = os.path.join(dpath, 'light_environment.pkl')
        with open(env_file, 'rb') as picklefile:
            self.db = pickle.load(picklefile)

        # Split rooms into seen and unseen
        seen_rooms = {i: r
                      for i, r in self.db['rooms'].items()
                      if int(r['room_id']) < 703}
        unseen_rooms = {i: r
                        for i, r in self.db['rooms'].items()
                        if int(r['room_id']) >= 703}
        if use_seen:
            self.rooms = seen_rooms
        else:
            self.rooms = unseen_rooms
        for i, room in self.rooms.items():
            room['id'] = i

        # only use annotated characters
        self.chars = {i: c
                      for i, c in self.db['characters'].items()
                      if 'desc' in c}
        for i, char in self.chars.items():
            char['id'] = i
        self.rooms_list = list(self.rooms.values())
        self.chars_list = list(self.chars.values())
        self.room_idx = 3  # Arbitrary start index for what room to load

    def props_from_obj(self, obj):
        use_classes = ['object']
        props = {
            'object': True,
            'size': 1,
            'food_energy': 0,
            'value': 1,
            'desc': random.choice(obj['descriptions'])
        }
        if obj['is_surface'] > 0.5:
            props['container'] = True
            props['contain_size'] = 3
            props['surface_type'] = 'on'
            use_classes.append('container')
        if obj['is_container'] > 0.5:
            props['container'] = True
            props['contain_size'] = 3
            props['surface_type'] = 'on'
            use_classes.append('container')
        if obj['is_drink'] > 0.5:
            props['drink'] = True
            use_classes.append('drink')
        if obj['is_food'] > 0.5:
            props['food'] = True
            use_classes.append('food')
        if obj['is_gettable'] < 0.33:
            use_classes.append('not_gettable')
        if obj['is_wearable'] > 0.5:
            props['wearable'] = True
            props['stats'] = {
                'attack': 1
            }
            use_classes.append('wearable')
        if obj['is_weapon'] > 0.5:
            props['weapon'] = True
            props['stats'] = {
                'attack': 1
            }
            use_classes.append('weapon')

        props['classes'] = use_classes

        return props

    def props_from_char(self, char):
        use_classes = ['agent']
        props = {
            'agent': True,
            'size': 20,
            'contain_size': 20,
            'health': 10,
            'food_energy': 1,
            'aggression': 0,
            'speed': 5,
            'char_type': char['char_type'],
            'desc': random.choice(char['personas']),
        }

        props['classes'] = use_classes

        return props

    def get_room(self):
        room = self.rooms_list[self.room_idx % len(self.rooms_list)]
        self.room_idx += 1

        g = graph.Graph(self.opt)

        room_gid = g.add_node(
            room['setting'],
            {
                'room': True,
                'desc': room['description'],
                'extra_desc': room['background'],
                'room': True,
                'contain_size': 2000,  # TODO turk room sizes
                'name_prefix': "the",
                'surface_type': "in",
                'classes': {'room'},
            }
        )

        # Add items to the graph
        added_objs = []
        for item_id in room['ex_objects']:
            if random.random() > 0.5:
                continue
            obj = self.db['objects'][item_id]
            use_desc = obj['name'] if obj['is_plural'] == 0 \
                else random.choice(obj['base_form'])
            if len(use_desc.split(' ')) > 5:
                continue  # Skip really long objects
            if use_desc.lower() in added_objs:
                continue
            added_objs.append(use_desc.lower())
            obj_id = g.add_node(use_desc, self.props_from_obj(obj))
            g.move_object(obj_id, room_gid)
        for item_id in room['in_objects']:
            obj = self.db['objects'][item_id]
            use_desc = obj['name'] if obj['is_plural'] == 0 \
                else random.choice(obj['base_form'])
            if len(use_desc.split(' ')) > 8:
                continue  # Skip really long objects
            if use_desc.lower() in added_objs:
                continue
            added_objs.append(use_desc.lower())
            props = self.props_from_obj(obj)
            obj_id = g.add_node(use_desc, props)
            g.move_object(obj_id, room_gid)

        # Add characters to the graph
        create_characters = []
        in_characters = []
        used_descs = []
        for char_id in room['ex_characters']:
            char = self.db['characters'][char_id]
            if char.get('id') is None:
                continue
            ignore_chance = 0.5
            if char.get('char_type') == 'object':
                # heavily downrank objects
                ignore_chance = 0.95
            if random.random() < ignore_chance:
                continue
            use_desc = char['name'] if char['is_plural'] == 0 \
                else random.choice(char['base_form'])
            use_desc = use_desc.lower()
            if use_desc in used_descs:
                continue
            used_descs.append(use_desc)
            create_characters.append([use_desc, char])
        for char_id in room['in_characters']:
            char = self.db['characters'][char_id]
            if char.get('id') is None:
                continue
            if char.get('char_type') == 'object':
                if random.random() < 0.95:
                    continue  # highly downrank objects
            use_desc = char['name'] if char['is_plural'] == 0 \
                else random.choice(char['base_form'])
            use_desc = use_desc.lower()
            if use_desc in used_descs:
                continue
            used_descs.append(use_desc)
            in_characters.append(use_desc)
            create_characters.append((use_desc, char))
        while len(create_characters) < 2:
            char = random.choice(self.chars_list)
            use_desc = char['name'] if char['is_plural'] == 0 \
                else random.choice(char['base_form'])
            use_desc = use_desc.lower()
            if use_desc in used_descs:
                continue
            used_descs.append(use_desc)
            create_characters.append((use_desc, char))

        random.shuffle(create_characters)
        player_characters = create_characters[:2]
        npc_characters = create_characters[2:]

        # Filter out characters that are in the room description
        # as they already exist in context
        npc_characters = [(ud, c) for (ud, c) in npc_characters
                          if ud not in in_characters]

        # only leave one npc character at most
        npc_characters = npc_characters[:1]

        # Add player characters to the world
        for use_desc, char in player_characters:
            g_id = g.add_node(use_desc, self.props_from_char(char),
                              is_player=True, uid=use_desc)
            g.move_object(g_id, room_gid)
            added_objs = []
            # add items to the player character
            for item_id in char['carrying_objects']:
                if random.random() > 0.5:
                    continue
                obj = self.db['objects'][item_id]
                use_desc = obj['name'] if obj['is_plural'] == 0 \
                    else random.choice(obj['base_form'])
                if len(use_desc.split(' ')) > 5:
                    continue  # Skip really long objects
                if use_desc.lower() in added_objs:
                    continue
                added_objs.append(use_desc.lower())
                obj_id = g.add_node(use_desc, self.props_from_obj(obj))
                g.move_object(obj_id, g_id)
            for item_id in char['wearing_objects']:
                if random.random() > 0.5:
                    continue
                obj = self.db['objects'][item_id]
                use_desc = obj['name'] if obj['is_plural'] == 0 \
                    else random.choice(obj['base_form'])
                if len(use_desc.split(' ')) > 5:
                    continue  # Skip really long objects
                if use_desc.lower() in added_objs:
                    continue
                added_objs.append(use_desc.lower())
                obj_id = g.add_node(use_desc, self.props_from_obj(obj))
                g.move_object(obj_id, g_id)
                g.set_prop(obj_id, 'equipped', 'wear')
            for item_id in char['wielding_objects']:
                if random.random() > 0.5:
                    continue
                obj = self.db['objects'][item_id]
                use_desc = obj['name'] if obj['is_plural'] == 0 \
                    else random.choice(obj['base_form'])
                if len(use_desc.split(' ')) > 5:
                    continue  # Skip really long objects
                if use_desc.lower() in added_objs:
                    continue
                added_objs.append(use_desc.lower())
                obj_id = g.add_node(use_desc, self.props_from_obj(obj))
                g.move_object(obj_id, g_id)
                g.set_prop(obj_id, 'equipped', 'wield')

        # add non player characters to the world
        for use_desc, char in npc_characters:
            g_id = g.add_node(use_desc, self.props_from_char(char),
                              uid=use_desc)
            g.move_object(g_id, room_gid)

        return g, room, player_characters


def main():
    '''Handles setting up and running a ParlAI-MTurk task by instantiating
    an MTurk manager and configuring it for the qa_data_collection task
    '''
    # Get relevant arguments
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '--light-unseen-rooms', default=False, type='bool',
        help='Launch using rooms from the unseen set rather than the seen')
    opt = argparser.parse_args()

    generator = GraphGenerator(opt, opt['light_unseen_rooms'])

    # Set the task name to be the folder name
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    # append the contents of task_config.py to the configuration
    opt.update(task_config)

    # Select an agent_id that worker agents will be assigned in their world
    mturk_agent_roles = ['worker_1', 'worker_2']

    # Set runtime to be an hour in case workers are slow
    opt['assignment_duration_in_seconds'] = 60 * 60

    # Instantiate an MTurkManager with the given options and a maximum number
    # of agents per world of 1 (based on the length of mturk_agent_ids)
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=mturk_agent_roles,
        use_db=True,
    )
    mturk_manager.setup_server(
        task_directory_path=os.path.dirname(os.path.abspath(__file__)))

    # Create an onboard_function, which will be run for workers who have
    # accepted your task and must be completed before they are put in the
    # queue for a task world.
    completed_agents = []

    def run_onboard(worker):
        nonlocal completed_agents
        if worker.worker_id in completed_agents:
            return
        else:
            world = LightChatOnboardingWorld(opt=opt, mturk_agent=worker)
            while not world.episode_done():
                world.parley()
            world.shutdown()
            completed_agents.append(worker.worker_id)
            print(worker.worker_id, 'took', world.turns, 'turns for onboarding')
            return world.prep_save_data([worker])

    # If we want to use the above onboard function, we can replace the below
    # with set_onboard_function(onboard_function=run_onboard)
    mturk_manager.set_onboard_function(onboard_function=run_onboard)

    qualification_id = \
        mturk_utils.find_qualification('adventure_chat_reject',
                                       opt['is_sandbox'],
                                       must_be_owned=False)
    print('Found qualification: ', qualification_id)

    try:
        # Initialize run information
        mturk_manager.start_new_run()

        # Set up the sockets and threads to recieve workers
        mturk_manager.ready_to_accept_workers()

        agent_qualifications = [{
            'QualificationTypeId': qualification_id,
            'Comparator': 'DoesNotExist',
            'RequiredToPreview': True
        }]

        # Create the hits as specified by command line arguments
        mturk_manager.create_hits(qualifications=agent_qualifications)

        # Check workers eligiblity acts as a filter, and should return
        # the list of all workers currently eligible to work on the task
        # Can be used to pair workers that meet certain criterea
        def check_workers_eligibility(workers):
            return workers

        eligibility_function = {
            'func': check_workers_eligibility,
            'multiple': True,
        }

        # Assign worker roles is used to determine what the role each worker
        # in the given worker list will play. Setting `id` to None will return
        # the worker to the pool rather than putting them in a given task,
        # which is useful for having tasks with different possible worker
        # counts.
        def assign_worker_roles(workers):
            workers[0].id = mturk_agent_roles[0]
            workers[1].id = mturk_agent_roles[1]

        # Define the task function, which will be run with workers that are
        # as the main task.
        global run_conversation

        def run_conversation(mturk_manager, opt, workers):
            # Create the task world
            g = None
            while g is None:
                try:
                    g, room, characters = generator.get_room()
                except Exception as e:
                    print('error when creating graph:', repr(e))
            world = LightChatTaskWorld(
                opt=opt,
                mturk_agents=workers,
                graph=g,
                room=room,
                characters=characters,
            )
            # run the world to completion
            while not world.episode_done():
                world.parley()

            # shutdown and review the work
            world.shutdown()
            world.review_work()

            # Return the contents for saving
            return world.prep_save_data(workers)

        # Begin the task, allowing mturk_manager to start running the task
        # world on any workers who connect
        mturk_manager.start_task(
            eligibility_function=eligibility_function,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )
    except BaseException:
        raise
    finally:
        # Any hits that aren't claimed or completed have to be shut down. Must
        # keep the world running until that point.
        mturk_manager.expire_all_unassigned_hits()
        # Shutdown the manager and free all related resources
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
