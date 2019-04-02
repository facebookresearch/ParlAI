#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.mturk.core.agents import \
    MTURK_DISCONNECT_MESSAGE, RETURN_MESSAGE, TIMEOUT_MESSAGE

import threading
import time


def is_disconnected(act):
    return 'text' in act and \
            act['text'] in [MTURK_DISCONNECT_MESSAGE, RETURN_MESSAGE,
                            TIMEOUT_MESSAGE]


class LightChatOnboardingWorld(MTurkOnboardWorld):
    '''Example onboarding world. Sends a message from the world to the
    worker and then exits as complete after the worker uses the interface
    '''
    instruction_act = {
        'id': 'System',
        'text':
            'Please attempt to take a turn given the setting and persona '
            'on the left. This is where your information will appear in '
            'the main task.',
        'task_data': {
            'base_name': 'bandit',
            'persona': (
                "I am an unforgiving bandit. Orcs have stolen my family away,"
                " and they treat me like trash. I one day want to have my "
                "revenge, be it through blood or gold. I'll try to get back at"
                " them any chance I get."
            ),
            'setting': (
                "You are in a bar. There isn't anything really special about "
                "it, and it's relatively quiet tonight. The air is somewhat "
                "tense. A sign above the bar says 'No Fighting' and something "
                "tells you that rule is taken pretty seriously. "
                "There is a beer here. There is an orc here. "
            ),
            'actions': [
                'wave at orc', 'steal coin purse from orc',
                'hit orc', 'give beer to orc', 'hug orc', 'get beer',
            ],
        }
    }

    bad_choice_act = {
        'id': 'System',
        'text':
            "Are you sure that's an appropriate action to take given your "
            "persona and the current setting? Try again."
    }

    too_short_act = {
        'id': 'System',
        'text':
            "Please generally speak in full sentences unless your persona "
            "implies that your character isn't able to."
    }

    block_act = {
        'id': 'System',
        'text':
            "Sorry, you've exceeded the maximum amount of tries to get the "
            "correct actions given your persona and the setting, and thus we "
            "don't believe you can complete the task correctly. Please return "
            "the HIT."
    }

    complete_act = {
        'id': 'System',
        'text':
            "Passed - We'll be pairing you with a partner. Hold on tight."
    }

    def block_loop(self):
        print('Worker {} failed onboarding'.format(self.mturk_agent.worker_id))
        self.mturk_agent.observe(self.block_act)
        self.mturk_agent.mturk_manager.soft_block_worker(
            self.mturk_agent.worker_id)
        act = self.mturk_agent.act()
        while not is_disconnected(act):
            self.mturk_agent.observe(self.block_act)
            act = self.mturk_agent.act()
        return True

    def parley(self):
        self.turns = 0
        self.mturk_agent.update_agent_id('Bandit')
        self.mturk_agent.observe(self.instruction_act)
        act = self.mturk_agent.act()  # first attempt, turns = 0
        data = act.get('task_data', {'action': None})
        while (data['action'] != 'steal coin purse from orc' or
                len(act['text']) < 4):
            if self.turns >= 2:  # if 3rd attempt wasn't correct, block worker
                self.block_loop()
                self.episodeDone = True
                return

            if is_disconnected(act):
                self.episodeDone = True
                return
            if data['action'] != 'steal coin purse from orc':
                self.mturk_agent.observe(self.bad_choice_act)
            else:
                self.mturk_agent.observe(self.too_short_act)
            self.turns += 1
            act = self.mturk_agent.act()
            data = act.get('task_data', {'action': None})

        self.mturk_agent.observe(self.complete_act)
        self.mturk_agent.onboarding_turns = self.turns
        self.episodeDone = True
        time.sleep(3)


class LightChatTaskWorld(MTurkTaskWorld):
    """
    World to demonstrate workers with assymetric roles. This task amounts
    to three rounds and then an evaluation step. It is purposefully created
    as a task to demo multiple views and has no other purpose.
    """

    collector_agent_id = 'Moderator'

    def __init__(self, opt, mturk_agents, graph, room, characters):
        self.mturk_agents = mturk_agents
        self.graph = graph
        self.room = room
        self.characters = characters
        # Extract the character names
        self.c_names = [characters[0][0].lower(), characters[1][0].lower()]
        self.graph_copy = graph.copy()
        self.mturk_agents[0].update_agent_id(self.c_names[0].capitalize())
        self.mturk_agents[1].update_agent_id(self.c_names[1].capitalize())
        self.episodeDone = False
        self.turns = 0
        self.acts = []
        self.graph.freeze(True)

    def get_context_actions_for(self, agent_name):
        self.graph.parse_exec(agent_name, 'look')
        self.graph.parse_exec(agent_name, 'inv')
        context = self.graph.get_text(agent_name).rstrip('\n')
        use_actions = [
            'get', 'put', 'drink', 'eat', 'steal', 'hit', 'hug', 'wear',
            'wield', 'drop', 'give', 'remove',
        ]
        actions = self.graph.get_possible_actions(
            agent_name, use_actions=use_actions)
        return context, actions

    def parley(self):
        if self.turns == 0:
            # Settings for both
            for i in [0, 1]:
                agent_name = self.c_names[i]
                self.graph.get_text(agent_name).rstrip('\n')
                context, actions = self.get_context_actions_for(agent_name)

                ad = {
                    'id': 'System',
                    'text':
                        "Your chat partner is: {}. "
                        "Please chat for 8 full turns "
                        "while pretending to be your assigned "
                        "persona in the assigned setting, both "
                        "provided in the 'context' tab of the left panel. "
                        "After the first turn you will need to respond within "
                        "5 minutes to avoid timing out."
                        "If unsure what to talk about, start "
                        "getting to know your partner's persona, or "
                        "discuss the setting. Take actions when/if it "
                        "feels appropriate to. "
                        "Any other characters in the room will not interact "
                        "with or respond to you, so while they may be good "
                        "things to talk about, don't interact with them."
                        "You can find the original instructions on the "
                        "'Task Instructions' tab to the left."
                        "".format(self.c_names[1-i]),
                    'task_data':
                        {
                            'base_name': self.c_names[i],
                            'persona': self.characters[i][1]['personas'][0],
                            'setting': context,
                            'actions': actions,
                        }
                }

                self.mturk_agents[i].observe(ad)

        if self.turns < 7:
            for i in [0, 1]:
                cur_agent = self.mturk_agents[i]
                other_agent = self.mturk_agents[1 - i]
                cur_agent_name = self.c_names[i]
                other_agent_name = self.c_names[1 - i]
                if self.turns == 0 and i == 0:
                    a = cur_agent.act()
                else:
                    a = cur_agent.act(timeout=5*60)

                self.acts.append(a)
                if is_disconnected(a):
                    self.episodeDone = True
                    return

                graph_action = a.get('task_data', {'action': ''})['action']
                observe_action = {
                    'id': cur_agent_name.capitalize(),
                    'text': a['text'],
                    'task_data': {}
                }
                if graph_action.startswith('gesture'):
                    observe_action['task_data']['action'] = graph_action
                elif graph_action != '':
                    # execute graph action
                    status, c_acts_text = self.graph.parse_exec(
                        cur_agent_name, graph_action)
                    if status:
                        self.graph.update_world()
                    # send new setting and actions to the actor
                    return_act_text = \
                        self.graph.get_text(cur_agent_name).rstrip('\n')
                    if status:
                        observe_action['task_data']['action'] = \
                            self.graph.get_text(other_agent_name).rstrip('\n')
                    context, actions = \
                        self.get_context_actions_for(cur_agent_name)
                    reflex_action = {
                        'id': 'System',
                        'text': return_act_text,
                        'task_data': {
                            'setting': context,
                            'actions': actions,
                        }
                    }
                    cur_agent.observe(reflex_action)

                    # Set the viewer context change and new actions
                    context, actions = \
                        self.get_context_actions_for(other_agent_name)
                    observe_action['task_data']['setting'] = context
                    observe_action['task_data']['actions'] = actions
                other_agent.observe(observe_action)
            self.turns += 1
        else:
            # evaluate
            ad = {
                'id': 'System',
                'text': "Thank you for the talk, the chat is complete.",
            }
            for agent in self.mturk_agents:
                agent.observe(ad)
            self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        # Parallel shutdown of agents
        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent
        threads = []
        for agent in self.mturk_agents:
            t = threading.Thread(target=shutdown_agent, args=(agent,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def review_work(self):
        # Can review the work here to accept or reject it
        pass

    def get_custom_task_data(self):
        # brings important data together for the task, to later be used for
        # creating the dataset. If data requires pickling, put it in a field
        # called 'needs-pickle'.
        return {
            'acts': self.acts,
            'room': self.room,
            'characters': self.characters,
            'needs-pickle': self.graph_copy,
        }
