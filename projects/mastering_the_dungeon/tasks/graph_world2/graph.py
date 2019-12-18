#!/usr/bin/env python3

##
## Copyright (c) Facebook, Inc. and its affiliates.
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##

import random
import numpy as np
from copy import deepcopy
from collections import Counter
import parlai.core.build_data as build_data
import torch
import os

DEFAULT_ROOMS = ['cavern', 'tower', 'forest']
DEFAULT_OBJECTS = [
    'rusty sword',
    'elven sword',
    'silver crown',
    'blue ring',
    'gold ring',
    'bread',
    'armor',
    'mace',
    'axe',
    'crossbow',
    'apple',
    'apple',
    'apple',
    'beer',
]
DEFAULT_OBJECT_PROPS = [
    'wieldable',
    'wieldable',
    'wearable',
    'wearable',
    'wearable',
    'food',
    'wearable',
    'wieldable',
    'wieldable',
    'wieldable',
    'food',
    'food',
    'food',
    'drink',
]
DEFAULT_CONTAINERS = ['treasure chest', 'leather pouch']
DEFAULT_AGENTS = ['dragon', 'orc', 'troll']
INIT_HEALTH = 1


def dedup(objects, props):
    visited = set()
    dedup_objects, dedup_props = [], []
    for i in range(len(objects)):
        if objects[i] in visited:
            continue
        visited.add(objects[i])
        dedup_objects.append(objects[i])
        dedup_props.append(props[i])
    return dedup_objects, dedup_props


DEDUP_OBJECTS, DEDUP_PROPS = dedup(DEFAULT_OBJECTS, DEFAULT_OBJECT_PROPS)


def rm(d, val):
    if val in d:
        del d[val]


class Graph(object):
    def __init__(self, opt):
        self._opt = opt
        self._node_to_edges = {}
        self._node_to_prop = {}
        self._node_contained_in = {}
        self._node_contains = {}
        self._node_follows = {}
        self._node_followed_by = {}
        self._node_npcs = (
            set()
        )  # non-player characters that we move during update_world func.
        self._node_to_desc = {}
        self._node_freeze = False
        self._cnt = 0
        self._save_fname = 'tmp.gw'
        self._node_to_text_buffer = {}

    def new_agent(self, id):
        self._node_to_text_buffer[id] = ''  # clear buffer

    def delete_node(self, id):
        rm(self._node_to_prop, id)
        if id in self._node_contains[self.location(id)]:
            self._node_contains[self.location(id)].remove(id)
        rm(self._node_contained_in, id)
        # all things inside this are zapped too
        os = deepcopy(self._node_contains[id])
        for o in os:
            self.delete_node(o)
        # now remove edges from other rooms
        for r in self._node_to_edges[id]:
            if r[0] == 'path_to':
                self._node_to_edges[r[1]].remove(['path_to', id])
        rm(self._node_to_edges, id)
        rm(self._node_to_text_buffer, id)
        rm(self._node_to_text_buffer, id)
        # remove all agents following this one:
        if id in self._node_followed_by:
            ags = deepcopy(self._node_followed_by[id])
            for a in ags:
                self.set_follow(a, None)
        rm(self._node_follows, id)
        if id in self._node_npcs:
            self._node_npcs.remove(id)

    def save_graph(self, fname):
        path = os.path.join(self._opt['datapath'], 'graph_world2')
        build_data.make_dir(path)
        if fname != '':
            self._save_fname = path + '/' + fname + '.gw2'
        else:
            fname = self._save_fname
        members = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr))
            and (not attr.startswith("__"))
            and (attr.startswith("_"))
        ]
        model = {}
        for m in members:
            model[m] = getattr(self, m)
        with open(fname, 'wb') as write:
            torch.save(model, write)

    def load_graph(self, fname):
        if fname != '':
            path = os.path.join(self._opt['datapath'], 'graph_world2')
            fname = path + '/' + fname + '.gw2'
        else:
            fname = self._save_fname
        if not os.path.isfile(fname):
            print("[graph file not found: " + fname + ']')
            return
        print("[loading graph: " + fname + ']')
        members = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr))
            and (not attr.startswith("__"))
            and (attr.startswith("_"))
        ]
        with open(fname, 'rb') as read:
            model = torch.load(read)
        for m in members:
            if m in model:
                setattr(self, m, model[m])
            else:
                print("[ loading: " + m + " is missing in file ]")
        self._save_fname = fname

    def freeze(self, freeze=None):
        if freeze is not None:
            self._node_freeze = freeze
        return self._node_freeze

    def node_path_to(self, id):
        rooms = self._node_to_edges[id]
        rooms = [r[1] for r in rooms if r[0] == 'path_to']
        return rooms

    def desc_to_node(
        self, desc, nearbyid=None, nearbytype=None, should_have=[], shouldnt_have=[]
    ):
        if nearbyid is not None:
            if nearbytype == 'path':
                o = self.node_path_to(self.location(nearbyid))
            elif nearbytype == 'carrying':
                o = self.node_contains(nearbyid)
            elif nearbytype == 'sameloc':
                o = self.node_contains(self.location(nearbyid))
            elif nearbytype == 'all':
                o1 = self.node_contains(nearbyid)
                o2 = self.node_contains(self.location(nearbyid))
                o3 = self.node_path_to(self.location(nearbyid))
                o = o1.union(o2).union(o3)
            else:
                o1 = self.node_contains(nearbyid)
                o2 = self.node_contains(self.location(nearbyid))
                o = o1.union(o2)
        else:
            o = set(self._node_to_desc.keys())

        o = [id for id in o if self._node_to_desc[id] == desc]
        o.sort()  # ensure deterministic order (e.g., which apple is got first)

        if len(o) == 0:
            return None  # No results given the nearby conditions

        for i in o:
            flag = True
            for prop in should_have:
                if not self.valid(i, prop):
                    flag = False
            for prop in shouldnt_have:
                if self.valid(i, prop):
                    flag = False
            if not flag:
                continue
            return i

        return False  # There are results given the nearby conditions, but they do not satisfy the property constraints.

    def copy(self):
        return deepcopy(self)

    def unique_hash(self):
        # TODO: make it independent of specific world settings
        # object_ids, agent_ids, and container_ids are set by construct_graph
        s = ''
        apple_s = []
        for id in self.object_ids + self.container_ids + self.agent_ids:
            cur_s = ''
            if not self.node_exists(id):
                cur_s += 'eaten'
            else:
                cur_s += self._node_contained_in[id]
                for prop in ['wielding', 'wearing', 'dead']:
                    if prop in self._node_to_prop[id]:
                        cur_s += prop
            if self.node_to_desc_raw(id) == 'apple':
                apple_s.append(cur_s)
            else:
                s += cur_s
        s += ''.join(sorted(apple_s))
        return s

    def __eq__(self, other):
        return self.unique_hash() == other.unique_hash()

    def add_node(self, desc, props):
        id = desc
        if id != 'dragon':
            id = id + "_" + str(self._cnt)
        self._cnt = self._cnt + 1
        if id in self._node_to_edges:
            return False
        self._node_to_edges[id] = []
        if type(props) == str:
            self._node_to_prop[id] = {}
            self._node_to_prop[id][props] = True
        else:
            self._node_to_prop[id] = {}
            for p in props:
                self._node_to_prop[id][p] = True
        self._node_contains[id] = set()
        self._node_to_desc[id] = desc

        if 'agent' in self._node_to_prop[id]:
            self.set_prop(id, 'health', INIT_HEALTH)

        return id

    def node_exists(self, id):
        return id in self._node_contained_in

    def set_desc(self, id, desc):
        self._node_to_desc[id] = desc

    def node_to_desc_raw(self, id):
        return self._node_to_desc[id]

    def node_to_desc(self, id, use_the=False):
        if id in self._node_to_desc:
            ent = self._node_to_desc[id]
            if self.has_prop(id, 'dead'):
                ent = 'dead ' + ent
            if self.has_prop(id, 'agent') or self.has_prop(id, 'object'):
                if use_the:
                    ent = 'the ' + ent
                else:
                    ent = (
                        'an ' + ent
                        if ent[0] in ['a', 'e', 'i', 'o', 'u']
                        else 'a ' + ent
                    )
            elif self.has_prop(id, 'room'):
                ent = 'the ' + ent
            return ent
        else:
            return id

    def add_edge(self, id1, edge, id2):
        if [edge, id2] not in self._node_to_edges[id1]:
            self._node_to_edges[id1].insert(0, [edge, id2])

    def add_path_to(self, id1, id2):
        if id1 == id2:
            return False
        self.add_edge(id1, 'path_to', id2)
        self.add_edge(id2, 'path_to', id1)
        return True

    def is_path_to(self, id1, id2):
        rooms = self._node_to_edges[id1]
        rooms = [r[1] for r in rooms if r[0] == 'path_to' and r[1] == id2]
        if len(rooms) == 1:
            return True
        else:
            return False

    def add_contained_in(self, id1, id2):
        if id1 in self._node_contained_in:
            i_am_in = self._node_contained_in[id1]
            self._node_contains[i_am_in].remove(id1)
        self._node_contained_in[id1] = id2
        self._node_contains[id2].add(id1)
        return True

    def node_contained_in(self, id):
        return self._node_contained_in[id]

    def set_follow(self, id1, id2):
        if id1 in self._node_follows:
            i_follow = self._node_follows[id1]
            self._node_followed_by[i_follow].remove(id1)
        if id2 is not None:
            self._node_follows[id1] = id2
            if id2 not in self._node_followed_by:
                self._node_followed_by[id2] = set()
            self._node_followed_by[id2].add(id1)
            return True
        else:
            if id1 in self._node_follows:
                self._node_follows.pop(id1)

    def valid(self, id, prop):
        if not id in self._node_to_prop:
            return False
        if not self.has_prop(id, prop):
            return False
        return True

    def messages_in_same_room_as(self, agent, txt):
        room = self._node_contained_in[agent]
        agents = self._node_contains[room]
        agents = [a for a in agents if self.has_prop(a, 'agent') and a != agent]
        if len(agents) > 0:
            for a in agents:
                self.send_msg(a, txt)

    def has_prop(self, id, prop):
        if id in self._node_to_prop:
            if prop in self._node_to_prop[id]:
                return self._node_to_prop[id][prop]
        return False

    def set_prop(self, id, prop, val=True):
        if id in self._node_to_prop:
            self._node_to_prop[id][prop] = val

    def inc_prop(self, id, prop, val=1):
        if id in self._node_to_prop:
            if prop not in self._node_to_prop[id]:
                self.set_prop(id, prop, 0)
            if type(self._node_to_prop[id][prop]) != int:
                self.set_prop(id, prop, 0)
            self._node_to_prop[id][prop] += val

    def delete_prop(self, id, prop):
        if id in self._node_to_prop:
            if prop in self._node_to_prop[id]:
                del self._node_to_prop[id][prop]

    def location(self, thing):
        return self._node_contained_in[thing]

    def room(self, thing):
        id = self._node_contained_in[thing]
        while not self.has_prop(id, 'room'):
            id = self._node_contained_in[id]
        return id

    def node_contains(self, loc):
        if loc in self._node_contains:
            return self._node_contains[loc]
        else:
            return set()

    def send_msg(self, agentid, txt):
        if agentid in self._node_to_text_buffer:
            self._node_to_text_buffer[agentid] += txt

    #### ----------------------------------------------------------------
    # TODO: Ideally, all functions below do not use the graph structure directly,
    # but only the accessor functions (i.e. should not use self._node_* ).

    def die(self, id):
        if not self.valid(id, 'agent'):
            return False
        self.send_msg(id, 'You are dead!!!!!!!\n')
        agent_desc = self.node_to_desc(id, use_the=True).capitalize()
        self.messages_in_same_room_as(id, agent_desc + ' is dead!!!!\n')
        self.set_follow(id, None)
        self.set_prop(id, 'dead')
        # self.delete_prop(id, 'agent')
        # self.set_prop(id, 'object')

    def move_agent(self, agent_id, to_txt=None, to_id=None):
        if to_id is None:
            to_id = self.desc_to_node(
                to_txt, nearbyid=agent_id, nearbytype='path', should_have=['room']
            )
            if to_id is None or to_id == False:
                return False
        if not self.valid(agent_id, 'agent'):
            return False
        # if not self.valid(to_id, 'room'): return False
        from_id = self.location(agent_id)
        to_desc = self.node_to_desc(to_id)
        from_desc = self.node_to_desc(from_id)
        if self.is_path_to(from_id, to_id):
            agent_desc = self.node_to_desc(agent_id, use_the=True).capitalize()
            self.messages_in_same_room_as(
                agent_id, agent_desc + ' leaves towards ' + to_desc + '.\n'
            )
            self.add_contained_in(agent_id, to_id)
            agent_desc = self.node_to_desc(agent_id).capitalize()
            self.messages_in_same_room_as(
                agent_id, agent_desc + ' enters from ' + from_desc + '.\n'
            )
            self.send_msg(agent_id, self.look(agent_id))
        else:
            return False
        if agent_id in self._node_followed_by:
            for ppl in self._node_followed_by[agent_id]:
                room2 = self.location(ppl)
                if from_id == room2:
                    self.send_msg(ppl, 'You follow. ')
                    self.move_agent(ppl, to_id=to_id)
        return True

    def follow(self, agent_id, params):
        if not self.valid(agent_id, 'agent'):
            return False
        thing = ' '.join(params)
        if thing == 'off' or thing == '':
            if agent_id in self._node_follows:
                thing_id = self._node_follows[agent_id]
                self.set_follow(agent_id, None)
                thing_desc = self.node_to_desc(thing_id, use_the=True)
                s = 'You stop following ' + thing_desc + '.\n'
                self.send_msg(agent_id, s)
                agent_desc = self.node_to_desc(agent_id, use_the=True).capitalize()
                s = agent_desc + ' stops following you.\n'
                self.send_msg(thing_id, s)
                return True
            else:
                s = 'You are not following anyone.\n'
                self.send_msg(agent_id, s)
                return True
        thing_id = self.desc_to_node(thing, nearbyid=agent_id, nearbytype='sameloc')
        if not thing_id or not self.valid(thing_id, 'agent'):
            return False
        room1_id = self.room(agent_id)
        room2_id = self.room(thing_id)
        thing_desc = self.node_to_desc(thing_id, use_the=True)
        if room1_id != room2_id:
            self.send_msg(agent_id, thing_desc + " is not here.")
            return True
        self.set_follow(agent_id, thing_id)
        s = 'You are following the ' + thing_desc + '.\n'
        self.send_msg(agent_id, s)
        agent_desc = self.node_to_desc(agent_id, use_the=True).capitalize()
        s = agent_desc + ' is following you.\n'
        self.send_msg(thing_id, s)
        return True

    def get_object(self, agent_id, obj_txt):
        if not self.valid(agent_id, 'agent'):
            return False
        obj_id = self.desc_to_node(
            obj_txt, nearbyid=agent_id, nearbytype='sameloc', should_have=['object']
        )
        if obj_id is None:
            self.send_msg(agent_id, 'It is not here.\n')
            return False
        if obj_id == False:
            self.send_msg(agent_id, 'It is not an object.\n')
            return False
        self.add_contained_in(obj_id, agent_id)
        self.send_msg(agent_id, 'Done.\n')
        return True

    def drop_object(self, agent_id, obj_txt):
        if not self.valid(agent_id, 'agent'):
            return False
        obj_id = self.desc_to_node(
            obj_txt,
            nearbyid=agent_id,
            nearbytype='carrying',
            should_have=['object'],
            shouldnt_have=['wearing', 'wielding'],
        )
        if obj_id is None:
            self.send_msg(agent_id, "You do not have that.\n")
            return False
        if obj_id == False:
            self.send_msg(agent_id, 'You must unwield/remove it before dropping it.\n')
            return False
        room_id = self.node_contained_in(agent_id)
        self.add_contained_in(obj_id, room_id)
        self.send_msg(agent_id, 'Done.\n')
        return True

    def put(self, agent_id, params):
        if not self.valid(agent_id, 'agent'):
            return False
        if not len(params) == 2:
            return False
        obj_id = self.desc_to_node(
            params[0],
            nearbyid=agent_id,
            nearbytype='carrying',
            should_have=['object'],
            shouldnt_have=['wearing', 'wielding'],
        )
        if obj_id is None:
            self.send_msg(agent_id, 'You do not have that.\n')
            return False
        if obj_id == False:
            self.send_msg(
                agent_id,
                'You must unwield/remove it before putting it into containers.\n',
            )
            return False
        receiver_id = self.desc_to_node(
            params[1], nearbyid=agent_id, should_have=['container']
        )
        if receiver_id is None:
            self.send_msg(agent_id, 'That is not here.\n')
            return False
        if receiver_id == False:
            self.send_msg(agent_id, 'It is not a container.\n')
            return False
        self.add_contained_in(obj_id, receiver_id)
        receiver_desc = self.node_to_desc(receiver_id, use_the=True)
        self.send_msg(
            agent_id,
            "You put "
            + self.display_node_list([obj_id])
            + " in "
            + receiver_desc
            + '.\n',
        )
        return True

    def get_from(self, agent_id, params):
        if not self.valid(agent_id, 'agent'):
            return False
        if not len(params) == 2:
            return False
        victim_id = self.desc_to_node(
            params[1], nearbyid=agent_id, should_have=['container']
        )
        if victim_id is None:
            self.send_msg(agent_id, 'That is not here.\n')
            return False
        if victim_id == False:
            self.send_msg(agent_id, 'It is not a container.\n')
            return False
        # if not self.valid(victim_id, 'container'): return False
        obj_id = self.desc_to_node(
            params[0], nearbyid=victim_id, nearbytype='carrying', should_have=['object']
        )
        if obj_id is None:
            self.send_msg(agent_id, "You couldn't find it.\n")
            return False
        if obj_id == False:
            self.send_msg(agent_id, 'It is not an object.\n')
            return False
        self.add_contained_in(obj_id, agent_id)
        agent_desc = self.node_to_desc(agent_id, use_the=True).capitalize()
        victim_desc = self.node_to_desc(victim_id, use_the=True)
        self.send_msg(
            agent_id,
            "You took "
            + self.display_node_list([obj_id])
            + " from "
            + victim_desc
            + '.\n',
        )
        self.send_msg(
            victim_id,
            agent_desc
            + " took the "
            + self.display_node_list([obj_id])
            + " from you.\n",
        )
        return True

    def give(self, agent_id, params):
        if not self.valid(agent_id, 'agent'):
            return False
        if not len(params) == 2:
            return False
        obj_id = self.desc_to_node(
            params[0],
            nearbyid=agent_id,
            nearbytype='carrying',
            should_have=['object'],
            shouldnt_have=['wearing', 'wielding'],
        )
        if obj_id is None:
            self.send_msg(agent_id, 'You do not have that.\n')
            return False
        if obj_id == False:
            self.send_msg(
                agent_id, 'You must remove/unwield it before giving it to others.\n'
            )
            return False
        receiver_id = self.desc_to_node(
            params[1],
            nearbyid=agent_id,
            nearbytype='sameloc',
            should_have=['agent'],
            shouldnt_have=['dead'],
        )
        if receiver_id == agent_id:
            return False
        if receiver_id is None:
            self.send_msg(agent_id, 'They are not here.\n')
            return False
        if receiver_id == False:
            self.send_msg(agent_id, 'They are not alive agents.\n')
            return False
        self.add_contained_in(obj_id, receiver_id)
        agent_desc = self.node_to_desc(agent_id, use_the=True).capitalize()
        receiver_desc = self.node_to_desc(receiver_id, use_the=True)
        self.send_msg(
            agent_id,
            "You gave "
            + self.display_node_list([obj_id])
            + " to "
            + receiver_desc
            + '.\n',
        )
        self.send_msg(
            receiver_id,
            agent_desc + " gave you " + self.display_node_list([obj_id]) + '\n',
        )
        return True

    def take(self, agent_id, params):
        if not self.valid(agent_id, 'agent'):
            return False
        if not len(params) == 2:
            return False
        victim_id = self.desc_to_node(
            params[1], nearbyid=agent_id, nearbytype='sameloc', should_have=['agent']
        )
        if victim_id == agent_id:
            return False
        if victim_id is None:
            self.send_msg(agent_id, 'They are not here.\n')
            return False
        if victim_id == False:
            self.send_msg(agent_id, 'It is not an agent.\n')
            return False
        obj_id = self.desc_to_node(
            params[0], nearbyid=victim_id, nearbytype='carrying', should_have=['object']
        )
        if obj_id is None:
            self.send_msg(agent_id, 'They do not have that.\n')
            return False
        if obj_id == False:
            self.send_msg(agent_id, 'It is not an object.\n')
            return False
        self.add_contained_in(obj_id, agent_id)
        agent_desc = self.node_to_desc(agent_id, use_the=True).capitalize()
        victim_desc = self.node_to_desc(victim_id, use_the=True)
        self.send_msg(
            agent_id,
            "You took "
            + self.display_node_list([obj_id])
            + " from "
            + victim_desc
            + '.\n',
        )
        self.send_msg(
            victim_id,
            agent_desc
            + " took the "
            + self.display_node_list([obj_id])
            + " from you.\n",
        )
        return True

    def hit_agent(self, agent_id, victim_txt, victim_id=None):
        if victim_id is None:
            victim_id = self.desc_to_node(
                victim_txt,
                nearbyid=agent_id,
                nearbytype='sameloc',
                should_have=['agent'],
                shouldnt_have=['dead'],
            )
            if victim_id is None:
                self.send_msg(agent_id, 'They are not here.\n')
                return False
            if victim_id == False:
                self.send_msg(agent_id, "You can't hit that.\n")
                return False
        if not self.valid(agent_id, 'agent'):
            return False
        agent_desc = self.node_to_desc(agent_id, use_the=True).capitalize()
        victim_desc = self.node_to_desc(victim_id, use_the=True)
        self.send_msg(agent_id, "You hit " + victim_desc + '! ')
        self.send_msg(victim_id, agent_desc + " attacked you! ")
        energy = self.has_prop(victim_id, 'health')
        if type(energy) == bool:
            energy = INIT_HEALTH
        energy = max(0, energy - 1)
        if energy == 0:
            self.die(victim_id)
        elif energy < 4 and energy > 0:
            self.send_msg(victim_id, 'You are ' + self.health(victim_id) + '.\n')
        self.set_prop(victim_id, 'health', energy)
        return True

    def wear(self, agent_id, thing):
        thing_id = self.desc_to_node(
            thing,
            nearbyid=agent_id,
            nearbytype='carrying',
            should_have=['wearable'],
            shouldnt_have=['wearing'],
        )
        if thing_id is None:
            self.send_msg(agent_id, "You do not have that.\n")
            return False
        if thing_id == False:
            # self.send_msg(agent_id, 'You are wearing that or it is not wearable.\n')
            self.send_msg(agent_id, "You can't do that.\n")
            return False
        self.set_prop(thing_id, 'wearing')
        self.send_msg(agent_id, "Done.\n")
        self.inc_prop(agent_id, 'armour', 1)
        return True

    def wield(self, agent_id, thing):
        thing_id = self.desc_to_node(
            thing,
            nearbyid=agent_id,
            nearbytype='carrying',
            should_have=['wieldable'],
            shouldnt_have=['wielding'],
        )
        if thing_id is None:
            self.send_msg(agent_id, "You do not have that.\n")
            return False
        if thing_id == False:
            # self.send_msg(agent_id, 'You are wielding that or it is not wieldable.\n')
            self.send_msg(agent_id, "You can't do that.\n")
            return False
        self.set_prop(thing_id, 'wielding')
        self.send_msg(agent_id, "Done.\n")
        self.inc_prop(agent_id, 'weapon', 1)
        return True

    def remove(self, agent_id, thing):
        thing_id_wear = self.desc_to_node(
            thing, nearbyid=agent_id, nearbytype='carrying', should_have=['wearing']
        )
        thing_id_wield = self.desc_to_node(
            thing, nearbyid=agent_id, nearbytype='carrying', should_have=['wielding']
        )
        thing_id = thing_id_wear or thing_id_wield
        if thing_id is None:
            self.send_msg(agent_id, "You do not have that.\n")
            return False
        if thing_id == False:
            self.send_msg(agent_id, 'You are not using that.\n')
            return False
        if self.has_prop(thing_id, 'wielding'):
            self.set_prop(thing_id, 'wielding', None)
            self.inc_prop(agent_id, 'weapon', -1)
        else:
            self.set_prop(thing_id, 'wearing', None)
            self.inc_prop(agent_id, 'armour', -1)
        self.send_msg(agent_id, "Done.\n")

        return True

    def ingest(self, agent_id, cmd, thing):
        if cmd == 'eat':
            thing_id = self.desc_to_node(
                thing, nearbyid=agent_id, nearbytype='carrying', should_have=['food']
            )
        else:
            thing_id = self.desc_to_node(
                thing, nearbyid=agent_id, nearbytype='carrying', should_have=['drink']
            )
        if thing_id is None:
            self.send_msg(agent_id, "You do not have that.\n")
            return False
        if thing_id == False:
            if cmd == 'eat':
                self.send_msg(agent_id, "You can't eat that.\n")
            else:
                self.send_msg(agent_id, "You can't drink that.\n")
            return False
        self.delete_node(thing_id)
        self.send_msg(agent_id, "Yum.\n")
        energy = self.has_prop(agent_id, 'health')
        if energy == False:
            energy = INIT_HEALTH
        if energy < 8:
            energy = energy + 1
        self.set_prop(agent_id, 'health', energy)
        return True

    def create(self, agent_id, params):
        # -- create commands: --
        # create room kitchen  -> creates room with path from this room
        # create path kitchen  -> create path to that room from this one
        # create agent orc
        # create object ring
        # create container box
        # create [un]freeze
        # create reset/load/save [fname]
        # create rename <node> <value>
        # create delete <node>
        # create set_prop orc to health=5
        if not self.valid(agent_id, 'agent'):
            return False
        room_id = self.room(agent_id)
        all = ' '.join(params)
        txt = ' '.join(params[1:])
        if not (all == 'save' or all == 'load' or all == 'freeze' or all == 'unfreeze'):
            if txt == '':
                return False
        if params[0] == 'save':
            self.save_graph(txt)
            self.send_msg(agent_id, "[ saved: " + self._save_fname + ']\n')
            return True
        if params[0] == 'load' or params[0] == 'reset':
            self.load_graph(txt)
            self.send_msg(agent_id, "[ loaded: " + self._save_fname + ']\n')
            return True
        if params[0] == 'freeze':
            self.freeze(True)
            self.send_msg(agent_id, "Frozen.\n")
            return True
        if params[0] == 'unfreeze':
            self.freeze(False)
            self.send_msg(agent_id, "Unfrozen.\n")
            return True
        if params[0] == 'delete' or params[0] == 'del' or params[0] == 'rm':
            id = self.desc_to_node(txt, nearbyid=agent_id, nearbytype='all')
            if id == False:
                return False
            self.delete_node(id)
            self.send_msg(agent_id, "Deleted.\n")
            return True
        if params[0] == 'rename':
            params = self.split_params(params[1:], 'to')
            to_id = self.desc_to_node(params[0], nearbyid=agent_id, nearbytype='all')
            if to_id == False:
                return False
            self.set_desc(to_id, params[1])
            self.send_msg(agent_id, "Done.\n")
            return True
        if params[0] == 'agent':
            new_id = self.add_node(txt, params[0])
            self.add_contained_in(new_id, room_id)
            self._node_npcs.add(new_id)
            self.send_msg(agent_id, "Done.\n")
            return True
        if params[0] == 'room':
            new_id = self.add_node(txt, params[0])
            self.add_contained_in(new_id, room_id)
            self.add_path_to(new_id, room_id)
            self.send_msg(agent_id, "Done.\n")
            return True
        if params[0] == 'set_prop':
            params = self.split_params(params[1:], 'to')
            print(params)
            to_id = self.desc_to_node(params[0], nearbyid=agent_id, nearbytype='all')
            if to_id == False:
                return False
            key = params[1]
            value = True
            if '=' in key:
                sp = key.split('=')
                if len(sp) != 2:
                    return False
                key = sp[0]
                value = sp[1]
                if value == 'True':
                    value = True
                try:
                    value = int(value)
                except ValueError:
                    pass
            self.set_prop(to_id, key, value)
            self.send_msg(agent_id, "Done.\n")
            return True
        if (
            params[0] == 'container'
            or params[0] == 'object'
            or params[0] == 'food'
            or params[0] == 'drink'
        ):
            new_id = self.add_node(txt, 'object')
            self.add_contained_in(new_id, room_id)
            self.set_prop(new_id, params[0])
            self.send_msg(agent_id, "Done.\n")
            return True
        if params[0] == 'path':
            to_id = self.desc_to_node(txt)
            if to_id == False:
                return False
            self.add_path_to(to_id, room_id)
            self.send_msg(agent_id, "Done.\n")
            return True
        return False

    def display_room_edges(self, roomid, third_person=False):
        s = ''
        rooms = self._node_to_edges[roomid]
        rooms = [r[1] for r in rooms if r[0] == 'path_to']
        room = self.node_to_desc(roomid)
        if third_person:
            s += '{} is connected to '.format(room).capitalize()
        else:
            if len(rooms) == 1:
                s += 'There is a path to '
            else:
                s += 'There are paths to '
        s += self.display_node_list(rooms)
        s += '.\n'
        return s

    def display_room_objects(self, roomid, third_person=False):
        s = ''
        objects = self.node_contains(roomid)
        objects = [o for o in objects if self.has_prop(o, 'object')]
        # import pdb; pdb.set_trace()
        room = self.node_to_desc(roomid)
        if len(objects) == 0:
            s += '{} is empty.\n'.format(room).capitalize()
        else:
            if third_person:
                s += 'In {} there is '.format(room)
            else:
                s += 'There is '
            s += self.display_node_list(objects)
            if third_person:
                s += '\n'
            else:
                s += ' here.\n'
        return s

    def display_room_agents(self, me, room, third_person=False):
        s = ''
        agents = self.node_contains(room)
        agents = [a for a in agents if self.has_prop(a, 'agent') and a != me]
        if len(agents) > 0:
            for a in agents:
                desc = self.node_to_desc(a).capitalize()
                s += desc + ' is here.\n'
        return s

    def get_text(self, agent):
        txt = ''
        if agent in self._node_to_text_buffer:
            txt = self._node_to_text_buffer[agent]
        self._node_to_text_buffer[agent] = ''  # clear buffer
        return txt

    def cnt_obj(self, obj, c):
        cnt = c[obj]
        if cnt == 1:
            return obj
        else:
            words = obj.split(' ')
            f = [
                'two',
                'three',
                'four',
                'five',
                'six',
                'seven',
                'eight',
                'nine',
                'a lot of',
            ]
            rep = ['a', 'an', 'the']
            cnt = cnt - 2
            if cnt > 8:
                cnt = 8
            cnt = f[cnt]
            if words[0] in rep:
                return cnt + ' ' + ' '.join(words[1:]) + 's'
            else:
                return cnt + ' ' + ' '.join(words) + 's'

    def display_node_list(self, l):
        if len(l) == 0:
            return 'nothing'
        l = [self.node_to_desc(ent) for ent in l]
        if len(l) == 1:
            return l[0]
        c = Counter(l)
        l = set(l)
        s = ''
        cnt = 0
        for o in l:
            s += self.cnt_obj(o, c)
            if len(l) > 2 and cnt < len(l) - 1:
                s += ','
            s += ' '
            cnt = cnt + 1
            if cnt == len(l) - 1:
                s += 'and '
        return s.rstrip(' ')

    def display_node(self, id):
        s = ''
        if len(self.node_contains(id)) > 0:
            s = (
                s
                + id
                + ' contains '
                + self.display_node_list(self.node_contains(id))
                + '\n'
            )
        return s

    def examine(self, agent_id, thing):
        thing_id = self.desc_to_node(thing, nearbyid=agent_id)
        if thing_id is None:
            self.send_msg(agent_id, "That is not here.\n")
            return True
        s = ''
        if self.has_prop(thing_id, 'agent'):
            s = self.inventory(thing_id, agent_id)
        else:
            object_ids = self.node_contains(thing_id)
            object_ids = [o for o in object_ids if self.has_prop(o, 'object')]
            thing_desc = self.node_to_desc(thing_id, use_the=True).capitalize()
            inside_txt = ' contains '
            if len(object_ids) == 0:
                s += thing_desc + inside_txt + 'nothing.\n'
            else:
                s += thing_desc + inside_txt
                s += self.display_node_list(object_ids)
                s += '.\n'
        self.send_msg(agent_id, s)
        return True

    def inventory(self, id, id2=None):
        s = ''
        carry_ids = []
        wear_ids = []
        wield_ids = []
        for o in self.node_contains(id):
            if self.has_prop(o, 'wearing'):
                wear_ids.append(o)
            elif self.has_prop(o, 'wielding'):
                wield_ids.append(o)
            else:
                carry_ids.append(o)
        if id2 is not None:
            thing_desc = self.node_to_desc(id, use_the=True).capitalize() + ' is'
        else:
            thing_desc = 'You are'
        if len(carry_ids) == 0:
            s += thing_desc + ' carrying nothing.\n'
        else:
            s += thing_desc + ' carrying ' + self.display_node_list(carry_ids) + '.\n'
        if len(wear_ids) > 0:
            s += thing_desc + ' wearing ' + self.display_node_list(wear_ids) + '.\n'
        if len(wield_ids) > 0:
            s += thing_desc + ' wielding ' + self.display_node_list(wield_ids) + '.\n'
        return s

    def health(self, id):
        health = self.has_prop(id, 'health')
        if health == None or health == False:
            health = 1
        if health > 8:
            health = 8
        f = [
            'dead',
            'on the verge of death',
            'very weak',
            'weak',
            'ok',
            'good',
            'strong',
            'very strong',
            'nigh on invincible',
        ]
        return f[health]

    def look(self, id):
        room = self.location(id)
        s = 'You are in {}.\n'.format(self.node_to_desc(room))
        s += self.display_room_agents(id, room)
        s += self.display_room_objects(room)
        s += self.display_room_edges(room)
        return s

    def split_params(self, params, word):
        return ' '.join(params).split(' {} '.format(word))

    def help(self):
        txt = (
            '----------------------------------\n'
            + 'Commands:\n'
            + 'look\n'
            + 'examine <thing>\n'
            + 'go <room>\n'
            + 'get/drop <object>\n'
            + 'eat/drink <object>\n'
            + 'wear/remove <object>\n'
            + 'wield/unwield <object>\n'
            + 'follow <agent>\n'
            + 'hit <agent>\n'
            + 'put <object> in <container>\n'
            + 'get <object> from <container>\n'
            + 'give <object> to <agent>\n'
            + 'take <object> from <agent>\n'
            + '----------------------------------\n'
        )
        return txt

    def get_possible_actions(self, my_agent_id='dragon'):
        # TODO: make it independent of specific world settings
        actions = []
        dragon_id = my_agent_id
        if self.valid(dragon_id, 'dead'):
            return actions
        current_room_id = self.node_contained_in(dragon_id)
        for id in self.node_path_to(current_room_id):
            actions.append('go {}'.format(self.node_to_desc_raw(id)))
        for id in self.object_ids + self.container_ids:
            if not self.node_exists(id):
                continue
            desc = self.node_to_desc_raw(id)
            if self.node_contained_in(id) == current_room_id:
                actions.append('get {}'.format(desc))
            if self.node_contained_in(id) == dragon_id:
                if not self.valid(id, 'wearing') and not self.valid(id, 'wielding'):
                    actions.append('drop {}'.format(desc))
                    for container_id in self.container_ids:
                        if container_id != id and (
                            self.node_contained_in(container_id) == dragon_id
                            or self.node_contained_in(container_id) == current_room_id
                        ):
                            actions.append(
                                'put {} in {}'.format(
                                    desc, self.node_to_desc_raw(container_id)
                                )
                            )
                    for agent_id in self.agent_ids:
                        if (
                            agent_id != dragon_id
                            and not self.valid(agent_id, 'dead')
                            and self.node_contained_in(agent_id) == current_room_id
                        ):
                            actions.append(
                                'give {} to {}'.format(
                                    desc, self.node_to_desc_raw(agent_id)
                                )
                            )
                if self.valid(id, 'food'):
                    actions.append('eat {}'.format(desc))
                if self.valid(id, 'drink'):
                    actions.append('drink {}'.format(desc))
                if self.valid(id, 'wearable') and not self.valid(id, 'wearing'):
                    actions.append('wear {}'.format(desc))
                if self.valid(id, 'wearing'):
                    actions.append('remove {}'.format(desc))
                if self.valid(id, 'wieldable') and not self.valid(id, 'wielding'):
                    actions.append('wield {}'.format(desc))
                if self.valid(id, 'wielding'):
                    actions.append('unwield {}'.format(desc))

            container_id = self.node_contained_in(id)
            if (
                self.valid(container_id, 'agent')
                and container_id != dragon_id
                and self.node_contained_in(container_id) == current_room_id
            ):
                actions.append(
                    'take {} from {}'.format(desc, self.node_to_desc_raw(container_id))
                )
            if self.valid(container_id, 'container') and (
                self.node_contained_in(container_id)
                == self.node_contained_in(dragon_id)
                or self.node_contained_in(container_id) == dragon_id
            ):
                actions.append(
                    'get {} from {}'.format(desc, self.node_to_desc_raw(container_id))
                )

        for id in self.agent_ids:
            if (
                id != dragon_id
                and not self.valid(id, 'dead')
                and self.node_contained_in(id) == current_room_id
            ):
                actions.append('hit {}'.format(self.node_to_desc_raw(id)))

        return list(set(actions))

    @staticmethod
    def parse_static(inst):
        inst = inst.lower().strip().split()
        symb_points = []
        for i, symb in enumerate(inst):
            if symb in [
                'go',
                'get',
                'drop',
                'hit',
                'examine',
                'ex',
                'give',
                'take',
                'follow',
                'put',
                'create',
                'c',
                'eat',
                'drink',
                'wear',
                'wield',
                'unwield',
                'remove',
                'look',
                'actions',
                'hints',
            ]:
                symb_points.append(i)
        symb_points.append(len(inst))
        return inst, symb_points

    @staticmethod
    def filter_actions(inst):
        ret_actions = []
        inst, symb_points = Graph.parse_static(inst)
        for i in range(len(symb_points) - 1):
            j, k = symb_points[i], symb_points[i + 1]
            if inst[j] in [
                'go',
                'get',
                'drop',
                'hit',
                'give',
                'take',
                'put',
                'eat',
                'drink',
                'wear',
                'wield',
                'unwield',
                'remove',
            ]:
                ret_actions.append(' '.join(inst[j:k]))
        return ' '.join(ret_actions)

    def parse(self, inst):
        return Graph.parse_static(inst)

    def parse_exec(self, agentid, inst=None):
        """ATTENTION: even if one of the actions is invalid, all actions before that will still be executed (the world state will be changed)!"""
        if inst is None:
            inst = agentid
            agentid = 'dragon'

        if self.has_prop(agentid, 'dead'):
            self.send_msg(agentid, "You are dead, you can't do anything, sorry.")
            return True
        inst, symb_points = self.parse(inst)
        if len(inst) == 1 and (
            inst[0] == 'a' or inst[0] == 'actions' or inst[0] == 'hints'
        ):
            self.send_msg(
                agentid,
                '\n'.join(sorted(self.get_possible_actions()))
                + '\ninventory\nlook\nexamine <object>\n',
            )
            return True
        if len(inst) == 1 and (
            inst[0] == 'i' or inst[0] == 'inv' or inst[0] == 'inventory'
        ):
            self.send_msg(agentid, self.inventory(agentid))
            return True
        if len(inst) == 1 and (inst[0] == 'health' or inst[0] == 'status'):
            self.send_msg(agentid, 'You are feeling ' + self.health(agentid) + '.\n')
            return True
        if len(inst) == 1 and (inst[0] == 'look' or inst[0] == 'l'):
            self.send_msg(agentid, self.look(agentid))
            return True
        if len(inst) == 1 and (inst[0] == 'wait' or inst[0] == 'w'):
            self.send_msg(agentid, 'You wait. ')
            return True
        if len(inst) == 1 and (inst[0] == 'help'):
            self.send_msg(agentid, self.help())
            return True
        if len(symb_points) <= 1 or symb_points[0] != 0:
            return False
        for i in range(len(symb_points) - 1):
            j, k = symb_points[i], symb_points[i + 1]
            params = inst[j + 1 : k]
            if inst[j] == 'go':
                room_name = ' '.join(inst[j + 1 : k])
                if not self.move_agent(agentid, room_name):
                    return False
            elif inst[j] == 'eat' or inst[j] == 'drink':
                thing = ' '.join(inst[j + 1 : k])
                if not self.ingest(agentid, inst[j], thing):
                    return False
            elif inst[j] == 'wear':
                thing = ' '.join(inst[j + 1 : k])
                if not self.wear(agentid, thing):
                    return False
            elif inst[j] == 'wield':
                thing = ' '.join(inst[j + 1 : k])
                if not self.wield(agentid, thing):
                    return False
            elif inst[j] == 'remove' or inst[j] == 'unwield':
                thing = ' '.join(inst[j + 1 : k])
                if not self.remove(agentid, thing):
                    return False
            elif inst[j] == 'put':
                params = self.split_params(params, 'in')
                if not self.put(agentid, params):
                    return False
            elif inst[j] == 'create' or inst[j] == 'c':
                if not self.create(agentid, params):
                    return False
            elif inst[j] == 'get':
                if 'from' in inst[j + 1 : k]:
                    # get X from Y
                    params = self.split_params(params, 'from')
                    if not self.get_from(agentid, params):
                        return False
                else:
                    # get from loc
                    object_name = ' '.join(inst[j + 1 : k])
                    if not self.get_object(agentid, object_name):
                        return False
            elif inst[j] == 'drop':
                object_name = ' '.join(inst[j + 1 : k])
                if not self.drop_object(agentid, object_name):
                    return False
            elif inst[j] == 'examine' or inst[j] == 'ex':
                thing = ' '.join(inst[j + 1 : k])
                if not self.examine(agentid, thing):
                    return False
            elif inst[j] == 'hit':
                victim = ' '.join(inst[j + 1 : k])
                if not self.hit_agent(agentid, victim):
                    return False
            elif inst[j] == 'give':
                params = self.split_params(params, 'to')
                if not self.give(agentid, params):
                    return False
            elif inst[j] == 'take':
                params = self.split_params(params, 'from')
                if not self.take(agentid, params):
                    return False
            elif inst[j] == 'follow':
                if not self.follow(agentid, params):
                    return False
            else:
                return False
                # assert False
        return True

    def update_world(self):
        # move all the agents and junk, unless world frozen
        if self.freeze():
            return
        for agent_id in self._node_npcs:
            if self.has_prop(agent_id, 'dead'):
                continue
            # random movement for npcs..
            locs = self.node_path_to(self.room(agent_id))
            loc = locs[random.randint(0, len(locs) - 1)]
            act = 'go ' + self.node_to_desc(loc)
            self.move_agent(agent_id, to_id=loc)
            if random.randint(0, 100) < 50:
                act = 'hit dragon'
            self.parse_exec(agent_id, act)


def construct_graph(opt, graph_file=None, save_file=None, freeze=True):
    g = Graph(opt)
    if graph_file is None or not g.load_graph(graph_file):
        edge_p = opt['edge_p']
        seed = opt['seed']
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
        room_ids = [''] * len(DEFAULT_ROOMS)
        for i, v in enumerate(DEFAULT_ROOMS):
            room_ids[i] = g.add_node(v, 'room')
        # ensure that the graph is connected
        N = len(room_ids)
        perm = np.random.permutation(N)
        for i in range(N - 1):
            g.add_path_to(room_ids[perm[i]], room_ids[perm[i + 1]])
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < edge_p:
                    g.add_path_to(room_ids[i], room_ids[j])

        container_ids = []
        for i, v in enumerate(DEFAULT_CONTAINERS):
            id = g.add_node(v, ['object', 'container'])
            container_ids.append(id)
            g.add_contained_in(id, room_ids[random.randint(0, N - 1)])

        agent_ids = []
        for i, v in enumerate(DEFAULT_AGENTS):
            id = g.add_node(v, 'agent')
            g.add_contained_in(id, room_ids[random.randint(0, N - 1)])
            agent_ids.append(id)

        all_ids = room_ids + container_ids + agent_ids
        M = len(all_ids)

        object_ids = []
        for ind, o in enumerate(DEFAULT_OBJECTS):
            id = g.add_node(o, 'object')
            g.set_prop(id, DEFAULT_OBJECT_PROPS[ind])
            room_id = all_ids[random.randint(0, M - 1)] if o != 'apple' else all_ids[0]
            # assign all apples to room 0 just to avoid ambiguity of "apple -> cavern"
            g.add_contained_in(id, room_id)
            object_ids.append(id)

        g.room_ids, g.container_ids, g.agent_ids, g.object_ids = (
            room_ids,
            container_ids,
            agent_ids,
            object_ids,
        )

        if save_file is not None:
            g.save_graph(save_file)

        if freeze:
            g._node_freeze = True

        g.new_agent('dragon')

    return g
