#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTICE NOTICE NOTICE NOTICE NOTICE
# Development of this graph engine is very much still a work in progress,
# and it is only provided in this form to be used for the light_dialog
# task as collected in the 'Learning to Speak and Act in a Fantasy
# Text Adventure Game' paper
# NOTICE NOTICE NOTICE NOTICE NOTICE

# import callbacks as Callbacks  - Callbacks not yet finalized
from collections import Counter
from copy import deepcopy
import random


INIT_HEALTH = 2

CONSTRAINTS = {}
GRAPH_FUNCTIONS = {}


def rm(d, val):
    if val in d:
        del d[val]


def format_observation(self, graph, viewing_agent, action, telling_agent=None,
                       is_constraint=False):
    """Return the observation text to display for an action"""
    use_actors = action['actors']
    if is_constraint:
        use_actors = use_actors[1:]
    descs = [graph.node_to_desc(a, from_id=action['room_id'], use_the=True)
             for a in use_actors]
    try:
        # Replace viewer with you
        i = use_actors.index(viewing_agent)
        descs[i] = 'you'
    except BaseException:
        pass

    if telling_agent is not None:
        try:
            # Replace telling agent with me or I depending
            i = use_actors.index(telling_agent)
            if i == 0:
                descs[0] = 'I'
            else:
                descs[i] = 'me'
        except BaseException:
            pass

    # Package descriptions and determine the format
    descs[0] = descs[0].capitalize()
    if 'add_descs' in action:
        descs += action['add_descs']
    if is_constraint:
        descs = [action['actors'][0]] + descs
    return self.get_action_observation_format(action, descs).format(*descs)


# Useful constraint sets
def is_held_item(item_idx):
    return [
        {'type': 'is_type', 'in_args': [item_idx], 'args': [['object']]},
        {'type': 'no_prop', 'in_args': [item_idx], 'args': ['equipped']},
    ]


def is_equipped_item(item_idx):
    return [
        {'type': 'is_type', 'in_args': [item_idx], 'args': [['object']]},
        {'type': 'has_prop', 'in_args': [item_idx], 'args': ['equipped']},
    ]


# Functions
class GraphFunction(object):
    """Initial description of function should only contain what the arguments
    are as set up by init, establishing a number to type relationship like:
    [Actor, item actor is carrying, container in same room as actor]
    """
    def __init__(self, function_name, possible_arg_nums, arg_split_words,
                 arg_find_type, arg_constraints, func_constraints,
                 callback_triggers):
        """Create a new graph function
        args:
        function name - name or array of aliases for the function
        possible_arg_nums - number or array of valid arg numbers for
            determining if given args are valid or for breaking text into
            a valid number of arguments before they're parsed into descs
        arg_split_words - array of words to be used to split input args
        arg_find_type - array of args for desc_to_nodes to use to find the
            argument at the given index, following the form
            {'type': <search type>, 'from': <arg idx of reference>}. If a
            list is given for each argument use the same element for each
        arg_constraints - constraints on whether or not found arguments are
            valid to be matched during the parsing step. Form is array of
            arrays of constraint types
        func_constraints - constraints on whether a function will pass with
            the given args, format is array of constraint types
        callback_triggers - callback hook names and argument idxs for making
            calls to callback functions upon completion of this function call
        """
        self.name = function_name
        self.possible_arg_nums = possible_arg_nums
        self.arg_split_words = arg_split_words
        self.arg_find_type = arg_find_type
        self.arg_constraints = arg_constraints
        self.func_constraints = func_constraints
        self.callback_triggers = callback_triggers
        self.formats = {'failed': '{1} couldn\'t do that'}

    format_observation = format_observation

    def valid_args(self, graph, args):
        if 'ALL' in self.possible_arg_nums and len(args) >= 2:
            return True
        if len(args) not in self.possible_arg_nums:
            return False
        args = [a.lower() for a in args]
        loc = graph.location(args[0])
        # convert the arguments supplied to text form
        text_args = [
            graph.node_to_desc_raw(i, from_id=loc).lower() for i in args]
        # ensure that those are valid from the parser perspective
        valid, args, _args = self.parse_descs_to_args(graph, text_args)
        if not valid:
            return False
        # ensure that they pass the function constraints
        valid, _cons_passed, _failure = self.evaluate_constraint_set(
            graph,
            args,
            self.func_constraints,
        )
        return valid

    def handle(self, graph, args):
        # if Callbacks.try_overrides(self, graph, args):
        #     return True
        is_constrained, _cons_passed, failure_action = \
            self.evaluate_constraint_set(graph, args, self.func_constraints)
        if not is_constrained:
            func_failure_action = self.get_failure_action(graph, args)
            # What failed?
            graph.send_action(args[0], func_failure_action)
            # Why it failed
            graph.send_action(args[0], failure_action)
            return False
        # callback_args = deepcopy(args)
        is_successful = self.func(graph, args)
        # if is_successful:
        #     Callbacks.handle_callbacks(self, graph, args)
        #     # TODO move callbacks call functionality elsewhere so other things
        #     # can call callbacks
        #     for trigger in self.callback_triggers:
        #         for arg in trigger['args']:
        #             if arg >= len(callback_args):
        #                 break  # Don't have enough args for this callback
        #             acting_agent = callback_args[arg]
        #             callbacks = graph.get_prop(acting_agent, 'callbacks', {})
        #             if trigger['action'] in callbacks:
        #                 use_args = [callback_args[i] for i in trigger['args']]
        #                 Callbacks.handle(graph, callbacks[trigger['action']],
        #                                  use_args)
        return is_successful

    def func(self, graph, args):
        """
        Execute action on the given graph
        Notation for documentation should be in the form of an action with <>'s
        around the incoming arguments in the order they are passed in. Optional
        arguments can be denoted by putting the clause in []'s.
        ex:
        <Actor> does thing with <object> [and other <object>]
        """
        raise NotImplementedError

    def get_action_observation_format(self, action, descs):
        """Given the action, return text to be filled by the parsed args
        Allowed to alter the descriptions if necessary
        Allowed to use the descriptions to determine the correct format as well
        """
        if action['name'] in self.formats:
            return self.formats[action['name']]
        raise NotImplementedError

    def get_failure_action(self, graph, args, spec_fail='failed'):
        """Returns an action for if the resolution of this function fails"""
        return {'caller': self.get_name(), 'name': spec_fail, 'actors': args,
                'room_id': graph.location(args[0])}

    def get_find_fail_text(self, arg_idx):
        """Text displayed when an arg find during parse_text_to_args call
        fails to find valid targets for the function.
        Can be overridden to provide different messages for different failed
        arguments.
        """
        return 'That isn\'t here'

    def get_improper_args_text(self, num_args):
        return 'Couldn\'t understand that. Try help for help with commands'

    def evaluate_constraint_set(self, graph, args, constraints):
        """Check a particular set of arguments against the given constraints
        returns (True, cons_passed, None) if the constraints all pass
        returns (False, cons_passed, failure_action) if any constraint fails
        """
        cons_passed = 0
        failure_action = None
        for constraint in constraints:
            con_args = [args[i] for i in ([0] + constraint['in_args'])]
            con_args += constraint['args']
            con_obj = CONSTRAINTS[constraint['type']]
            if con_obj.evaluate_constraint(graph, con_args):
                cons_passed += 1
            else:
                spec_fail = constraint.get('spec_fail', 'failed')
                failure_action = \
                    con_obj.get_failure_action(graph, con_args, spec_fail)
                break
        if failure_action is None:
            return True, cons_passed, None
        else:
            return False, cons_passed, failure_action

    def check_possibilites_vs_constraints(self, graph, output_args, arg_idx,
                                          possible_ids):
        """Iterate through the given possible ids for a particular arg index
        and see if any match all the constraints for that argument

        Returns (True, valid_ID) on success
        Returns (False, violator_action) on failure
        """
        constraints = self.arg_constraints[arg_idx]
        most_constraints = -1
        final_failure_action = None
        for pos_id in possible_ids:
            output_args[arg_idx] = pos_id
            success, cons_passed, failure_action = \
                self.evaluate_constraint_set(graph, output_args, constraints)
            if success:
                return True, pos_id
            else:
                output_args[arg_idx] = None
                if cons_passed > most_constraints:
                    most_constraints = cons_passed
                    final_failure_action = failure_action
        return False, final_failure_action

    def parse_text_to_args(self, graph, actor_id, text):
        """Breaks text into function arguments based on this function's
        delimiters
        Returns:
        (None, display text, None) if parsing failed to find valid candidates
        (True, arg_ids, canonical_text_args) if parsing and constraints succeed
        (False, Failure action, matched_args)
                if parsing succeeds but constraints fail
        """
        actor_id = actor_id.lower()
        c_arg = []
        descs = [actor_id]
        for word in text:
            if word.lower() in self.arg_split_words:
                descs.append(' '.join(c_arg))
                c_arg = []
            else:
                c_arg.append(word)
        if len(c_arg) != 0:
            descs.append(' '.join(c_arg))
        if len(descs) not in self.possible_arg_nums and \
                'ALL' not in self.possible_arg_nums:
            return (None, self.get_improper_args_text(len(descs)), None)
        return self.parse_descs_to_args(graph, descs)

    def try_callback_override_args(self, graph, args):
        output_args = [None] * (len(args))
        output_args[0] = args[0]
        args = [arg.lower() for arg in args]
        # While there are still arguments to fill
        while None in output_args:
            for arg_idx, arg_id in enumerate(output_args):
                if arg_id is not None:  # This was processed in an earlier pass
                    continue
                nearbyid = output_args[0]

                # Get the possible IDs from the graph in the area
                arg_text = args[arg_idx]
                possible_ids = graph.desc_to_nodes(arg_text, nearbyid,
                                                   'all+here')
                # can't have reflexive actions with the same arg
                for had_id in output_args:
                    if had_id in possible_ids:
                        possible_ids.remove(had_id)
                if len(possible_ids) == 0:
                    return None, None, None

                output_args[arg_idx] = possible_ids[0]

        # Determine canonical arguments
        # if Callbacks.has_valid_override(self, graph, output_args):
        #     loc = graph.location(output_args[0])
        #     text_args = \
        #         [graph.node_to_desc_raw(i, from_id=loc) for i in output_args]
        #     return True, output_args, text_args
        return None, None, None

    def parse_descs_to_args_helper(self, graph, args, output_args,
                                   arg_find_idx=None):
        args = [a.lower() for a in args]
        for arg_idx, arg_id in enumerate(output_args):
            if arg_id is not None:  # This was processed in an earlier pass
                continue
            arg_find_type = self.arg_find_type[arg_idx]
            if arg_find_idx is not None:
                arg_find_type = arg_find_type[arg_find_idx]
            nearby_idx = arg_find_type['from']
            # we cant process a from when the nearby hasn't been found
            if output_args[nearby_idx] is None:
                continue
            nearbyid = output_args[nearby_idx]

            # Make sure we have all the constraints filled
            constraints = self.arg_constraints[arg_idx]
            have_all_args = True
            for constraint in constraints:
                for alt_id_idx in constraint['in_args']:
                    if arg_idx == alt_id_idx:
                        continue  # We plan to fill this slot with a guess
                    if output_args[alt_id_idx] is None:
                        # We don't have all the constraints for this
                        have_all_args = False
            if not have_all_args:
                continue

            # Get the possible IDs from the graph in the area
            arg_text = args[arg_idx]
            possible_ids = graph.desc_to_nodes(arg_text, nearbyid,
                                               arg_find_type['type'])
            # can't have reflexive actions with the same arg
            for had_id in output_args:
                if had_id in possible_ids:
                    possible_ids.remove(had_id)
            res = None
            while len(possible_ids) > 0:
                # See if any match the constraints
                success, res = self.check_possibilites_vs_constraints(
                    graph, output_args, arg_idx, possible_ids)
                if success:
                    output_args[arg_idx] = res
                    possible_ids.remove(res)
                    success_again, res, _args = \
                        self.parse_descs_to_args_helper(graph, args,
                                                        output_args,
                                                        arg_find_idx)
                    if success_again:
                        return True, None, output_args  # success!
                elif res is not None:
                    return False, res, args  # failure case, no matching option
                else:
                    break
            if res is not None:
                return False, res, args  # failure case, no matching option
            return None, self.get_find_fail_text(arg_idx), None  # failed find
        return True, '', output_args  # base case, all filled

    def parse_descs_to_args(self, graph, args):
        """iterates through the args and the constraints, solving dependency
        order on the fly.
        """
        # TODO improve performance by instead constructing a dependency graph
        # at runtime to avoid repeat passes through the loop and to prevent
        # circular dependencies
        args = [a.lower() for a in args]
        success, output_args, text_args = \
            self.try_callback_override_args(graph, args)
        if success:
            return success, output_args, text_args
        output_args = [None] * (len(args))
        output_args[0] = args[0]
        if 'ALL' in self.possible_arg_nums:
            output_args[1] = ' '.join(args[1:])
            return True, output_args, output_args
        args = [arg.lower() for arg in args]

        # While there are still arguments to fill
        if type(self.arg_find_type[0]) is list:
            arg_find_idx = 0
            success = False
            while arg_find_idx < len(self.arg_find_type[0]) and not success:
                output_args = [None] * (len(args))
                output_args[0] = args[0]
                success, res, output_args = \
                    self.parse_descs_to_args_helper(graph, args,
                                                    output_args, arg_find_idx)
                arg_find_idx += 1
        else:
            success, res, output_args = \
                self.parse_descs_to_args_helper(graph, args, output_args)

        if success:
            # Determine canonical arguments
            loc = graph.location(output_args[0])
            text_args = \
                [graph.node_to_desc_raw(i, from_id=loc) for i in output_args]
            return True, output_args, text_args
        else:
            return success, res, output_args

    def get_name(self):
        """Get the canonical name of this function"""
        name = self.name
        if type(name) is list:
            name = name[0]
        return name

    def get_canonical_form(self, graph, args):
        """Get canonical form for the given args on this function"""
        if 'ALL' in self.possible_arg_nums:
            # All is reserved for functions that take all arguments
            return ' '.join(args)
        loc = graph.location(args[0])
        text_args = \
            [graph.node_to_desc_raw(i, from_id=loc) for i in args]
        if len(args) == 1:
            return self.get_name()
        if len(self.arg_split_words) == 0 or len(args) == 2:
            return self.get_name() + ' ' + text_args[1]
        return (self.get_name() + ' ' +
                (' ' + self.arg_split_words[0] + ' ').join(text_args[1:]))

    def get_reverse(self, graph, args):
        """Function to parse the graph and args and return the reverse
        function and args"""
        return None, []


class SayFunction(GraphFunction):
    """Output all the following arguments [Actor, raw text]"""
    def __init__(self):
        super().__init__(
            function_name='say',
            possible_arg_nums=['ALL'],
            arg_split_words=[],
            arg_find_type=[{}],
            arg_constraints=[
                [],  # no constraints on the actor
            ],
            func_constraints=[],
            callback_triggers=[
                {'action': 'said', 'args': [0, 1]},
            ]
        )

    def func(self, graph, args):
        """<Actor> sends message history to room [or specified <recipient>]"""
        actor = args[0]
        words = args[1].strip()
        if words.startswith('"') and words.endswith('"'):
            words = words[1:-1]
        room_id = graph.location(actor)
        agent_ids = graph.get_room_agents(room_id)
        said_action = {
            'caller': self.get_name(), 'name': 'said', 'room_id': room_id,
            'actors': [actor], 'present_agent_ids': agent_ids,
            'content': words,
        }
        graph.broadcast_to_room(said_action, exclude_agents=[actor])
        return True

    def format_observation(self, graph, viewing_agent, action,
                           telling_agent=None):
        """Parse the observations to recount, then return them"""
        actor = action['actors'][0]
        content = action['content']
        if viewing_agent == actor:
            actor_text = 'You'
        else:
            actor_text = graph.node_to_desc(actor, use_the=True).capitalize()
        if content[-1] not in ['.', '?', '!']:
            content += '.'
        return '{} said "{}"\n'.format(actor_text, content)


class TellFunction(GraphFunction):
    """Output all the following arguments [Actor, Target, raw text]"""
    def __init__(self):
        super().__init__(
            function_name='tell',
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{}, {'type': 'sameloc+players', 'from': 0}],
            arg_constraints=[[], []],
            func_constraints=[],
            callback_triggers=[
                {'action': 'told', 'args': [0, 1, 2]},
            ]
        )

    def func(self, graph, args):
        """<Actor> sends message history to room [or specified <recipient>]"""
        actor = args[0]
        words = args[2]
        room_id = graph.location(actor)
        agent_ids = graph.get_room_agents(room_id)
        said_action = {
            'caller': self.get_name(), 'name': 'said', 'room_id': room_id,
            'actors': args[:2], 'present_agent_ids': agent_ids,
            'content': words,
        }
        graph.broadcast_to_room(said_action, exclude_agents=[actor])
        graph._last_tell_target[actor] = args[1]
        return True

    def format_observation(self, graph, viewing_agent, action,
                           telling_agent=None):
        """Parse the observations to recount, then return them"""
        actor = action['actors'][0]
        target = action['actors'][1]
        content = action['content']
        involved = False
        if viewing_agent == actor:
            actor_text = 'You'
            involved = True
        else:
            actor_text = graph.node_to_desc(actor, use_the=True).capitalize()
        if viewing_agent == target:
            target_text = 'you'
            involved = True
        else:
            target_text = graph.node_to_desc(target, use_the=True)
        if involved:
            if content[-1] not in ['.', '?', '!']:
                content += '.'
            return '{} told {} "{}"\n'.format(actor_text, target_text, content)
        return '{} whispered something to {}.'.format(actor_text, target_text)

    def valid_args(self, graph, args):
        args = [a.lower() for a in args]
        loc = graph.location(args[0])
        try:
            # convert the arguments supplied to text form
            text_args = [graph.node_to_desc_raw(i, from_id=loc).lower()
                         for i in args[:2]]
            # ensure that those are valid from the parser perspective
            valid, args, _args = self.parse_descs_to_args(graph, text_args)
            if not valid:
                return False
            # ensure that they pass the function constraints
            valid, _cons_passed, _failure = self.evaluate_constraint_set(
                graph,
                args,
                self.func_constraints,
            )
            return valid
        except BaseException:
            return False

    def evaluate_constraint_set(self, graph, args, constraints):
        """Can't talk to yourself, only constraint"""
        if args[0] == args[1]:
            return False, 0, {
                'caller': None,
                'room_id': self.location(args[0]),
                'txt': 'If you talk to yourself, you might seem crazy...',
            }
        return True, 0, None

    def parse_text_to_args(self, graph, actor_id, text):
        """Breaks text into function arguments based on tell target "something"
        Returns:
        (None, display text, None) if parsing failed to find valid candidates
        (True, arg_ids, canonical_text_args) if parsing and constraints succeed
        (False, Failure action, matched_args)
                if parsing succeeds but constraints fail
        """
        actor_id = actor_id.lower()
        descs = [actor_id]
        text = ' '.join(text)
        text = text.lower()
        if '"' not in text or not text.endswith('"'):
            return None, 'Be sure to use quotes to speak (").', None
        try:
            target = text.split('"')[0].strip()
            content = text.split('"')[1:]
            content = '"'.join(content).strip('"')
            descs = [actor_id, target]
            valid, arg_ids, c_args = self.parse_descs_to_args(graph, descs)
            if not valid:
                return None, "You don't know what you're talking to", None
            arg_ids.append(content)
            c_args.append(content)
            return valid, arg_ids, c_args
        except Exception:
            # import sys, traceback
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            # traceback.print_tb(exc_traceback)
            # print(repr(e))
            # TODO the above in debug mode? would be useful
            return None, "You couldn't talk to that now", None

    def get_name(self):
        """Get the canonical name of this function"""
        name = self.name
        if type(name) is list:
            name = name[0]
        return name

    def get_canonical_form(self, graph, args):
        """Get canonical form for the given args on this function"""
        loc = graph.location(args[0])
        target_text = graph.node_to_desc_raw(args[1], from_id=loc)
        return 'tell {} "{}"'.format(target_text, args[2])


class UseFunction(GraphFunction):
    def __init__(self):
        super().__init__(
            function_name='use',
            possible_arg_nums=[3],
            arg_split_words=['with'],
            arg_find_type=[{},  # no constraints on the actor
                           {'type': 'carrying', 'from': 0},
                           {'type': 'all+here', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                is_held_item(1),
                [{'type': 'is_type', 'in_args': [2],
                  'args': [['object'], 'Both have to be objects. ']}],
            ],
            func_constraints=[],
            callback_triggers=[
                {'action': 'agent_used_with', 'args': [0, 1, 2]},
            ]
        )
        self.formats = {'cant_use_with': "{0} couldn't find out how to "
                                         "use {1} with {2}. ",
                        'failed': '{0} couldn\'t do that. '}

    def get_find_fail_text(self, arg_idx):
        if arg_idx == 1:
            return "You don't seem to have that. "
        return 'That isn\'t here. '

    def func(self, graph, args):
        """<Actor> uses <held object> with <other object>"""
        # held_obj = args[1]
        # callbacks = graph.get_prop(held_obj, 'callbacks', {})
        # if 'use_with' in callbacks:
        #     rets = Callbacks.handle(graph, callbacks['use_with'], args)
        #     if False in rets:
        #         return False  # This action was unsuccessfully handled
        #     if True in rets:
        #         return True  # This action was successfully handled

        # other_obj = args[2]
        # callbacks = graph.get_prop(other_obj, 'callbacks', {})
        # if 'use_with' in callbacks:
        #     rets = Callbacks.handle(graph, callbacks['use_with'], args)
        #     if False in rets:
        #         return False  # This action was unsuccessfully handled
        #     if True in rets:
        #         return True  # This action was successfully handled

        agent_id = args[0]
        room_id = graph.location(agent_id)
        cant_use_action = {'caller': self.get_name(), 'name': 'cant_use_with',
                           'room_id': room_id, 'actors': args}
        graph.send_action(args[0], cant_use_action)
        return False

    def get_action_observation_format(self, action, descs):
        if action['name'] == 'used_with':
            return '{0} {1} '
        return super().get_action_observation_format(action, descs)


class MoveAgentFunction(GraphFunction):
    """[actor, destination room or path name from actor's room]"""
    def __init__(self):
        super().__init__(
            function_name='go',
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{},  # no constraints on the actor
                           {'type': 'path', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                [
                    {'type': 'is_type', 'in_args': [1], 'args': ['room']},
                    {'type': 'is_locked', 'in_args': [1], 'args': [False]},
                    {'type': 'not_location_of', 'in_args': [1], 'args': []},
                ]
            ],
            func_constraints=[
                {'type': 'fits_in', 'in_args': [0, 1], 'args': []},
            ],
            callback_triggers=[
                {'action': 'moved_to', 'args': [0, 1]},
                {'action': 'agent_entered', 'args': [0, 1]},
            ]
        )
        self.formats = {'left': '{0} left towards {1}. ',
                        'arrived': '{0} arrived from {1}. ',
                        'failed': '{0} can\'t do that. ',
                        'follow': '{0} follow. ',
                        'followed': '{0} followed {1} here.'}

    def follow_agent(self, graph, args, followed):
        """handler for all following actions"""
        agent_id = args[0]
        old_room_id = graph.location(agent_id)
        followers = graph.get_followers(agent_id)
        room_id = args[1]
        leave_action = {'caller': self.get_name(), 'name': 'left',
                        'room_id': old_room_id,
                        'actors': [agent_id, room_id]}
        graph.broadcast_to_room(leave_action)
        graph.move_object(agent_id, room_id)

        # Prepare and send action for listing room contents
        is_return = room_id in graph._visited_rooms[agent_id]
        if is_return or not graph.has_prop(room_id, 'first_desc'):
            if not graph.has_prop(room_id, 'short_desc'):
                room_desc = \
                    graph.get_classed_prop(room_id, 'desc', agent_id, None)
            else:
                room_desc = \
                    graph.get_classed_prop(room_id, 'short_desc', agent_id)
        else:
            room_desc = graph.get_classed_prop(room_id, 'first_desc', agent_id)
        if is_return:
            agent_ids, agent_descs = \
                graph.get_nondescribed_room_agents(room_id)
            object_ids, object_descs = \
                graph.get_nondescribed_room_objects(room_id)
            _room_ids, room_descs = graph.get_room_edges(room_id)
        else:
            agent_ids, agent_descs = [], []
            object_ids, object_descs = [], []
            room_descs = []
        list_room_action = {
            'caller': 'look', 'name': 'list_room', 'room_id': room_id,
            'actors': [args[0]], 'agent_ids': agent_ids,
            'present_agent_ids': agent_ids, 'object_ids': object_ids,
            'agent_descs': agent_descs, 'object_descs': object_descs,
            'room_descs': room_descs, 'room_desc': room_desc,
            'returned': is_return,
        }
        graph.send_action(args[0], list_room_action)
        graph._last_rooms[agent_id] = old_room_id
        graph._visited_rooms[agent_id].add(old_room_id)
        graph._visited_rooms[agent_id].add(room_id)

        # Handle follows
        for other_agent in list(followers):
            room2 = graph.location(other_agent)
            if old_room_id == room2:
                follow_action = {'caller': self.get_name(), 'name': 'follow',
                                 'room_id': old_room_id,
                                 'actors': [other_agent]}
                graph.send_msg(other_agent, 'You follow.\n', follow_action)
                new_args = args.copy()
                new_args[0] = other_agent
                self.follow_agent(graph, new_args, agent_id)

        # Now that everyone is here, handle arrivals
        arrive_action = {'caller': self.get_name(), 'name': 'followed',
                         'room_id': room_id,
                         'actors': [agent_id, followed]}
        graph.broadcast_to_room(arrive_action, exclude_agents=followers)
        # Callbacks.handle_callbacks(self, graph, args)
        return True

    def func(self, graph, args):
        """<Actor> moves to <location>"""
        agent_id = args[0]
        old_room_id = graph.location(agent_id)
        followers = graph.get_followers(agent_id)
        room_id = args[1]
        leave_action = {'caller': self.get_name(), 'name': 'left',
                        'room_id': old_room_id,
                        'actors': [agent_id, room_id]}
        graph.broadcast_to_room(leave_action)
        graph.move_object(agent_id, room_id)

        # Prepare and send action for listing room contents
        is_return = room_id in graph._visited_rooms[agent_id]
        if is_return or not graph.has_prop(room_id, 'first_desc'):
            if not graph.has_prop(room_id, 'short_desc'):
                room_desc = \
                    graph.get_classed_prop(room_id, 'desc', agent_id, None)
            else:
                room_desc = \
                    graph.get_classed_prop(room_id, 'short_desc', agent_id)
        else:
            room_desc = graph.get_classed_prop(room_id, 'first_desc', agent_id)
        if is_return or not graph.has_prop(room_id, 'first_desc'):
            agent_ids, agent_descs = \
                graph.get_nondescribed_room_agents(room_id)
            object_ids, object_descs = \
                graph.get_nondescribed_room_objects(room_id)
            _room_ids, room_descs = graph.get_room_edges(room_id)
        else:
            agent_ids, agent_descs = [], []
            object_ids, object_descs = [], []
            room_descs = []
        list_room_action = {
            'caller': 'look', 'name': 'list_room', 'room_id': room_id,
            'actors': [args[0]], 'agent_ids': agent_ids,
            'present_agent_ids': agent_ids, 'object_ids': object_ids,
            'agent_descs': agent_descs, 'object_descs': object_descs,
            'room_descs': room_descs, 'room_desc': room_desc,
            'returned': is_return,
        }
        graph.send_action(args[0], list_room_action)
        graph._last_rooms[agent_id] = old_room_id
        graph._visited_rooms[agent_id].add(old_room_id)
        graph._visited_rooms[agent_id].add(room_id)

        # Handle follows
        for other_agent in list(followers):
            room2 = graph.location(other_agent)
            if old_room_id == room2:
                follow_action = {'caller': self.get_name(), 'name': 'follow',
                                 'room_id': old_room_id,
                                 'actors': [other_agent]}
                graph.send_msg(other_agent, 'You follow.\n', follow_action)
                new_args = args.copy()
                new_args[0] = other_agent
                self.follow_agent(graph, new_args, agent_id)

        # Now that everyone is here, handle arrivals
        arrive_action = {'caller': self.get_name(), 'name': 'arrived',
                         'room_id': room_id,
                         'actors': [agent_id, old_room_id]}
        graph.broadcast_to_room(arrive_action, exclude_agents=followers)
        return True

    def parse_text_to_args(self, graph, actor_id, text):
        """Wraps the super parse_text_to_args to be able to handle the special
        case of "go back"
        """
        actor_id = actor_id.lower()
        if len(text) == 0:
            return super().parse_text_to_args(graph, actor_id, text)
        elif text[-1].lower() == 'back':
            if graph._last_rooms[actor_id] is None:
                return None, 'Back where exactly? You\'ve just started!', None
            text[-1] = graph.node_to_desc_raw(
                graph._last_rooms[actor_id], from_id=graph.location(actor_id))
        return super().parse_text_to_args(graph, actor_id, text)

    def get_find_fail_text(self, arg_idx):
        return 'Where is that? You don\'t see it'

    def get_action_observation_format(self, action, descs):
        if 'past' not in action:
            if descs[0] == 'You':
                return ''  # Don't tell someone where they just went
        return super().get_action_observation_format(action, descs)

    REPLACEMENTS = {'e': 'east', 'w': 'west', 'n': 'north', 's': 'south',
                    'u': 'up', 'd': 'down'}

    def parse_descs_to_args(self, graph, args):
        # Replace shortcuts with full words for better matching
        if args[1] in self.REPLACEMENTS:
            args[1] = self.REPLACEMENTS[args[1]]
        return super().parse_descs_to_args(graph, args)

    # TODO implement reverse for go


class FollowFunction(GraphFunction):
    """[Actor, agent actor should follow in same room]"""
    def __init__(self):
        super().__init__(
            function_name=['follow'],
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{}, {'type': 'sameloc', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                [{'type': 'is_type', 'in_args': [1],
                  'args': [['agent'], 'You can\'t follow that.']}],
            ],
            func_constraints=[],
            callback_triggers=[
                {'action': 'followed', 'args': [0, 1]},
            ]
        )
        self.formats = {'followed': '{0} started following {1}. ',
                        'failed': '{0} couldn\'t follow that. '}

    def func(self, graph, args):
        """<actor> start following [<agent>] (unfollow if no agent supplied)"""
        agent_id = args[0]
        room_id = graph.location(args[0])
        following = graph.get_following(agent_id)

        if len(args) > 1 and following == args[1]:
            graph.send_msg(agent_id, 'You are already following them.\n')
            return True

        # Have to unfollow if currently following someone
        if following is not None:
            GRAPH_FUNCTIONS['unfollow'].handle(graph, [args[0]])

        graph.set_follow(agent_id, args[1])
        follow_action = {
            'caller': self.get_name(), 'name': 'followed',
            'room_id': room_id, 'actors': args,
        }
        graph.broadcast_to_room(follow_action)
        return True

    def get_reverse(self, graph, args):
        return 'unfollow', [args[0]]


class HitFunction(GraphFunction):
    """[Actor, victim in same room]"""
    def __init__(self):
        super().__init__(
            function_name=['hit'],
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{}, {'type': 'sameloc', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                [{'type': 'is_type', 'in_args': [1],
                  'args': [['agent'], 'You can\'t hit that.']},
                 {'type': 'not_type', 'in_args': [1],
                  'args': [['described'], 'You decide against attacking.']}],
            ],
            func_constraints=[],
            callback_triggers=[{'action': 'hit', 'args': [0, 1]}]
        )
        self.formats = {'attacked': '{0} attacked {1}. ',
                        'missed': '{0} attacked {1}, but missed. ',
                        'blocked': '{0} attacked {1}, but '
                                   '{1} blocked the attack. ',
                        'failed': '{0} couldn\'t hit that. '}

    def handle(self, graph, args):
        # if Callbacks.try_overrides(self, graph, args):
        #     return True
        is_constrained, _cons_passed, failure_action = \
            self.evaluate_constraint_set(graph, args, self.func_constraints)
        if not is_constrained:
            func_failure_action = self.get_failure_action(graph, args)
            # What failed?
            graph.send_action(args[0], func_failure_action)
            # Why it failed
            graph.send_action(args[0], failure_action)
            return False
        is_successful = self.func(graph, args)
        return is_successful

    def func(self, graph, args):
        """<Actor> attempts attack on <enemy>"""
        agent_id = args[0]
        room_id = graph.location(args[0])
        victim_id = args[1]
        damage = graph.get_prop(agent_id, 'damage', 0)
        armor = graph.get_prop(victim_id, 'defense', 0)
        attack = random.randint(0, damage + 1)
        defend = random.randint(0, armor)
        action = {
            'caller': self.get_name(), 'room_id': room_id, 'actors': args,
        }
        did_hit = False
        if damage == -1:  # It's not a very physical class
            if random.randint(0, 1):
                did_hit = True
                action['name'] = 'attacked'
                attack = 0
                defend = 0
            else:
                action['name'] = 'missed'
        elif (attack == 0 or attack - defend < 1) and not graph.freeze():
            action['name'] = 'missed' if attack == 0 else 'blocked'
        else:
            did_hit = True
            action['name'] = 'attacked'

        graph.broadcast_to_room(action)

        if did_hit:
            # Callbacks.handle_callbacks(self, graph, args)
            # TODO move energy updates to a shared function
            energy = graph.get_prop(victim_id, 'health')
            energy = max(0, energy - (attack - defend))
            if energy <= 0:
                graph.die(victim_id)
            elif energy < 4 and energy > 0:
                # TODO move getting a health action to the health function
                health_text = graph.health(victim_id)
                my_health_action = {'caller': 'health', 'actors': [victim_id],
                                    'name': 'display_health',
                                    'room_id': room_id,
                                    'add_descs': [health_text]}
                graph.send_action(victim_id, my_health_action)
            graph.set_prop(victim_id, 'health', energy)
        # else:
        #     Callbacks.handle_miss_callbacks(self, graph, args)
        return True

    def get_find_fail_text(self, arg_idx):
        return 'Where are they? You don\'t see them'


class HugFunction(GraphFunction):
    """[Actor, target in same room]"""
    def __init__(self):
        super().__init__(
            function_name=['hug'],
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{}, {'type': 'sameloc', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                [{'type': 'is_type', 'in_args': [1],
                  'args': [['agent'], 'You can\'t hug that.']}],
            ],
            func_constraints=[],
            callback_triggers=[{'action': 'hug', 'args': [0, 1]}]
        )
        self.formats = {'hug': '{0} hugged {1}. ',
                        'failed': '{0} couldn\'t hug that. '}

    def func(self, graph, args):
        """<Actor> hugs <target>"""
        room_id = graph.location(args[0])
        action = {
            'caller': self.get_name(), 'room_id': room_id, 'actors': args,
            'name': 'hug',
        }
        graph.broadcast_to_room(action)
        return True

    def get_find_fail_text(self, arg_idx):
        return 'Where are they? You don\'t see them'

    def get_reverse(self, graph, args):
        return False, []


class GetObjectFunction(GraphFunction):
    """[Actor, object attainable by actor, optional container for object]"""
    def __init__(self):
        super().__init__(
            function_name=['get', 'take'],
            possible_arg_nums=[2, 3],
            arg_split_words=['from'],
            arg_find_type=[
                [{}, {}],  # no constraints on the actor
                [{'type': 'all+here', 'from': 0},
                 {'type': 'carrying', 'from': 2}],
                [{'type': 'contains', 'from': 1},
                 {'type': 'all+here', 'from': 0}],
            ],
            arg_constraints=[
                [],  # no constraints on the actor
                [{'type': 'is_type', 'in_args': [1], 'args': ['object']},
                 {'type': 'not_type', 'in_args': [1],
                  'args': [['not_gettable'], "That isn't something you can get"]},
                 {'type': 'not_type', 'in_args': [1],
                  'args': [['described'],
                           "You choose not to take that and ruin "
                           "this room's description"]}],
                [{'type': 'is_type', 'in_args': [2],
                  'args': [['container', 'room'], 'That\'s not a container.']},
                 {'type': 'is_locked', 'in_args': [2], 'args': [False]}],
            ],
            func_constraints=[
                {'type': 'fits_in', 'in_args': [1, 0], 'args': [],
                 'spec_fail': 'cant_carry'},
            ],
            callback_triggers=[
                {'action': 'moved_to', 'args': [1, 0]},
                {'action': 'agent_got', 'args': [0, 1]},
            ]
        )
        self.formats = {'got': '{0} got {1}. ',
                        'got_from': '{0} got {1} from {2}. ',
                        'failed': '{0} couldn\'t get {1}. '}

    def func(self, graph, args):
        """<Actor> gets <item> [from <container>]"""
        agent_id, object_id = args[0], args[1]
        container_id = graph.location(object_id)
        room_id = graph.location(agent_id)
        graph.move_object(object_id, agent_id)
        if 'room' in graph.get_prop(container_id, 'classes'):
            get_action = {'caller': self.get_name(), 'name': 'got',
                          'room_id': room_id,
                          'actors': [agent_id, object_id]}
        else:
            get_action = {'caller': self.get_name(), 'name': 'got_from',
                          'room_id': room_id,
                          'actors': [agent_id, object_id, container_id]}
        graph.broadcast_to_room(get_action)
        return True

    def get_find_fail_text(self, arg_idx):
        if arg_idx == 2:
            return 'That isn\'t there. '
        else:
            return 'That isn\'t here. '

    def parse_descs_to_args(self, graph, args):
        # Append a location for the object, defaulting to the room if the
        # immediate container of something can't be found
        args = [a.lower() for a in args]
        if len(args) < 3:
            try:
                object_ids = graph.desc_to_nodes(args[1], args[0], 'all+here')
                i = 0
                container_id = args[0]
                # avoid reflexive matches!
                while container_id == args[0]:
                    container_id = graph.location(object_ids[i])
                    i += 1
                container_name = graph.node_to_desc_raw(container_id)
                args.append(container_name)
            except Exception:
                args.append(graph.node_to_desc_raw(graph.location(args[0])))
        return super().parse_descs_to_args(graph, args)

    def get_canonical_form(self, graph, args):
        """Get canonical form for the given args on this function"""
        if len(args) == 3 and graph.get_prop(args[2], 'container') is True:
            return super().get_canonical_form(graph, args)
        else:
            return super().get_canonical_form(graph, args[:2])

    def get_reverse(self, graph, args):
        if graph.get_prop(args[2], 'container') is True:
            return 'put', args
        else:
            return 'drop', args[:2]


# TODO override get_canonical_form for on/in
class PutObjectInFunction(GraphFunction):
    """[Actor, object actor is carrying, container]"""
    def __init__(self):
        super().__init__(
            function_name='put',
            possible_arg_nums=[3],
            arg_split_words=['in', 'into', 'on', 'onto'],
            arg_find_type=[{},  # no constraints on the actor
                           {'type': 'carrying', 'from': 0},
                           {'type': 'all+here', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                is_held_item(1),
                [{'type': 'is_type', 'in_args': [2],
                  'args': [['container'], 'That\'s not a container.']}],
            ],
            func_constraints=[
                {'type': 'fits_in', 'in_args': [1, 2], 'args': []},
            ],
            callback_triggers=[
                {'action': 'moved_to', 'args': [1, 2]},
                {'action': 'agent_put_in', 'args': [0, 1, 2]},
            ]
        )
        self.formats = {'put_in': '{0} put {1} into {2}. ',
                        'put_on': '{0} put {1} onto {2}. ',
                        'failed': '{0} couldn\'t put that. '}

    def func(self, graph, args):
        """<Actor> puts <object> in or on <container>"""
        agent_id, object_id, container_id = args[0], args[1], args[2]
        graph.move_object(object_id, container_id)
        act_name = 'put_' + graph.get_prop(container_id, 'surface_type', 'in')
        room_id = graph.location(agent_id)
        put_action = {'caller': self.get_name(), 'name': act_name,
                      'room_id': room_id,
                      'actors': [agent_id, object_id, container_id]}
        graph.broadcast_to_room(put_action)
        return True

    def get_find_fail_text(self, arg_idx):
        if arg_idx == 1:
            return 'You don\'t have that. '
        else:
            return 'That isn\'t here. '

    def get_reverse(self, graph, args):
        return 'get', args


class DropObjectFunction(GraphFunction):
    """[Actor, object actor is carrying]"""
    def __init__(self):
        super().__init__(
            function_name='drop',
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{},  # no constraints on the actor
                           {'type': 'carrying', 'from': 0},
                           {'type': 'all+here', 'from': 0}],
            arg_constraints=[[], is_held_item(1), []],
            func_constraints=[
                {'type': 'fits_in', 'in_args': [1, 2], 'args': []},
            ],
            callback_triggers=[
                {'action': 'moved_to', 'args': [1, 2]},
                {'action': 'agent_dropped', 'args': [0, 1]},
            ]
        )
        self.formats = {'dropped': '{0} dropped {1}. ',
                        'failed': '{0} couldn\'t drop that. '}

    def func(self, graph, args):
        """<Actor> drops <object>"""
        agent_id, object_id, room_id = args[0], args[1], args[2]
        graph.move_object(object_id, room_id)
        put_action = {'caller': self.get_name(), 'name': 'dropped',
                      'room_id': room_id, 'actors': [agent_id, object_id]}
        graph.broadcast_to_room(put_action)
        return True

    def get_find_fail_text(self, arg_idx):
        return 'You don\'t have that. '

    def parse_descs_to_args(self, graph, args):
        # appends the room as the target of the move
        args.append(graph.node_to_desc_raw(graph.location(args[0])))
        return super().parse_descs_to_args(graph, args)

    def get_reverse(self, graph, args):
        return 'get', args


class GiveObjectFunction(GraphFunction):
    """[Actor, object actor is carrying, other agent in same room]"""
    def __init__(self):
        super().__init__(
            function_name='give',
            possible_arg_nums=[3],
            arg_split_words=['to'],
            arg_find_type=[{},  # no constraints on the actor
                           {'type': 'carrying', 'from': 0},
                           {'type': 'sameloc', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                is_held_item(1),
                [{'type': 'is_type', 'in_args': [2],
                  'args': [['agent'], 'The recipient is a thing.']}],
            ],
            func_constraints=[
                {'type': 'fits_in', 'in_args': [1, 2], 'args': [],
                 'spec_fail': 'cant_carry'},
            ],
            callback_triggers=[
                {'action': 'moved_to', 'args': [1, 2]},
                {'action': 'agent_received_from', 'args': [2, 1, 0]},
            ]
        )
        self.formats = {'gave': '{0} gave {2} {1}. ',
                        'failed': '{0} couldn\'t give that. '}

    def func(self, graph, args):
        """<Actor> gives <object> to <agent>"""
        agent_id, object_id, receiver_id = args[0], args[1], args[2]
        graph.move_object(object_id, receiver_id)
        room_id = graph.location(agent_id)
        give_action = {'caller': self.get_name(), 'name': 'gave',
                       'room_id': room_id,
                       'actors': [agent_id, object_id, receiver_id]}
        graph.broadcast_to_room(give_action)
        return True

    def get_find_fail_text(self, arg_idx):
        if arg_idx == 1:
            return 'You don\'t have that. '
        else:
            return 'They aren\'t here. '

    def get_reverse(self, graph, args):
        return 'steal', args


class StealObjectFunction(GraphFunction):
    """[Actor, object other agent is carrying, other agent in same room]"""
    def __init__(self):
        super().__init__(
            function_name='steal',
            possible_arg_nums=[2, 3],
            arg_split_words=['from'],
            arg_find_type=[
                [{}, {}],  # no constraints on the actor
                [{'type': 'all+here', 'from': 0},
                 {'type': 'carrying', 'from': 2}],
                [{'type': 'contains', 'from': 1},
                 {'type': 'all+here', 'from': 0}],
            ],
            arg_constraints=[
                [],  # no constraints on the actor
                is_held_item(1),
                [{'type': 'is_type', 'in_args': [2],
                  'args': [['agent'],
                           'You can\'t steal from that.']}],
            ],
            func_constraints=[
                {'type': 'fits_in', 'in_args': [1, 0], 'args': [],
                 'spec_fail': 'cant_carry'},
            ],
            callback_triggers=[
                {'action': 'moved_to', 'args': [1, 0]},
                {'action': 'agent_stole_from', 'args': [0, 1, 2]},
            ]
        )
        self.formats = {'stole': '{0} stole {1} from {2}. ',
                        'failed': '{0} couldn\'t give that. '}

    def func(self, graph, args):
        """<Actor> steals <object> from <agent>"""
        agent_id, object_id, victim_id = args[0], args[1], args[2]
        graph.move_object(object_id, agent_id)
        room_id = graph.location(agent_id)
        give_action = {'caller': self.get_name(), 'name': 'stole',
                       'room_id': room_id,
                       'actors': [agent_id, object_id, victim_id]}
        graph.broadcast_to_room(give_action)
        return True

    def parse_descs_to_args(self, graph, args):
        # Append a location for the object, defaulting to the room if the
        # immediate container of something can't be found
        args = [a.lower() for a in args]
        if len(args) < 3:
            try:
                object_ids = graph.desc_to_nodes(args[1], args[0], 'all+here')
                i = 0
                container_id = args[0]
                # avoid reflexive matches!
                while container_id == args[0]:
                    container_id = graph.location(object_ids[i])
                    i += 1
                container_name = graph.node_to_desc_raw(container_id)
                args.append(container_name)
            except Exception:
                args.append(graph.node_to_desc_raw(graph.location(args[0])))
        return super().parse_descs_to_args(graph, args)

    def get_find_fail_text(self, arg_idx):
        if arg_idx == 1:
            return 'They don\'t have that. '
        else:
            return 'They aren\'t here. '

    def get_reverse(self, graph, args):
        return 'give', args


class EquipObjectFunction(GraphFunction):
    """[Actor, object actor is carrying]"""
    def __init__(self, function_name='equip', action='equipped',
                 additional_constraints=None):
        if additional_constraints is None:
            additional_constraints = []
        super().__init__(
            function_name=function_name,
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{}, {'type': 'carrying', 'from': 0}],
            arg_constraints=[[], is_held_item(1) + additional_constraints],
            func_constraints=[],
            callback_triggers=[
                {'action': 'agent_' + action, 'args': [0, 1]},
                {'action': action + '_by', 'args': [1, 0]},
            ]
        )
        self.formats = {action: '{0} ' + action + ' {1}. ',
                        'failed': '{0} couldn\'t ' + function_name + ' that. '}
        self.action = action

    def func(self, graph, args):
        """<Agent> equips <object>"""
        agent_id, object_id = args[0], args[1]
        graph.set_prop(object_id, 'equipped', self.name)
        for n, s in graph.get_prop(object_id, 'stats', {'defense': 1}).items():
            graph.inc_prop(agent_id, n, s)
        room_id = graph.location(agent_id)
        equip_action = {'caller': self.get_name(), 'name': self.action,
                        'room_id': room_id, 'actors': [agent_id, object_id]}
        graph.broadcast_to_room(equip_action)
        return True

    def get_find_fail_text(self, arg_idx):
        return 'You don\'t have that.'

    def get_reverse(self, graph, args):
        return 'remove', args


class WearObjectFunction(EquipObjectFunction):
    """[Actor, object actor is carrying]"""
    def __init__(self):
        super().__init__(
            function_name='wear', action='wore',
            additional_constraints=[
                {'type': 'is_type', 'in_args': [1],
                 'args': [['wearable'], 'That isn\'t wearable.']}
            ]
        )


class WieldObjectFunction(EquipObjectFunction):
    """[Actor, object actor is carrying]"""
    def __init__(self):
        super().__init__(
            function_name='wield', action='wielded',
            additional_constraints=[
                {'type': 'is_type', 'in_args': [1],
                 'args': [['weapon'], 'That isn\'t wieldable.']}
            ]
        )


class RemoveObjectFunction(GraphFunction):
    """[Actor, object actor is carrying]"""
    def __init__(self):
        super().__init__(
            function_name=['remove', 'unwield'],
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{}, {'type': 'carrying', 'from': 0}],
            arg_constraints=[[], is_equipped_item(1)],
            func_constraints=[],
            callback_triggers=[
                {'action': 'agent_removed', 'args': [0, 1]},
                {'action': 'removed_by', 'args': [1, 0]},
            ]
        )
        self.formats = {'removed': '{0} put {1} away. ',
                        'failed': '{0} couldn\'t remove that. '}

    def func(self, graph, args):
        """<Actor> unequips <object>"""
        agent_id, object_id = args[0], args[1]
        graph.set_prop(object_id, 'equipped', None)
        for n, s in graph.get_prop(object_id, 'stats', {'defense': 1}).items():
            graph.inc_prop(agent_id, n, -s)
        room_id = graph.location(agent_id)
        put_action = {'caller': self.get_name(), 'name': 'removed',
                      'room_id': room_id, 'actors': [agent_id, object_id]}
        graph.broadcast_to_room(put_action)
        return True

    def get_find_fail_text(self, arg_idx):
        return 'You don\'t have that equipped.'

    def get_reverse(self, graph, args):
        return 'equip', args


class IngestFunction(GraphFunction):
    """[Actor, object actor is carrying]"""
    def __init__(self, function_name='ingest', action='ingested',
                 additional_constraints=None):
        if additional_constraints is None:
            additional_constraints = []
        super().__init__(
            function_name=function_name,
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{}, {'type': 'carrying', 'from': 0}],
            arg_constraints=[[], is_held_item(1) + additional_constraints],
            func_constraints=[],
            callback_triggers=[
                {'action': 'agent_' + action, 'args': [0, 1]},
                {'action': action + '_by', 'args': [1, 0]},
            ]
        )
        self.formats = {action: '{0} ' + action + ' {1}. ',
                        'failed': '{0} couldn\'t ' + function_name + ' that. '}
        self.action = action

    def func(self, graph, args):
        """<Actor> consumes <object>"""
        agent_id, object_id = args[0], args[1]
        fe = graph.get_prop(object_id, 'food_energy')
        thing_desc = graph.node_to_desc(object_id, use_the=True)
        room_id = graph.location(agent_id)
        graph.delete_node(object_id)
        ingest_action = {'caller': self.get_name(), 'name': self.action,
                         'room_id': room_id, 'actors': [agent_id],
                         'add_descs': [thing_desc]}
        graph.broadcast_to_room(ingest_action)
        if fe >= 0:
            graph.send_msg(agent_id, "Yum.\n")
        else:
            graph.send_msg(agent_id, "Gross!\n")

        energy = graph.get_prop(agent_id, 'health')
        if energy < 8:
            prev_health = graph.health(agent_id)
            energy = energy + fe
            graph.set_prop(agent_id, 'health', energy)
            new_health = graph.health(agent_id)
            if energy <= 0:
                self.die(agent_id)
            elif prev_health != new_health:
                health_action = {'caller': 'health', 'name': 'changed',
                                 'room_id': room_id, 'actors': [args[0]],
                                 'add_descs': [prev_health, new_health]}
                graph.broadcast_to_room(health_action)
        return True

    def get_find_fail_text(self, arg_idx):
        return 'You don\'t have that.'


class EatFunction(IngestFunction):
    """[Actor, object actor is carrying]"""
    def __init__(self):
        super().__init__(
            function_name='eat', action='ate',
            additional_constraints=[
                {'type': 'is_type', 'in_args': [1],
                 'args': [['food'], 'That isn\'t food.']}
            ]
        )


class DrinkFunction(IngestFunction):
    """[Actor, object actor is carrying]"""
    def __init__(self):
        super().__init__(
            function_name='drink', action='drank',
            additional_constraints=[
                {'type': 'is_type', 'in_args': [1],
                 'args': [['drink'], 'That isn\'t a drink.']}
            ]
        )


class LockFunction(GraphFunction):
    """[Actor, lockable thing in same location, key actor is carrying]"""
    def __init__(self):
        super().__init__(
            function_name='lock',
            possible_arg_nums=[3],
            arg_split_words=['with'],
            arg_find_type=[{},  # no constraints on the actor
                           {'type': 'all+here', 'from': 0},
                           {'type': 'carrying', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                [{'type': 'is_lockable', 'in_args': [1], 'args':[True]}],
                [{'type': 'is_type', 'in_args': [2],
                  'args': [['key'], 'That isn\'t a key.']}],
            ],
            func_constraints=[
                {'type': 'is_locked', 'in_args': [1], 'args':[False]},
                {'type': 'locked_with', 'in_args': [1, 2], 'args':[]},
            ],
            callback_triggers=[{'action': 'agent_unlocked', 'args': [0, 1]}]
        )
        self.formats = {'locked': '{0} locked {1}. ',
                        'failed': '{0} couldn\'t lock that. '}

    def get_improper_args_text(self, num_args):
        if num_args == 2:
            return 'Lock that with what?'
        else:
            return 'Lock is used with "lock <object/path> with <key>"'

    def func(self, graph, args):
        """<Actor> locks <object> using <key>"""
        actor_id, target_id, key_id = args[0], args[1], args[2]
        room_id = graph.location(actor_id)
        if 'room' in graph.get_prop(target_id, 'classes'):
            graph.lock_path(graph.location(actor_id), target_id, key_id)
        else:
            # TODO implement lock_object
            graph.lock_object(target_id, key_id)
        lock_action = {'caller': self.get_name(), 'name': 'locked',
                       'room_id': room_id, 'actors': [actor_id, target_id]}
        graph.broadcast_to_room(lock_action)
        return True

    def get_reverse(self, graph, args):
        return 'lock', args


class UnlockFunction(GraphFunction):
    """[Actor, lockable thing in same location, key actor is carrying]"""
    def __init__(self):
        super().__init__(
            function_name='unlock',
            possible_arg_nums=[3],
            arg_split_words=['with'],
            arg_find_type=[{},  # no constraints on the actor
                           {'type': 'all+here', 'from': 0},
                           {'type': 'carrying', 'from': 0}],
            arg_constraints=[
                [],  # no constraints on the actor
                [{'type': 'is_lockable', 'in_args': [1], 'args':[True]}],
                [{'type': 'is_type', 'in_args': [2],
                  'args': [['key'], 'That isn\'t a key.']}],
            ],
            func_constraints=[
                {'type': 'is_locked', 'in_args': [1], 'args':[True]},
                {'type': 'locked_with', 'in_args': [1, 2], 'args':[]},
            ],
            callback_triggers=[{'action': 'agent_unlocked', 'args': [0, 1]}]
        )
        self.formats = {'unlocked': '{0} unlocked {1}. ',
                        'failed': '{0} couldn\'t unlock that. '}

    def func(self, graph, args):
        """<Actor> unlocks <object> using <key>"""
        actor_id, target_id, key_id = args[0], args[1], args[2]
        room_id = graph.location(actor_id)
        if 'room' in graph.get_prop(target_id, 'classes'):
            graph.unlock_path(graph.location(actor_id), target_id, key_id)
        else:
            # TODO implement unlock_object
            graph.unlock_object(target_id, key_id)
        lock_action = {'caller': self.get_name(), 'name': 'unlocked',
                       'room_id': room_id, 'actors': [actor_id, target_id]}
        graph.broadcast_to_room(lock_action)
        return True

    def get_improper_args_text(self, num_args):
        if num_args == 2:
            return 'Unlock that with what?'
        else:
            return 'Unlock is used with "unlock <object/path> with <key>"'

    def get_reverse(self, graph, args):
        return 'lock', args


class ExamineFunction(GraphFunction):
    """[Actor, anything accessible to the actor]"""
    def __init__(self):
        super().__init__(
            function_name=['examine', 'ex'],
            possible_arg_nums=[2],
            arg_split_words=[],
            arg_find_type=[{}, {'type': 'all+here', 'from': 0}],
            arg_constraints=[[], []],
            func_constraints=[],
            callback_triggers=[{'action': 'agent_examined', 'args': [0, 1]}]
        )

    def func(self, graph, args):
        """<Actor> examines <thing>"""
        agent_id, object_id = args[0], args[1]
        room_id = graph.location(agent_id)
        examine_action = {'caller': self.get_name(), 'name': 'witnessed',
                          'room_id': room_id, 'actors': [agent_id, object_id]}
        graph.broadcast_to_room(examine_action, [agent_id])

        if 'room' in graph.get_prop(object_id, 'classes'):
            object_type = 'room'
            add_descs = [graph.get_path_ex_desc(room_id, object_id, agent_id)]
        elif 'container' in graph.get_prop(object_id, 'classes'):
            # Mark containers as being examined so that they can be added
            # to the pool of all+here
            graph.set_prop(object_id, 'examined', True)
            object_type = 'container'
            add_descs = []
            add_descs.append(
                graph.get_classed_prop(object_id, 'desc', agent_id)
            )
            if len(graph.node_contains(object_id)) > 0:
                add_descs.append(graph.display_node(object_id))
        elif 'agent' in graph.get_prop(object_id, 'classes'):
            graph.set_prop(object_id, 'examined', True)
            object_type = 'agent'
            add_descs = []
            inv_txt = graph.get_inventory_text_for(object_id, give_empty=False)
            if graph.has_prop(object_id, 'desc'):
                add_descs.append(
                    graph.get_classed_prop(object_id, 'desc', agent_id)
                )
            if inv_txt != '':
                object_desc = graph.node_to_desc(object_id,
                                                 use_the=True).capitalize()
                add_descs.append(object_desc + ' is ' + inv_txt)
        else:
            object_type = 'object'
            add_descs = [graph.get_classed_prop(object_id, 'desc', agent_id)]
        if len(add_descs) == 0 or add_descs[0] is None:
            add_descs = ['There is nothing special about it. ']
        add_descs = ['\n'.join(add_descs)]  # Compress the descriptions to one
        examine_object_action = {
            'caller': self.get_name(), 'name': 'examine_object',
            'room_id': room_id, 'actors': args, 'add_descs': add_descs,
            'object_type': object_type
        }
        graph.send_action(agent_id, examine_object_action)
        return True

    def get_action_observation_format(self, action, descs):
        if action['name'] == 'witnessed':
            return '{0} looked at {1}. '
        elif action['name'] == 'examine_object':
            if 'past' in action:
                return '{0} examined {1}.\n{2}'
            else:
                return '{2}'
        return super().get_action_observation_format(action, descs)

    def get_find_fail_text(self, arg_idx):
        return 'That isn\'t here. '

    def get_improper_args_text(self, num_args):
        if num_args == 1:
            return 'Examine what?'
        else:
            return 'Examine is used as "examine <object>"'

    def get_reverse(self, graph, args):
        return False, args


class SoloFunction(GraphFunction):
    """[Actor]"""
    def __init__(self, function_name, callback_triggers):
        super().__init__(
            function_name=function_name,
            possible_arg_nums=[1],
            arg_split_words=[],
            arg_find_type=[{}],
            arg_constraints=[[]],
            func_constraints=[],
            callback_triggers=callback_triggers
        )

    def get_reverse(self, graph, args):
        return False, args


class WaitFunction(SoloFunction):
    def __init__(self):
        super().__init__(
            function_name='wait',
            callback_triggers=[
                {'action': 'agent_waited', 'args': [0]},
            ]
        )
        self.formats = {'waited': '{0} waited. '}

    def func(self, graph, args):
        room_id = graph.location(args[0])
        graph.send_action(
            args[0],
            {'caller': self.get_name(), 'name': 'waited',
             'room_id': room_id, 'actors': [args[0]]}
        )
        return True


class InventoryFunction(SoloFunction):
    def __init__(self):
        super().__init__(
            function_name=['inventory', 'inv', 'i'],
            callback_triggers=[
                {'action': 'check_inventory', 'args': [0]},
            ]
        )

    def get_action_observation_format(self, action, descs):
        if action['name'] == 'check_inventory':
            if descs[0] == 'You':
                return 'You checked your inventory. '
            elif descs[0] == 'I':
                return 'I checked my inventory'
            else:
                return '{0} checked their inventory. '
        elif action['name'] == 'list_inventory':
            if 'past' in action:
                if descs[0] == 'You':
                    return '{0} were {1}'
                else:
                    return '{0} was {1}'
            else:
                return '{0} are {1}'

    def func(self, graph, args):
        room_id = graph.location(args[0])
        inv_text = graph.get_inventory_text_for(args[0])
        my_inv_action = {'caller': self.get_name(), 'name': 'list_inventory',
                         'room_id': room_id, 'actors': [args[0]],
                         'add_descs': [inv_text]}
        graph.send_action(args[0], my_inv_action)
        return True


class HealthFunction(SoloFunction):
    def __init__(self):
        super().__init__(
            function_name=['health', 'status'],
            callback_triggers=[
                {'action': 'check_health', 'args': [0]},
            ]
        )

    def get_action_observation_format(self, action, descs):
        if action['name'] == 'check_health':
            if descs[0] == 'You':
                return 'You checked your health. '
            elif descs[0] == 'I':
                return 'I checked my health. '
            else:
                return '{0} checked their health. '
        elif action['name'] == 'display_health':
            if 'past' in action:
                if descs[0] == 'You':
                    return '{0} were feeling {1}. '
                else:
                    return '{0} was feeling {1}. '
            else:
                if descs[0] == 'You':
                    return '{0} are feeling {1}. '
                else:
                    return '{0} is feeling {1}. '
        elif action['name'] == 'changed':
            return '{0} went from feeling {1} to feeling {2}. \n'
        return super().get_action_observation_format(action, descs)

    def func(self, graph, args):
        room_id = graph.location(args[0])
        health_text = graph.health(args[0])
        my_health_action = {'caller': self.get_name(),
                            'name': 'display_health',
                            'room_id': room_id, 'actors': [args[0]],
                            'add_descs': [health_text]}
        graph.send_action(args[0], my_health_action)
        return True


class LookFunction(SoloFunction):
    def __init__(self):
        super().__init__(
            function_name=['look', 'l'],
            callback_triggers=[
                {'action': 'looked', 'args': [0]},
            ]
        )
        self.formats = {'looked': '{0} looked around. '}

    def func(self, graph, args):
        room_id = graph.location(args[0])

        # Prepare and send action for listing room contents
        is_return = room_id in graph._visited_rooms[args[0]]
        if is_return or not graph.has_prop(room_id, 'first_desc'):
            room_desc = graph.get_classed_prop(room_id, 'desc', args[0], None)
        else:
            room_desc = graph.get_classed_prop(room_id, 'first_desc', args[0])
        graph._visited_rooms[args[0]].add(room_id)
        if is_return or not graph.has_prop(room_id, 'first_desc'):
            agent_ids, agent_descs = \
                graph.get_nondescribed_room_agents(room_id)
            object_ids, object_descs = \
                graph.get_nondescribed_room_objects(room_id)
            _room_ids, room_descs = graph.get_room_edges(room_id)
        else:
            agent_ids, agent_descs = [], []
            object_ids, object_descs = [], []
            room_descs = []
        list_room_action = {
            'caller': self.get_name(), 'name': 'list_room', 'room_id': room_id,
            'actors': [args[0]], 'agent_ids': agent_ids,
            'present_agent_ids': agent_ids, 'object_ids': object_ids,
            'agent_descs': agent_descs, 'object_descs': object_descs,
            'room_descs': room_descs, 'room_desc': room_desc,
            'returned': False,
        }
        graph.send_action(args[0], list_room_action)
        return True

    def format_observation(self, graph, viewing_agent, action,
                           telling_agent=None, is_constraint=False):
        """Return the observation text to display for an action for the case
        of look. Look is a special case, as the description can change more
        than just tense based on who or what was seen and who you tell it to.
        """
        if action['name'] != 'list_room':
            return super().format_observation(graph, viewing_agent, action,
                                              telling_agent, is_constraint)
        room_desc = action['room_desc']
        object_descs = action['object_descs']
        object_ids = action['object_ids']
        ents = {object_descs[i]: object_ids[i] for i in range(len(object_ids))}
        agent_descs = action['agent_descs'][:]
        agent_ids = action['agent_ids']
        room_descs = action['room_descs']
        room = action['room_id']
        actor_id = action['actors'][0]
        returned = action['returned']

        if telling_agent is None:
            # Override for first description to set it to EXACTLY what was
            # given.
            has_actors = len(agent_ids) > 0
            if not has_actors:
                return room_desc

            try:
                # Remove viewing agent from the list (it's implied)
                i = agent_ids.index(viewing_agent)
                del agent_descs[i]
            except BaseException:
                pass
            if returned:
                s = 'You are back '
            else:
                s = 'You are '
            s += 'in {}.\n'.format(graph.node_to_desc(room))
            if room_desc is not None:
                s += room_desc + '\n'
            if len(object_ids) > 0:
                s += graph.get_room_object_text(object_descs, ents)
            if len(agent_descs) > 0:
                s += graph.get_room_agent_text(agent_descs)
            if len(room_descs) > 0:
                s += graph.get_room_edge_text(room_descs)
            return s
        else:
            is_present = room == graph.location(actor_id)

            if telling_agent in agent_ids:
                # Remove telling agent from descriptions (its implied)
                i = agent_ids.index(telling_agent)
                del agent_descs[i]
                del agent_ids[i]
            if telling_agent in agent_ids:
                # Replace telling agent with I (I was there)
                i = agent_ids.index(telling_agent)
                if i < len(agent_descs):
                    agent_descs[i] = 'I'
            actor_desc = graph.node_to_desc(actor_id) + ' was'
            if actor_id == telling_agent:
                actor_desc = 'I am' if is_present else 'I was'
            elif actor_id == viewing_agent:
                actor_desc = 'You are' if is_present else 'You were'

            s = '{} in {}.\n'.format(actor_desc, graph.node_to_desc(room))
            if room_desc is not None:
                s += room_desc + '\n'
            s += graph.get_room_object_text(object_descs, ents,
                                            past=not is_present,
                                            give_empty=False)
            if viewing_agent in agent_ids:
                # Replace viewing agent with you were
                i = agent_ids.index(viewing_agent)
                del agent_descs[i]
                del agent_ids[i]
                s += 'You are here.\n' if is_present else 'You were there.\n'
            s += graph.get_room_agent_text(agent_descs, past=not is_present)
            s += graph.get_room_edge_text(room_descs, past=not is_present)
            return s


class UnfollowFunction(SoloFunction):
    def __init__(self):
        super().__init__(
            function_name=['unfollow'],
            callback_triggers=[
                {'action': 'stopped_following', 'args': [0]},
            ]
        )
        self.formats = {'unfollowed': '{0} stopped following {1}. ',
                        'failed': '{0} couldn\'t follow that. '}

    def func(self, graph, args):
        agent_id = args[0]
        room_id = graph.location(args[0])
        following = graph.get_following(agent_id)
        if following is None:
            graph.send_msg(agent_id, 'You are not following anyone.\n')
        else:
            graph.set_follow(agent_id, None)
            unfollow_action = {
                'caller': self.get_name(), 'name': 'unfollowed',
                'room_id': room_id, 'actors': [args[0], following],
            }
            graph.broadcast_to_room(unfollow_action)
        return True


# Constraints
class GraphConstraint(object):
    """Stub class to define standard for graph constraints, implements shared
    code for executing them
    """
    def format_observation(self, graph, viewing_agent, action,
                           telling_agent=None):
        return format_observation(self, graph, viewing_agent, action,
                                  telling_agent, is_constraint=True)

    def get_failure_action(self, graph, args, spec_fail='failed'):
        """Given the args, return an action to represent the failure of
        meeting this constraint
        """
        raise NotImplementedError

    def get_action_observation_format(self, action, descs):
        """Given the action, return text to be filled by the parsed args"""
        raise NotImplementedError

    def evaluate_constraint(self, graph, args):
        """Evaluates a constraint, returns true or false for if it is met"""
        raise NotImplementedError


class FitsConstraint(GraphConstraint):
    """Determining if one object will fit into another

    args are:
        0 => actor_id
        1 => object_id
        2 => container_id
    """
    name = 'fits_in'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        # Special case for being unable to lift room objects
        if graph.get_prop(args[1], 'size') >= 150:
            return {'caller': self.name, 'name': 'cant_lift',
                    'room_id': graph.location(args[0]),
                    'actors': args}
        return {'caller': self.name, 'name': spec_fail,
                'room_id': graph.location(args[0]),
                'actors': args}

    def get_action_observation_format(self, action, descs):
        if action['name'] == 'cant_carry':
            descs[2] = descs[2].capitalize()
            if 'past' in action:
                return '{2} couldn\'t carry that much more'
            return '{2} can\'t carry that much more. '
        if action['name'] == 'cant_lift':
            return '{1} isn\'t something you can pick up'
        if 'past' in action:
            return '{1} didn\'t fit. '
        elif descs[1] in ['You', 'I']:
            return '{1} do not fit. '
        else:
            return '{1} does not fit. '

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object would fit in the
        container
        """
        return graph.obj_fits(args[1], args[2])


class IsTypeConstraint(GraphConstraint):
    """Determining if an object has inherited a particular type"""
    name = 'is_type'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        action = {'caller': self.name, 'name': spec_fail,
                  'room_id': graph.location(args[0]),
                  'actors': args[:2]}
        if len(args) == 4:
            action['add_descs'] = [args[3]]
        return action

    def get_action_observation_format(self, action, descs):
        if len(descs) == 3:
            return descs[2]
        if 'past' in action:
            return 'I couldn\'t find that. '
        else:
            return 'You couldn\'t find that. '

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object has any wanted class

        args are:
            0 => actor_id
            1 => found_id
            2 => expected_class/es
            3 => alternate failure text
        """
        found_id = args[1]
        expected_classes = args[2]
        if type(expected_classes) is not list:
            expected_classes = [expected_classes]
        obj_classes = graph.get_prop(found_id, 'classes')
        for want_class in expected_classes:
            if want_class in obj_classes:
                return True
        return False


class NotTypeConstraint(GraphConstraint):
    """Determining if an object has inherited a particular type"""
    name = 'not_type'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        action = {'caller': self.name, 'name': spec_fail,
                  'room_id': graph.location(args[0]),
                  'actors': args[:2]}
        if len(args) == 4:
            action['add_descs'] = [args[3]]
        return action

    def get_action_observation_format(self, action, descs):
        if len(descs) == 3:
            return descs[2]
        if 'past' in action:
            return 'I couldn\'t find that. '
        else:
            return 'You couldn\'t find that. '

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object has any wanted class

        args are:
            0 => actor_id
            1 => found_id
            2 => expected_class/es
            3 => alternate failure text
        """
        found_id = args[1]
        expected_classes = args[2]
        if type(expected_classes) is not list:
            expected_classes = [expected_classes]
        obj_classes = graph.get_prop(found_id, 'classes')
        for want_class in expected_classes:
            if want_class in obj_classes:
                return False
        return True


class HasPropConstraint(GraphConstraint):
    """Determining if an object doesn't have a prop it should have"""
    name = 'has_prop'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        return {'caller': self.name, 'name': spec_fail,
                'room_id': graph.location(args[0]),
                'actors': args[:2], 'add_descs': [args[2]]}

    spec_attribute_map = {
        'equipped': ['{1} had to be equipped first. ',
                     '{1} has to be equipped first. '],
    }

    def get_action_observation_format(self, action, descs):
        if 'past' in action:
            if descs[2] in self.spec_attribute_map:
                return self.spec_attribute_map[descs[2]][0]
            return '{1} wasn\'t {2}'
        else:
            if descs[2] in self.spec_attribute_map:
                return self.spec_attribute_map[descs[2]][1]
            return '{1} isn\'t {2}'

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object doesn't have a prop

        args are:
            0 => viewer
            1 => prop_id
            2 => bad_prop
        """
        return graph.has_prop(args[1], args[2])


class NoPropConstraint(GraphConstraint):
    """Determining if an object has a prop it shouldn't have"""
    name = 'no_prop'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        return {'caller': self.name, 'name': spec_fail,
                'room_id': graph.location(args[0]),
                'actors': args[:2], 'add_descs': [args[2]]}

    spec_attribute_map = {
        'equipped': ['{1} had to be put away first. ',
                     '{1} has to be put away first. '],
    }

    def get_action_observation_format(self, action, descs):
        if 'past' in action:
            if descs[2] in self.spec_attribute_map:
                return self.spec_attribute_map[descs[2]][0]
            return '{1} was {2}'
        else:
            if descs[2] in self.spec_attribute_map:
                return self.spec_attribute_map[descs[2]][1]
            return '{1} is {2}'

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object doesn't have a prop

        args are:
            0 => viewer
            1 => prop_id
            2 => bad_prop
        """
        return not graph.has_prop(args[1], args[2])


class LockableConstraint(GraphConstraint):
    """Determining if a path has a particular status

    args are:
        0 => actor_id
        1 => target_id
        2 => should be lockable
    """
    name = 'is_lockable'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        return {'caller': self.name, 'name': spec_fail,
                'room_id': graph.location(args[0]),
                'actors': args[:2], 'add_descs': [args[2]]}

    def get_action_observation_format(self, action, descs):
        if descs[2]:  # Failed when it was supposed to be lockable
            return '{1} can\'t be locked! '
        else:
            return '{1} is lockable. '

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object would fit in the
        container
        """
        actor_id, target_id, want_lockable = args[0], args[1], args[2]
        if 'room' in graph.get_prop(target_id, 'classes'):
            room_id = graph.location(actor_id)
            is_lockable = \
                graph.get_path_locked_with(room_id, target_id) is not None
        else:
            is_lockable = graph.get_prop(target_id, 'lockable', False)
        return want_lockable == is_lockable


class LockedConstraint(GraphConstraint):
    """Determining if a path has a particular status

    args are:
        0 => actor_id
        1 => target_id
        2 => should be locked
    """
    name = 'is_locked'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        return {'caller': self.name, 'name': spec_fail,
                'room_id': graph.location(args[0]),
                'actors': args[:2], 'add_descs': [args[2]]}

    def get_action_observation_format(self, action, descs):
        if descs[2]:  # Failed when it was supposed to be locked
            return '{1} is unlocked. '
        else:
            return '{1} is locked. '

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object would fit in the
        container
        """
        actor_id, target_id, want_locked = args[0], args[1], args[2]
        if 'room' in graph.get_prop(target_id, 'classes'):
            room_id = graph.location(actor_id)
            if room_id == target_id:
                return True
            is_locked = \
                graph.path_is_locked(room_id, target_id)
        else:
            is_locked = graph.get_prop(target_id, 'locked', False)
        return want_locked == is_locked


class LockedWithConstraint(GraphConstraint):
    """Determining if a path has a particular status

    args are:
        0 => actor_id
        1 => target_id
        2 => key_id
    """
    name = 'locked_with'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        item_desc = graph.node_to_desc(args[2])
        return {'caller': self.name, 'name': spec_fail,
                'room_id': graph.location(args[0]),
                'actors': args[:2], 'add_descs': [item_desc]}

    def get_action_observation_format(self, action, descs):
        descs[2] = descs[2].capitalize()
        return '{2} doesn\'t work with that. '

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object would fit in the
        container
        """
        actor_id, target_id, key_id = args[0], args[1], args[2]
        if 'room' in graph.get_prop(target_id, 'classes'):
            room_id = graph.location(actor_id)
            locked_with = \
                graph.get_path_locked_with(room_id, target_id)
        else:
            locked_with = graph.get_prop(target_id, 'locked_with', False)
        return locked_with == key_id


class NotLocationOfConstraint(GraphConstraint):
    """Ensuring a location isn't the same one as the actor

    args are:
        0 => actor_id
        1 => target_id
    """
    name = 'not_location_of'

    def get_failure_action(self, graph, args, spec_fail='failed'):
        return {'caller': self.name, 'name': spec_fail,
                'room_id': graph.location(args[0]),
                'actors': args, 'add_descs': []}

    def get_action_observation_format(self, action, descs):
        return "You're already in that location. "

    def evaluate_constraint(self, graph, args):
        """Return true or false for whether the object would fit in the
        container
        """
        actor_id, target_id = args[0], args[1]
        return graph.location(actor_id) != target_id


con_list = [
    FitsConstraint(),
    IsTypeConstraint(),
    HasPropConstraint(),
    NoPropConstraint(),
    LockableConstraint(),
    LockedConstraint(),
    LockedWithConstraint(),
    NotLocationOfConstraint(),
    NotTypeConstraint(),
]
func_list = [
    MoveAgentFunction(),
    GetObjectFunction(),
    PutObjectInFunction(),
    DropObjectFunction(),
    GiveObjectFunction(),
    StealObjectFunction(),
    WearObjectFunction(),
    WieldObjectFunction(),
    RemoveObjectFunction(),
    WaitFunction(),
    InventoryFunction(),
    HealthFunction(),
    EatFunction(),
    DrinkFunction(),
    LookFunction(),
    LockFunction(),
    UnlockFunction(),
    ExamineFunction(),
    FollowFunction(),
    UnfollowFunction(),
    HitFunction(),
    TellFunction(),
    HugFunction(),
    # UseFunction(),
    SayFunction(),
]
CONSTRAINTS = {c.name: c for c in con_list}

# Construct list of graph functions by usable names
CANNONICAL_GRAPH_FUNCTIONS = {}
for f in func_list:
    names = f.name
    if type(names) is str:
        names = [names]
    for name in names:
        GRAPH_FUNCTIONS[name] = f
    CANNONICAL_GRAPH_FUNCTIONS[names[0]] = f

# Functions for the possible action parser to ignore
FUNC_IGNORE_LIST = ['tell', 'say']


class Graph(object):

    def __init__(self, opt):
        """
        Initialize a graph for the game to run on. Creates all of the
        required bookkeeping members and initializes them to empty. Most
        members of the graph are keyed by a node's unique ids.

        Development of this graph is very much still a work in progress,
        and it is only provided in this form to be used for the light_dialog
        task as collected in the 'Learning to Speak and Act in a Fantasy
        Text Adventure Game' paper
        """

        # callbacks hold custom registered functions for the graph to execute
        # under predefined constraints
        self.callbacks = {}
        # Variables can be tied to custom callbacks to track custom variables
        # that are parts of the graph state
        self.variables = {}

        # ParlAI opt
        self._opt = opt

        # Dict of edges from a given node, in the format of
        # (edge_type, target) -> dict of options for that pair. This format
        # is subject to change, but new versions will include conversion
        # functions
        self._node_to_edges = {}
        # Dict of properties from a given node, in the format of
        # prop_name -> value
        self._node_to_prop = {}
        # dict of node -> node for what nodes are contained in what
        self._node_contained_in = {}
        # dict of node -> node list for what nodes a node contains
        self._node_contains = {}
        # dict of node -> node for what nodes a node is following
        self._node_follows = {}
        # dict of node -> node list for what nodes a node is following
        self._node_followed_by = {}
        # dict of node -> string description for that node
        self._node_to_desc = {}

        # dict of node -> set of room ids that that agent has visited so far
        self._visited_rooms = {}
        # dict of node -> immediately previous room id the agent has been in.
        # to enable "go back"
        self._last_rooms = {}
        # dict of node -> agent for the last agent that an actor used a tell
        # command on. Useful for trying to determine who an agent is talking
        # to once they start using say rather than tell.
        self._last_tell_target = {}

        # non-player characters that we move during update_world func.
        self._node_npcs = set()
        # flag for if the graph is to be frozen, stopping time steps
        self._node_freeze = False
        # count of nodes in the graph
        self._cnt = 0

        # buffers for holding actions until an agent is able to observe them
        self._node_to_text_buffer = {}
        self._node_to_observations = {}

        # node ids that can be associated with an agent
        self._player_ids = []
        # node ids of the player_ids that have already been registered to by
        # an agent
        self._used_player_ids = []

        # special member function to register callback events to
        # _node_to_text_buffer and _node_to_observations for cases where
        # polling the graph is not appropriate
        self.__message_callbacks = {}

        # global variable for if NPCs should respond to Says or only Tells
        self.npc_say_respond = True

    # -- Graph properties -- #

    def copy(self):
        """return a copy of this graph"""
        return deepcopy(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if '__message_callbacks' not in k:  # we cant copy callback anchors
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, {})
        return result

    def populate_ids(self):
        self.object_ids = self.get_all_by_prop('object')
        self.container_ids = self.get_all_by_prop('container')
        self.agent_ids = self.get_all_by_prop('agent')

    def unique_hash(self):
        # TODO: make it independent of specific world settings
        # TODO: Improve everything about this, it's really old
        # object_ids, agent_ids, and container_ids are set by construct_graph
        self.populate_ids()
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

    def all_node_ids(self):
        """return a list of all the node ids in the graph"""
        return list(self._node_to_prop.keys())

    def freeze(self, freeze=None):
        if freeze is not None:
            self._node_freeze = freeze
        return self._node_freeze

    def version(self):
        return 3

    def add_message_callback(self, id, func):
        """
        Register a message callback to a given agent by their node id. This
        function will be called on the graph and an action when the subscribed
        actor is the subject of an action
        """
        self.__message_callbacks[id] = func

    def has_remaining_player_slots(self):
        """Note if the graph has any space for new players to play a role"""
        return len(self._player_ids) != 0

    def get_unused_player_id(self):
        """
        Return one of the unused player ids randomly and mark it as registered
        """
        player_id = random.choice(self._player_ids)
        self._player_ids.remove(player_id)
        self._used_player_ids.append(player_id)
        return player_id

    # -- Callbacks and variables -- #

    def register_callbacks(self, callbacks, variables):
        """ Documentation not yet available"""
        self.callbacks = callbacks
        self.variables = variables

    # -- Full node editors/creators/getters/deleters -- #

    def get_player(self, id):
        """Get the state of a player (along with everything they carry) from
        the graph
        """
        state = {}
        state['id'] = id
        if id in self._node_to_text_buffer:
            state['node_to_text_buffer'] = self._node_to_text_buffer[id]
        if id in self._node_contains:
            state['node_contains'] = self._node_contains[id]
            state['contained_nodes'] = []
            for obj_id in self._node_contains[id]:
                props = self._node_to_prop[obj_id]
                if 'persistent' in props and props['persistent'] == 'level':
                    continue  # Leave level-based items behind
                state['contained_nodes'].append(self.get_obj(obj_id))
        if id in self._node_to_prop:
            state['node_to_prop'] = self._node_to_prop[id]
        if id in self._node_to_desc:
            state['node_to_desc'] = self._node_to_desc[id]
        return state

    def get_obj(self, id):
        """Get the state of an object from the graph"""
        state = {}
        state['id'] = id
        if id in self._node_contains:
            state['node_contains'] = self._node_contains[id]
            state['contained_nodes'] = []
            for obj_id in self._node_contains[id]:
                state['contained_nodes'].append(self.get_obj(obj_id))
        if id in self._node_to_prop:
            state['node_to_prop'] = self._node_to_prop[id]
        if id in self._node_contained_in:
            state['node_contained_in'] = self._node_contained_in[id]
        if id in self._node_to_desc:
            state['node_to_desc'] = self._node_to_desc[id]
        return state

    def set_player(self, state):
        """Instantiate a player into the graph with the given state"""
        id = state['id']
        if 'node_to_text_buffer' in state:
            self._node_to_text_buffer[id] = \
                state['node_to_text_buffer'] + self._node_to_text_buffer[id]
        if 'node_contains' in state:
            self._node_contains[id] = state['node_contains']
            for obj_state in state['contained_nodes']:
                self.set_obj(obj_state)
        if 'node_to_prop' in state:
            self._node_to_prop[id] = state['node_to_prop']

    def set_obj(self, state):
        """Instantiate an object into the graph with the given state"""
        id = state['id']
        self._node_to_edges[id] = []
        self._node_to_prop[id] = state['node_to_prop']
        self._node_contains[id] = state['node_contains']
        for obj_state in state['contained_nodes']:
            self.set_obj(obj_state)
        self._node_to_desc[id] = state['node_to_desc']
        self._node_contained_in[id] = state['node_contained_in']

    def add_node(self, desc, props, is_player=False, uid=''):
        id = desc.lower()
        if id != 'dragon' and not is_player:
            id = "{}_{}".format(id, self._cnt)
            if uid != '':
                id += '_{}'.format(uid)
        self._cnt = self._cnt + 1
        if id in self._node_to_edges:
            return False
        self._node_to_edges[id] = {}
        if type(props) == str:
            self._node_to_prop[id] = {}
            self._node_to_prop[id][props] = True
            self.set_prop(id, 'classes', [props])
        elif type(props) == dict:
            self._node_to_prop[id] = {}
            for p in props:
                self._node_to_prop[id][p] = props[p]
        else:
            self._node_to_prop[id] = {}
            for p in props:
                self._node_to_prop[id][p] = True
        if self._node_to_prop[id].get('names') is None:
            self._node_to_prop[id]['names'] = [desc]
        self._node_to_prop[id]['is_player'] = is_player
        self._node_contains[id] = set()
        self._node_to_desc[id] = desc
        if 'agent' in self._node_to_prop[id]['classes'] or is_player:
            if not self.has_prop(id, 'health'):
                self.set_prop(id, 'health', 2)
            self.new_agent(id)
        if is_player:
            self._player_ids.append(id)
        return id

    def new_agent(self, id):
        """
        Initialize special state for the given agent
        """
        self._node_to_text_buffer[id] = ''  # clear buffer
        self._node_to_observations[id] = []  # clear buffer
        self._visited_rooms[id] = set()
        self._last_rooms[id] = None
        self._last_tell_target[id] = None

    def delete_node(self, id):
        """Remove a node from the graph, keeping the reference for printing"""
        # Remove from the container
        loc = self.location(id)
        if loc is not None and id in self._node_contains[loc]:
            self._node_contains[self.location(id)].remove(id)
        # remove size from above object, then remove contained in edge
        above_id = self.node_contained_in(id)
        if above_id is not None:
            size = self.get_prop(id, 'size')
            omax_size = self.get_prop(above_id, 'contain_size')
            omax_size = omax_size + size
            self.set_prop(above_id, 'contain_size', omax_size)
            rm(self._node_contained_in, id)
        # all things inside this are zapped too
        if id in self._node_contains:
            objs = deepcopy(self._node_contains[id])
            for o in objs:
                self.delete_node(o)
        # now remove edges from other rooms
        for r in self._node_to_edges[id]:
            if r[0] == 'path_to':
                self._node_to_edges[r[1]].remove(['path_to', id])
        # stop all agents from following this one
        if id in self._node_followed_by:
            ags = deepcopy(self._node_followed_by[id])
            for a in ags:
                self.set_follow(a, None)
        rm(self._node_follows, id)
        # Remove from npc list
        if id in self._node_npcs:
            self._node_npcs.remove(id)

    # -- Node-in-graph properties -- #

    def add_edge(self, id1, edge, id2, edge_label=None, locked_with=None,
                 edge_desc=None, full_label=False):
        """Add an edge of the given type from id1 to id2. Optionally can set
        a label for that edge
        """
        if edge_desc is None:
            edge_desc = self.get_props_from_either(id2, id1, 'path_desc')[0]
        if (edge, id2) not in self._node_to_edges[id1]:
            self._node_to_edges[id1][(edge, id2)] = {
                'label': edge_label,
                'examine_desc': edge_desc,
                # TODO get these from the lock or something idk
                'locked_desc': edge_desc,
                'unlocked_desc': edge_desc,
                'locked_with': locked_with,
                'is_locked': False,
                'full_label': full_label,
            }

    def add_one_path_to(self, id1, id2, label=None,
                        locked_with=None, desc=None, full_label=False):
        if id1 == id2:
            return False
        self.add_edge(id1, 'path_to', id2, label, locked_with, desc,
                      full_label)
        return True

    def add_path_to(self, id1, id2, desc1=None, desc2=None, locked_with=None,
                    examine1=None, examine2=None):
        """Create a path between two rooms"""
        if id1 == id2:
            return False
        self.add_edge(id1, 'path_to', id2, desc1, locked_with, examine1)
        self.add_edge(id2, 'path_to', id1, desc2, locked_with, examine2)
        return True

    def is_path_to(self, id1, id2):
        """determine if there is a path from id1 to id2"""
        return ('path_to', id2) in self._node_to_edges[id1]

    def node_contains(self, loc):
        """Get the set of all things that a node contains"""
        if loc in self._node_contains:
            return set(self._node_contains[loc])
        else:
            return set()

    def add_contained_in(self, id1, id2):
        if id1 in self._node_contained_in:
            i_am_in = self._node_contained_in[id1]
            self._node_contains[i_am_in].remove(id1)
        self._node_contained_in[id1] = id2
        self._node_contains[id2].add(id1)
        return True

    def node_contained_in(self, id):
        if id not in self._node_contained_in:
            return None
        return self._node_contained_in[id]

    def set_follow(self, id1, id2):
        """Set id1 to be following id2, unfollowing whatever id1 followed
        before if necessary
        """
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

    def get_followers(self, agent_id):
        """Get the nodes following the given agent"""
        if agent_id in self._node_followed_by:
            return self._node_followed_by[agent_id]
        return []

    def get_following(self, agent_id):
        """get the node that the given agent is following, if any"""
        if agent_id in self._node_follows:
            return self._node_follows[agent_id]
        return None

    def combine_classed_descs(descs):
        adj_descs = [{'default': d} if type(d) is str else d for d in descs]
        last_round_descs = {'default': ''}
        for desc_set in [adj_descs]:
            round_descs = {}
            for class_name in desc_set:
                if class_name not in last_round_descs:
                    old_d = last_round_descs['default'].strip()
                    new_d = desc_set[class_name].strip()
                    round_descs[class_name] = (old_d + ' ' + new_d).strip()
                else:
                    old_d = last_round_descs[class_name].strip()
                    new_d = desc_set[class_name].strip()
                    round_descs[class_name] = (old_d + ' ' + new_d).strip()
            last_round_descs = round_descs
        return last_round_descs

    def lock_path(self, id1, id2, key_id):
        """lock the edge from id1 to id2 using the given key"""
        self._node_to_edges[id1][('path_to', id2)]['is_locked'] = True
        self._node_to_edges[id1][('path_to', id2)]['examine_desc'] = \
            self._node_to_edges[id1][('path_to', id2)]['locked_desc']
        self._node_to_edges[id2][('path_to', id1)]['is_locked'] = True
        self._node_to_edges[id2][('path_to', id1)]['examine_desc'] = \
            self._node_to_edges[id2][('path_to', id1)]['locked_desc']

    def unlock_path(self, id1, id2, key_id):
        """unlock the edge from id1 to id2 using the given key"""
        self._node_to_edges[id1][('path_to', id2)]['is_locked'] = False
        self._node_to_edges[id1][('path_to', id2)]['examine_desc'] = \
            self._node_to_edges[id1][('path_to', id2)]['unlocked_desc']
        self._node_to_edges[id2][('path_to', id1)]['is_locked'] = False
        self._node_to_edges[id2][('path_to', id1)]['examine_desc'] = \
            self._node_to_edges[id2][('path_to', id1)]['unlocked_desc']

    def path_is_locked(self, id1, id2):
        return self._node_to_edges[id1][('path_to', id2)]['is_locked']

    def get_path_locked_with(self, id1, id2):
        if id1 == id2:
            # TODO get the locked_with of a room when the path is to itself
            # as this function shouldn't be called like this
            return None
        return self._node_to_edges[id1][('path_to', id2)]['locked_with']

    def set_desc(self, id, desc):
        self._node_to_desc[id] = desc

    def node_exists(self, id):
        return id in self._node_contained_in

    # -- Prop accessors -- #

    def get_prop(self, id, prop, default=None):
        """Get the given prop, return None if it doesn't exist"""
        if id in self._node_to_prop:
            if prop in self._node_to_prop[id]:
                return self._node_to_prop[id][prop]
        return default

    def extract_classed_from_dict(self, prop_d, viewer=None, default=None):
        if type(prop_d) is not dict:
            if prop_d is None:
                return default
            return prop_d
        if viewer is not None:
            viewer_class = self.get_prop(viewer, 'class')
            if viewer_class in prop_d:
                return prop_d[viewer_class]
            for viewer_class in self.get_prop(viewer, 'classes'):
                if viewer_class in prop_d:
                    return prop_d[viewer_class]
        if 'default' not in prop_d:
            return default
        return prop_d['default']

    def get_classed_prop(self, id, prop, viewer, default=None):
        prop_d = self.get_prop(id, prop)
        val = self.extract_classed_from_dict(prop_d, viewer, default)
        if type(val) is dict and 'iter' in val:
            i = val['iter']
            val['iter'] = (i + 1) % len(val['data'])
            return val['data'][i]
        return val

    def get_props_from_either(self, id1, id2, prop):
        """Try to get ones own prop, fallback to other's prop. Do this for
        both provided ids symmetrically"""
        resp1 = self.get_prop(id1, prop, self.get_prop(id2, prop))
        resp2 = self.get_prop(id2, prop, self.get_prop(id1, prop))
        return resp1, resp2

    def set_prop(self, id, prop, val=True):
        """Set a prop to a given value, True otherwise"""
        if id in self._node_to_prop:
            self._node_to_prop[id][prop] = val

    def has_prop(self, id, prop):
        """Return whether or not id has the given prop"""
        if self.get_prop(id, prop) is not None:
            return True
        return False

    def inc_prop(self, id, prop, val=1):
        """Increment a given prop by the given value"""
        if id in self._node_to_prop:
            if prop not in self._node_to_prop[id]:
                self.set_prop(id, prop, 0)
            if type(self._node_to_prop[id][prop]) != int:
                self.set_prop(id, prop, 0)
            self._node_to_prop[id][prop] += val

    def delete_prop(self, id, prop):
        """Remove a prop from the node_to_prop map"""
        if id in self._node_to_prop:
            if prop in self._node_to_prop[id]:
                del self._node_to_prop[id][prop]

    def add_class(self, object_id, class_name):
        curr_classes = self.get_prop(object_id, 'classes')
        curr_classes.append(class_name)

    def remove_class(self, object_id, class_name):
        curr_classes = self.get_prop(object_id, 'classes')
        curr_classes.remove(class_name)

    # -- Graph locators -- #

    def location(self, thing):
        """Get whatever the immediate container of a node is"""
        if thing in self._node_contained_in:
            return self._node_contained_in[thing]
        else:
            return None

    def room(self, thing):
        """Get the room that contains a given node"""
        if thing not in self._node_contained_in:
            return None
        id = self._node_contained_in[thing]
        while not self.get_prop(id, 'room'):
            id = self._node_contained_in[id]
        return id

    def desc_to_nodes(self, desc, nearbyid=None, nearbytype=None):
        """Get nodes nearby to a given node from that node's perspective"""
        from_id = self.location(nearbyid)
        if nearbyid is not None:
            o = set()
            if 'all' in nearbytype and 'here' in nearbytype:
                o = self.get_local_ids(nearbyid)
            else:
                if 'path' in nearbytype:
                    o1 = self.node_path_to(from_id)
                    o = set(o).union(o1).union([from_id])
                if 'carrying' in nearbytype:
                    o = set(o).union(set(self.node_contains(nearbyid)))
                if 'sameloc' in nearbytype:
                    o1 = set(self.node_contains(from_id))
                    o = o.union(o1)
                if 'all' in nearbytype:
                    o1 = self.node_contains(nearbyid)
                    o2 = self.node_contains(from_id)
                    o3 = self.node_path_to(from_id)
                    o = o.union(o1).union(o2).union(o3)
                if 'contains' in nearbytype:
                    o = o.union({self.location(nearbyid)})
                if 'players' in nearbytype:
                    o = o.union(self._used_player_ids)
                if 'others' in nearbytype:
                    for item in o:
                        if self.get_prop(item, 'agent') \
                                or self.get_prop(item, 'container'):
                            o = o.union(self.node_contains(item))
                # if len(o) == 0:
                #     o1 = self.node_contains(nearbyid)
                #     o2 = self.node_contains(self.location(nearbyid))
                #     o = o1.union(o2)
        else:
            o = set(self._node_to_desc.keys())
        # Go through official in-game names
        if nearbyid is not None and self.room(nearbyid) in o and \
                desc == 'room':
            return [self.room(nearbyid)]

        found_pairs = [(id, self.node_to_desc(id, from_id=from_id).lower())
                       for id in o]
        valid_ids_1 = [(id, name) for (id, name) in found_pairs
                       if desc.lower() in name+'s']

        # Check the parent name trees for names that also could match in the
        # case that nothing could be found
        all_subnames = [(id, self.get_prop(id, 'names')) for id in o]
        all_pairs = [(id, name)
                     for (id, name_list) in all_subnames
                     for name in name_list]
        valid_ids_2 = [(id, name) for (id, name) in all_pairs
                       if desc.lower() in name+'s']

        valid_ids_1.sort(key=lambda x: len(x[0]))
        valid_ids_2.sort(key=lambda x: len(x[1]))
        valid_ids = valid_ids_1 + valid_ids_2
        valid_ids = [id for (id, _name) in valid_ids]
        return valid_ids

    def get_all_by_prop(self, prop):
        objects = self.all_node_ids()
        return [id for id in objects if self.get_prop(id, prop)]

    def node_path_to(self, id):
        if id is None:
            return []
        rooms = self._node_to_edges[id]
        rooms = [r[1] for r in rooms if r[0] == 'path_to']
        return rooms

    def get_actionable_ids(self, actor_id):
        o = self.get_local_ids(actor_id)
        new_o = set(o)
        for obj in o:
            if self.get_prop(obj, 'container') or self.get_prop(obj, 'agent'):
                new_o = new_o.union(self.node_contains(obj))
        return new_o

    def get_local_ids(self, actor_id):
        """Return all accessible ids for an actor given the current area"""
        loc = self.location(actor_id)
        o1 = self.node_contains(actor_id)
        o2 = self.node_contains(loc)
        o3 = self.node_path_to(loc)
        o4 = [loc]
        local_ids = o1.union(o2).union(o3).union(o4)
        check_local_ids = list(local_ids)
        for pos_id in check_local_ids:
            if self.get_prop(pos_id, 'examined'):
                local_ids = local_ids.union(self.node_contains(pos_id))
        return local_ids

    # -- Text creators -- #

    def get_inventory_text_for(self, id, give_empty=True):
        """Get a description of what id is carrying or equipped with"""
        s = ''
        carry_ids = []
        wear_ids = []
        wield_ids = []
        for o in self.node_contains(id):
            if self.get_prop(o, 'equipped'):
                if 'wearable' in self.get_prop(o, 'classes'):
                    wear_ids.append(o)
                elif 'weapon' in self.get_prop(o, 'classes'):
                    wield_ids.append(o)
            else:
                carry_ids.append(o)
        if len(carry_ids) == 0:
            if not give_empty:
                return ''
            s += 'carrying nothing'
        else:
            s += 'carrying ' + self.display_node_list(carry_ids)
        if len(wear_ids) > 0:
            s += ',\n'
            if len(wield_ids) == 0:
                s += 'and '
            s += 'wearing ' + self.display_node_list(wear_ids)
        if len(wield_ids) > 0:
            s += ',\nand wielding ' + self.display_node_list(wield_ids)
        return s + '.'

    def health(self, id):
        """Return the text description of someone's numeric health"""
        health = self.get_prop(id, 'health')
        if health is None or health is False:
            health = 1
        if health > 8:
            health = 8
        f = ['dead', 'on the verge of death',
             'very weak', 'weak', 'ok',
             'good', 'strong', 'very strong',
             'nigh on invincible']
        return f[int(health)]

    # -- Text accessors -- #

    def get_action_history(self, agent_id):
        observations = self._node_to_observations[agent_id]
        self._node_to_observations[agent_id] = []
        return observations

    def node_to_desc(self, id, from_id=False, use_the=False):
        if from_id:
            # A description of something (right now, a location)
            # from another location.
            # This gets the description from the path edge.
            path_desc = self.path_to_desc(id, from_id)
            if path_desc is not False:
                if path_desc.startswith('the') or path_desc.startswith('a'):
                    return path_desc
                return 'the ' + path_desc

        if id in self._node_to_desc:
            ent = self._node_to_desc[id]
            if self.get_prop(id, 'capitalize', False) is True:
                ent = ent.capitalize()
            if self.has_prop(id, 'dead'):
                ent = 'dead ' + ent
            if self.has_prop(id, 'player_name'):
                ent = self.get_prop(id, 'player_name')
            elif self.has_prop(id, 'agent') or self.has_prop(id, 'object'):
                prefix = self.name_prefix(id, ent, use_the)
                if prefix != '':
                    ent = prefix + ' ' + ent
            elif self.has_prop(id, 'room'):
                prefix = self.name_prefix(id, ent, use_the)
                if prefix != '':
                    ent = prefix + ' ' + ent
                else:
                    ent = 'the ' + ent
            return ent
        else:
            return id

    def get_path_ex_desc(self, id1, id2, looker_id=None):
        """Return a path description. If both ids are the same return the
        room description instead.
        """
        if id1 == id2:
            if looker_id is not None:
                desc = self.get_classed_prop(id1, 'desc', looker_id)
                extra_desc = \
                    self.get_classed_prop(id1, 'extra_desc', looker_id)
                return extra_desc if extra_desc is not None else desc
            desc = self.get_prop(id1, 'desc')
            return self.get_prop(id1, 'extra_desc', desc)
        desc_set = self._node_to_edges[id1][('path_to', id2)]['examine_desc']
        val = self.extract_classed_from_dict(desc_set, looker_id)
        if type(val) is dict and 'iter' in val:
            i = val['iter']
            val['iter'] = (i + 1) % len(val['data'])
            return val['data'][i]
        return val

    def path_to_desc(self, id1, id2):
        """get the description for a path from the perspective of id2"""
        rooms = self._node_to_edges[id2]
        for r in rooms:
            if r[0] == 'path_to' and r[1] == id1:
                if 'label' in rooms[r] and rooms[r]['label'] is not None:
                    if rooms[r]['full_label']:
                        return rooms[r]['label']
                    else:
                        return 'a path to the ' + rooms[r]['label']
        return False

    def node_to_desc_raw(self, id, from_id=False):
        if from_id:
            path_desc = self.path_to_desc(id, from_id)
            if path_desc is not False:
                return path_desc
        return self._node_to_desc[id]

    def name_prefix(self, id, txt, use_the):
        """Get the prefix to prepend an object with in text form"""
        # Get the preferred prefix type.
        pre = self.get_prop(id, 'name_prefix')

        if pre == '':
            return pre

        if use_the is True:
            return 'the'

        if pre is False or pre is None or pre == 'auto':
            txt = 'an' if txt[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
            return txt
        return pre

    # -- Messaging commands -- #

    def send_action(self, agent_id, action):
        """Parse the action and send it to the agent with send_msg"""
        if action['caller'] is None:
            val = \
                self.extract_classed_from_dict(action['txt'], agent_id, '')
            if type(val) is dict and 'iter' in val:
                i = val['iter']
                val['iter'] = (i + 1) % len(val['data'])
                extracted_text = val['data'][i]
            else:
                extracted_text = val
            self.send_msg(agent_id, extracted_text, action)
            return
        try:
            func_wrap = GRAPH_FUNCTIONS[action['caller']]
        except BaseException:
            func_wrap = CONSTRAINTS[action['caller']]
        try:
            t = func_wrap.format_observation(self, agent_id, action).rstrip()
        except Exception:
            return  # the agent doesn't accept observations
        t += ' '
        self.send_msg(agent_id, t, action)

    def send_msg(self, agent_id, txt, action=None):
        """Send an agent an action and a message"""
        if agent_id in self._node_to_text_buffer:
            if action is None:
                action = {
                    'caller': None,
                    'room_id': self.location(agent_id),
                    'txt': txt,
                }

            if not hasattr(self, '_node_to_observations'):
                # TODO remove when all the samples are converted
                self._node_to_observations = {}
            if agent_id not in self._node_to_observations:
                self._node_to_observations[agent_id] = []
            self._node_to_observations[agent_id].append(action)
            self._node_to_text_buffer[agent_id] += txt
        if agent_id in self.__message_callbacks:
            self.__message_callbacks[agent_id](self, action)

    def broadcast_to_room(self, action, exclude_agents=None, told_by=None):
        """send a message to everyone in a room"""
        if exclude_agents is None:
            exclude_agents = []
        else:
            exclude_agents = list(exclude_agents)

        agents_list, _descs = self.get_room_agents(action['room_id'])
        agents = set(agents_list)
        if 'actors' in action:
            # send message to the actor first
            actor = action['actors'][0]
            if actor in agents and actor not in exclude_agents:
                self.send_action(actor, action)
                exclude_agents.append(actor)

        action['present_agent_ids'] = agents
        for a in agents:
            if a in exclude_agents:
                continue
            self.send_action(a, action)

    # ----------------------------------------------------------------
    # TODO: Ideally, all functions below do not use the graph structure
    # directly but only the accessor functions (should not use self._node_* ).

    # -- Helpers -- #

    def clean_props(self, object_id):
        """ensures all necessary props are set on items that might not have
        been on earlier versions of graphworld
        """
        # TODO remove when all samples are converted
        size = self.get_prop(object_id, 'size')
        if size is None or type(size) == bool:
            self.set_prop(object_id, 'size', 1)
        contain_size = self.get_prop(object_id, 'contain_size')
        if contain_size is None or type(contain_size) == bool:
            self.set_prop(object_id, 'contain_size', 3)
        classes = self.get_prop(object_id, 'classes')
        if type(classes) == str:
            self.set_prop(object_id, 'classes', [classes])
        elif type(classes) != list:
            self.set_prop(object_id, 'classes', [])

    # -- Commands -- #

    def create(self, agent_id, params):
        # -- create commands: --
        # *create room kitchen  -> creates room with path from this room
        # *create path kitchen  -> create path to that room from this one
        # *create agent orc
        # *create object ring
        # *create key golden key
        # create lockable tower with golden key
        # create container box
        # create [un]freeze
        # create reset/load/save [fname]
        # create rename <node> <value>    <-- **crashes*
        # create delete <node>
        # create set_prop orc to health=5
        from parlai_internal.tasks.graph_world3.class_nodes import \
            create_thing, CLASS_NAMES
        if not self.has_prop(agent_id, 'agent'):
            return False, 'create'
        if params is None:
            return False, 'create'
        room_id = self.room(agent_id)
        all_params = ' '.join(params)
        txt = ' '.join(params[1:])
        resp = 'create ' + ' '.join(params)
        if not (all_params in ['save', 'load', 'freeze', 'unfreeze']):
            if txt == '':
                return False, resp
        if params[0] == 'print':
            ids = self.desc_to_nodes(txt, nearbyid=agent_id, nearbytype='all')
            if len(ids) == 0:
                self.send_msg(agent_id, "Not found.\n ")
                return False, resp
            id = ids[0]
            self.send_msg(agent_id,
                          id + " has:\n{}".format(self._node_to_prop[id]))
            return True, resp
        if params[0] == 'save':
            self.save_graph(txt)
            self.send_msg(agent_id, "[ saved: " + self._save_fname + ' ]\n')
            return True, resp
        if params[0] == 'load' or params[0] == 'reset':
            self.load_graph(txt)
            self.send_msg(agent_id, "[ loaded: " + self._save_fname + ' ]\n')
            return True, resp
        if params[0] == 'freeze':
            self.freeze(True)
            self.send_msg(agent_id, "Frozen.\n")
            return True, resp
        if params[0] == 'unfreeze':
            self.freeze(False)
            self.send_msg(agent_id, "Unfrozen.\n")
            return True, resp
        if params[0] in ['delete', 'del', 'rm']:
            ids = self.desc_to_nodes(txt, nearbyid=agent_id, nearbytype='all')
            if len(ids) == 0:
                self.send_msg("Not found.\n ")
                return False, resp
            id = ids[0]
            self.delete_node(id)
            self.send_msg(agent_id, "Deleted.\n")
            return True, resp
        if params[0] == 'rename':
            params = self.split_params(params[1:], 'to')
            to_ids = self.desc_to_nodes(params[0], nearbyid=agent_id,
                                        nearbytype='all')
            if len(to_ids) == 0:
                self.send_msg("Not found.\n ")
                return False, resp
            to_id = to_ids[0]
            self.set_desc(to_id, params[1])
            self.send_msg(agent_id, "Done.\n")
            return True, resp
        if params[0] == 'agent':
            create_thing(self, room_id, params[0], force=True, use_name=txt)
            self.send_msg(agent_id, "Done.\n")
            return True, resp
        if params[0] == 'room':
            new_id = self.add_node(txt, params[0])
            self.add_path_to(new_id, room_id)
            self.set_prop(new_id, 'contain_size', 2000)
            self.send_msg(agent_id, "Done.\n")
            return True, resp
        if params[0] == 'set_prop':
            params = self.split_params(params[1:], 'to')
            to_ids = self.desc_to_nodes(params[0], nearbyid=agent_id,
                                        nearbytype='all')
            if len(to_ids) == 0:
                self.send_msg("Not found.\n ")
                return False, resp
            to_id = to_ids[0]
            key = params[1]
            value = True
            if '=' in key:
                sp = key.split('=')
                if len(sp) != 2:
                    return False, resp
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
            return True, resp
        if (params[0] in CLASS_NAMES):
            if params[0] == 'key' and txt.find('key') == -1:
                self.send_msg(agent_id, "Keys must be called keys!\n")
                return False, resp
            create_thing(self, room_id, params[0], force=True, use_name=txt)
            self.send_msg(agent_id, "Done.\n")
            return True, resp
        if params[0] == 'lockable':
            ps = self.split_params(params[1:], 'with')
            if len(ps) != 2:
                return False, resp
            to_ids = self.desc_to_nodes(ps[0], nearbyid=agent_id,
                                        nearbytype='all')
            with_ids = self.desc_to_nodes(ps[1], nearbyid=agent_id,
                                          nearbytype='all')
            if len(to_ids) == 0 or len(with_ids) == 0:
                self.send_msg("Something was not found.\n ")
                return False, resp
            to_id = to_ids[0]
            with_id = with_ids[0]
            if not self.get_prop(with_id, 'key'):
                self.send_msg(agent_id, "You need to use a key!\n")
                return False, resp
            self.set_prop(to_id, 'locked_with', with_id)
            self.set_prop(to_id, 'locked', True)
            self.send_msg(agent_id, "Done.\n")
            return True, resp
        if params[0] == 'path':
            to_id = self.desc_to_nodes(txt)
            if to_id is False:
                return False, resp
            self.add_path_to(to_id, room_id)
            self.send_msg(agent_id, "Done.\n")
            return True, resp
        self.send_msg(agent_id, 'Create is not supported for: ' + resp)
        return False, resp

    # -- Create helpers -- #

    def split_params(self, params, word):
        if type(word) is str:
            word = {word}
        for w in word:
            search = ' {} '.format(w)
            phrase = ' '.join(params)
            if phrase.find(w) != -1:
                return phrase.split(search)
        return None

    # -- GraphFunction Helpers -- #

    def obj_fits(self, object_id, container_id):
        """Return if one object will fit into another"""
        size = self.get_prop(object_id, 'size')
        max_size = self.get_prop(container_id, 'contain_size')
        if size is None or max_size is None:
            # TODO log these kinds of things
            print('None compare between {} and {}'.format(
                object_id, container_id), self._node_to_prop)
        return size <= max_size

    def move_object(self, object_id, container_id):
        """Move an object from wherever it is into somewhere else"""
        size = self.get_prop(object_id, 'size')
        # Remove from the old container
        old_id = self.node_contained_in(object_id)
        if old_id is not None:
            contain_size = self.get_prop(old_id, 'contain_size')
            contain_size += size
            self.set_prop(old_id, 'contain_size', contain_size)
        # Put in the new container
        contain_size = self.get_prop(container_id, 'contain_size')
        contain_size -= size
        self.set_prop(container_id, 'contain_size', contain_size)
        self.add_contained_in(object_id, container_id)

    def die(self, id):
        """Update an agent into a dead state"""
        if not self.has_prop(id, 'agent'):
            return False
        if self.get_prop(id, 'is_player', False) is True:
            self.set_follow(id, None)
            room = self.location(id)
            add_text = ''
            contents = self.node_contains(id)
            if len(contents) > 0:
                add_text += \
                    '[*SPLIT*] Your travel companion leaves behind {}.'.format(
                        self.display_node_list(contents)
                    )
                for content in contents:
                    self.move_object(content, room)
            text = {
                'elf':
                    "Josephine collapses to the ground! "
                    "She looks like she's asleep except for an awful "
                    "stillness about her. Although her home is in the "
                    "mountains, she seems a part of the woods as well. You "
                    "hope to meet her again in the next life. And as you "
                    "think this, a bright white light filters through the "
                    "trees, engulfing the troll's peaceful form. You stare in "
                    "wonder: when the light fades away, so does Josephine!" +
                    add_text,
                'troll':
                    "You watch in horror as the elf drops lifelessly to the "
                    "ground. Lying there, Alixlior looks strangely peaceful "
                    "and as much a part of the woods as when alive. You know "
                    "that were your friend alive, you would be told not to "
                    "mourn and that all of this is a journey that will repeat "
                    "itself - where you will undoubtedly meet again. But for "
                    "now, this kind of blows. Before you can wonder if you "
                    "should conduct a burial ceremony, a strange white light "
                    "engulfs the elf's body, causing you to close your eyes. "
                    "When you open them, Alixlior is gone!" + add_text,
            }
            self.broadcast_to_room({
                'caller': None,
                'room_id': room,
                'txt': text,
            }, [id])
            self.set_prop(id, 'dead', True)
            self.delete_node(id)
            return True

        agent_desc = self.node_to_desc(id, use_the=True).capitalize()
        self.broadcast_to_room({
            'caller': None,
            'room_id': self.location(id),
            'txt': agent_desc + ' died!\n',
        }, [id])
        self.set_follow(id, None)
        self.set_prop(id, 'dead', True)
        self.delete_prop(id, 'agent')
        self.remove_class(id, 'agent')
        self.add_class(id, 'container')
        self.add_class(id, 'object')
        self.add_class(id, 'food')
        self.set_prop(id, 'container')
        self.set_prop(id, 'object')
        self.set_prop(id, 'food')
        self.set_prop(id, 'examined', True)
        return True

    def get_room_edge_text(self, room_descs, past=False):
        """Get text for all the edges outside of a room"""
        if len(room_descs) == 1:
            if past:
                return 'There was ' + room_descs[0] + '.\n'
            else:
                return 'There\'s ' + room_descs[0] + '.\n'
        default_paths = [path[10:] for path in room_descs
                         if path.startswith('a path to')]
        non_default_paths = [path for path in room_descs
                             if not path.startswith('a path to')]
        if len(default_paths) == 0:
            if past:
                s = 'There was '
            else:
                s = 'There\'s '
            s += self.display_desc_list(non_default_paths)
            s += '.\n'
        elif len(non_default_paths) == 0:
            if past:
                s = 'There were paths to '
            else:
                s = 'There are paths to '

            s += self.display_desc_list(default_paths)
            s += '.\n'
        else:
            if past:
                s = 'There was '
            else:
                s = 'There\'s '
            s += ", ".join(non_default_paths)
            if len(default_paths) == 1:
                s += ', and a path to '
            else:
                s += ', and paths to '
            s += self.display_desc_list(default_paths)
            s += '.\n'
        return s

    def get_room_object_text(self, object_descs, ents, past=False,
                             give_empty=True):
        """Get text for all the objects in a room"""
        s = ''
        tensed_is = 'was' if past else 'is'
        tensed_are = 'were' if past else 'are'
        loc = 'there' if past else 'here'
        if len(object_descs) == 0:
            if not give_empty:
                return ''
            s += 'It ' + tensed_is + ' empty.\n'
        elif len(object_descs) > 20:
            s += 'There ' + tensed_are + ' a lot of things here.\n'
        else:
            s += 'There\'s '
            s += self.display_desc_list(object_descs, ents)
            s += ' {}.\n'.format(loc)
        return s

    def get_room_agent_text(self, agent_descs, past=False):
        """Get text for all the agents in a room"""
        loc = 'there' if past else 'here'
        you_are = 'You were ' if past else 'You are '
        is_tensed = ' was ' if past else ' is '
        are_tensed = ' were ' if past else ' are '
        count = len(agent_descs)
        if count == 0:
            return ''
        elif count == 1:
            if agent_descs[0] == 'you':
                return you_are + loc + '.\n'
            else:
                return agent_descs[0].capitalize() + is_tensed + loc + '.\n'
        elif count == 2:
            all_descs = ' and '.join(agent_descs).capitalize()
            return all_descs + are_tensed + loc + '.\n'
        else:
            before_and = ', '.join(agent_descs[:-1]).capitalize()
            all_descs = before_and + ', and ' + agent_descs[-1]
            return all_descs + are_tensed + loc + '.\n'

    def get_room_edges(self, room_id):
        """Return a list of edges from a room and their current descriptions"""
        rooms = self._node_to_edges[room_id]
        rooms = [r[1] for r in rooms if r[0] == 'path_to']
        room_descs = [self.node_to_desc(ent, room_id) for ent in rooms]
        return rooms, room_descs

    def get_nondescribed_room_objects(self, room_id):
        """Return a list of objects in a room and their current descriptions"""
        objects = self.node_contains(room_id)
        objects = [o for o in objects if self.get_prop(o, 'object')]
        objects = [o for o in objects
                   if 'described' not in self.get_prop(o, 'classes')]
        object_descs = [self.node_to_desc(o) for o in objects]
        return objects, object_descs

    def get_room_objects(self, room_id):
        """Return a list of objects in a room and their current descriptions"""
        objects = self.node_contains(room_id)
        objects = [o for o in objects if self.get_prop(o, 'object')]
        object_descs = [self.node_to_desc(o) for o in objects]
        return objects, object_descs

    def get_nondescribed_room_agents(self, room):
        """Return a list of agents in a room and their current descriptions
        if those agents aren't described in the room text"""
        agents = self.node_contains(room)
        agents = [a for a in agents if self.get_prop(a, 'agent')]
        agents = [a for a in agents
                  if 'described' not in self.get_prop(a, 'classes')]
        agent_descs = [self.node_to_desc(a) for a in agents]
        return agents, agent_descs

    def get_room_agents(self, room):
        """Return a list of agents in a room and their current descriptions"""
        agents = self.node_contains(room)
        agents = [a for a in agents if self.get_prop(a, 'agent')]
        agent_descs = [self.node_to_desc(a) for a in agents]
        return agents, agent_descs

    def get_text(self, agent, clear_actions=True):
        txt = ''
        if agent in self._node_to_text_buffer:
            txt = self._node_to_text_buffer[agent]
        self._node_to_text_buffer[agent] = ''  # clear buffer
        if clear_actions:
            self._node_to_observations[agent] = []  # clear buffer
        return txt

    def cnt_obj(self, obj, c, ents=None):
        """Return a text form of the count of an object"""
        cnt = c[obj]
        if cnt == 1:
            return obj
        else:
            if ents is not None:
                if self.get_prop(ents[obj], 'plural') is not None:
                    words = self.get_prop(ents[obj], 'plural').split(' ')
                else:
                    words = (obj + 's').split(' ')
            f = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                 'nine', 'a lot of']
            rep = ['a', 'an', 'the']
            cnt = cnt - 2
            if cnt > 8:
                cnt = 8
            cnt = f[cnt]
            if words[0] in rep:
                return cnt + ' ' + ' '.join(words[1:])
            else:
                return cnt + ' ' + ' '.join(words)

    def display_desc_list(self, descs, ents=None):
        if len(descs) == 0:
            return 'nothing'
        if len(descs) == 1:
            return descs[0]
        c = Counter(descs)
        unique_items = set(descs)
        s = ''
        cnt = 0
        for obj in unique_items:
            s += self.cnt_obj(obj, c, ents)
            if len(unique_items) > 2 and cnt < len(unique_items) - 1:
                s += ','
            s += ' '
            cnt = cnt + 1
            if cnt == len(unique_items) - 1:
                s += 'and '
        return s.rstrip(' ')

    def display_node_list(self, l, from_id=False):
        desc_to_ent = {self.node_to_desc(ent, from_id): ent for ent in l}
        descs = [self.node_to_desc(ent, from_id) for ent in l]
        return self.display_desc_list(descs, desc_to_ent)

    def display_node(self, id):
        if self.get_prop(id, 'surface_type') == 'on':
            contents = self.node_contains(id)
            content_desc = self.display_node_list(contents)
            obj_desc = self.node_to_desc(id, use_the=True)
            return "There's {} on {}".format(content_desc, obj_desc)
        else:
            s = self.node_to_desc(id, use_the=True).capitalize() + ' contains '
            contents = self.node_contains(id)
            return s + self.display_node_list(contents) + '.\n'

    def help(self):
        txt = (
            '----------------------------------\n'
            'Commands:\n'
            'look (l, for short)\n'
            'inventory (i or inv, for short)\n'
            'examine <thing>\n'
            'status/health\n'
            'go <place>\n'
            'get/drop <object>\n'
            'eat/drink <object>\n'
            'wear/remove <object>\n'
            'wield/unwield <object>\n'
            'lock/unlock <object> with <object>\n'
            'follow <agent>\n'
            'hit <agent>\n'
            'put <object> in <container>\n'
            'get <object> from <container>\n'
            'give <object> to <agent>\n'
            'steal <object> from <agent>\n'
            'say "<thing you want to say>"\n'
            'tell <agent> "<something>"\n'
            '----------------------------------\n'
        )
        return txt

    def args_to_descs(self, args):
        loc = self.location(args[0])
        return [self.node_to_desc_raw(i, from_id=loc) for i in args]

    def get_possible_actions(self, my_agent_id='dragon', use_actions=None):
        """
        Get all actions that are possible from a given agent
        """
        # TODO rather than iterating over objects and checking validity for
        # all functions, iterate over functions and query valid objects
        if self.get_prop(my_agent_id, 'dead'):
            return []
        o = self.get_actionable_ids(my_agent_id)
        final_o = o
        for item in o:
            if self.get_prop(item, 'agent') \
                    or self.get_prop(item, 'container'):
                final_o = final_o.union(self.node_contains(item))
        actions = []
        use_functions = CANNONICAL_GRAPH_FUNCTIONS.items()
        if use_actions is not None:
            use_functions = [
                (fn, func) for (fn, func) in use_functions
                if fn in use_actions]
        for func_name, func in use_functions:
            if func_name in FUNC_IGNORE_LIST:
                continue  # Ignore functions we don't want to expose directly
            args = [my_agent_id]
            if func.valid_args(self, args):
                canon = func.get_canonical_form(self, args)
                actions.append(canon)
            for id1 in final_o:
                if id1 in args:
                    continue
                args = [my_agent_id, id1]
                if func.valid_args(self, args):
                    canon = func.get_canonical_form(self, args)
                    actions.append(canon)
                for id2 in final_o:
                    if id2 in args:
                        continue
                    args = [my_agent_id, id1, id2]
                    if func.valid_args(self, args):
                        canon = func.get_canonical_form(self, args)
                        actions.append(canon)
        return list(set(actions))

    @staticmethod
    def parse_static(inst):
        inst = inst.strip().split()
        symb_points = []
        in_quotes = False
        for i, symb in enumerate(inst):
            if '"' in symb:
                in_quotes = not in_quotes
            if symb.lower() in GRAPH_FUNCTIONS.keys() and not in_quotes:
                symb_points.append(i)
        symb_points.append(len(inst))
        return inst, symb_points

    @staticmethod
    def filter_actions(inst):
        ret_actions = []
        inst, symb_points = Graph.parse_static(inst)
        for i in range(len(symb_points) - 1):
            j, k = symb_points[i], symb_points[i + 1]
            if inst[j].lower() in GRAPH_FUNCTIONS.keys():
                ret_actions.append(' '.join(inst[j: k]))
        return ' '.join(ret_actions), ret_actions

    def parse(self, inst):
        return Graph.parse_static(inst)

    def canonical_action(self, agentid, inst):
        words = inst.split(' ')
        if (words[0].lower() in GRAPH_FUNCTIONS.keys()):
            func_wrap = GRAPH_FUNCTIONS[words[0].lower()]
            valid, args, _canon_args = \
                func_wrap.parse_text_to_args(self, agentid, words[1:])
            if not valid:
                return False, inst
            return True, func_wrap.get_canonical_form(self, args)
        return False, inst

    def get_reverse_action(self, agentid, action, old_g):
        """Attempts to get the reverse action for the given action. Makes no
        guarantees, could fail miserably"""
        inst, symb_points = self.parse(action)
        j, k = symb_points[0], symb_points[1]
        if (inst[j].lower() in GRAPH_FUNCTIONS.keys()):
            func_wrap = GRAPH_FUNCTIONS[inst[j].lower()]
            valid, args, _canon_args = \
                func_wrap.parse_text_to_args(old_g, agentid, inst[j + 1: k])
            rev_func_name, new_args = func_wrap.get_reverse(self, args)
            if rev_func_name is None or rev_func_name is False:
                return rev_func_name, ''
            rev_func_wrap = GRAPH_FUNCTIONS[rev_func_name]
            rev_action = rev_func_wrap.get_canonical_form(self, new_args)
            return True, rev_action
        return None, ''

    def valid_exec(self, agentid, inst=None):
        if inst is None:
            inst = agentid
            agentid = 'dragon'

        if self.get_prop(agentid, 'dead'):
            return False

        inst, symb_points = self.parse(inst)
        inst[0] = inst[0].lower()

        if len(inst) == 1 and (inst[0] == 'help'):
            return True

        if inst[0] not in GRAPH_FUNCTIONS.keys():
            return False

        for i in range(len(symb_points) - 1):
            j, k = symb_points[i], symb_points[i + 1]
            params = inst[j + 1: k]
            if (inst[j].lower() in GRAPH_FUNCTIONS.keys()):
                func_wrap = GRAPH_FUNCTIONS[inst[j].lower()]
                valid, args, _canon_args = \
                    func_wrap.parse_text_to_args(self, agentid, params)
                if not valid:
                    return False
            else:
                return False
        return True

    def parse_exec(self, agentid, inst=None):
        """ATTENTION: even if one of the actions is invalid, all actions
        before that will still be executed (the world state will be changed)!
        """
        if inst is None:
            inst = agentid
            agentid = 'dragon'

        if inst.startswith('look '):
            inst = 'look'

        if self.get_prop(agentid, 'dead'):
            self.send_msg(agentid,
                          "You are dead, you can't do anything, sorry.")
            return False, 'dead'
        inst, symb_points = self.parse(inst)
        inst[0] = inst[0].lower()

        hint_calls = ['a', 'actions', 'hints']
        if len(inst) == 1 and (inst[0] in hint_calls):
            # TODO remove the list of valid instructions from the main game,
            # Perhaps behind and admin gatekeeper of sorts
            self.send_msg(agentid, '\n'.join(
                sorted(self.get_possible_actions(agentid))) + '\n')
            return True, 'actions'
        if len(inst) == 1 and (inst[0] == 'help'):
            self.send_msg(agentid, self.help())
            return True, 'help'
        # if inst[0] == 'create':
        #     return self.create(agentid, inst[1:])
        if inst[0] not in GRAPH_FUNCTIONS.keys():
            # if Callbacks.handle_failed_parse_callbacks(
            #         inst[0].lower(), self, inst[1:], agentid):
            #     return False, inst[0]
            self.send_msg(agentid, "You can't {}.".format(inst[0]))
            return False, inst[0]
        acts = []
        for i in range(len(symb_points) - 1):
            j, k = symb_points[i], symb_points[i + 1]
            params = inst[j + 1: k]
            if (inst[j].lower() in GRAPH_FUNCTIONS.keys()):
                func_wrap = GRAPH_FUNCTIONS[inst[j].lower()]
                valid, args, _canon_args = \
                    func_wrap.parse_text_to_args(self, agentid, params)
                if not valid:
                    # if Callbacks.handle_failed_parse_callbacks(
                    #         func_wrap.get_name(), self, params, agentid):
                    #     return False, ' '.join(acts)
                    if type(args) is str:
                        self.send_msg(agentid, args)
                    else:
                        self.send_action(agentid, args)
                    return False, ' '.join(acts)
                new_act = func_wrap.get_canonical_form(self, args)
                success = func_wrap.handle(self, args)
                acts.append(new_act)
                if not success:
                    return False, ' '.join(acts)
            # elif Callbacks.handle_failed_parse_callbacks(
            #         inst[j].lower(), self, params, agentid):
            #     return False, ' '.join(acts)
            else:
                return False, ' '.join(acts)
        return True, ' '.join(acts)

    def update_world(self):
        """
        Move agents, do basic actions with npcs, handle telling npcs things
        """
        if self.freeze():
            return
        for agent_id in self._node_npcs:
            if not self.get_prop(agent_id, 'agent'):
                continue
            if self.get_prop(agent_id, 'dead'):
                continue
            # if agent_id in self._node_to_observations:
            #     for obs in self._node_to_observations[agent_id]:
            #         if obs['caller'] == 'say' and self.npc_say_respond:
            #             # TODO make a property of model agents rather than
            #             # manually present in the graph engine
            #             actor = obs['actors'][0]
            #             listeners = obs['present_agent_ids']
            #             name_guess = ' '.join(agent_id.split('_')[:-1])
            #             content_words = obs['content'].split(' ')
            #             resp_confidence = 1 / (len(listeners)-1)
            #             if self._last_tell_target[actor] == agent_id:
            #                 resp_confidence += 0.5
            #
            #             # Filter out the name from the word list if its there
            #             fin_content = []
            #             for word in content_words:
            #                 stripped = word.replace('.', '').replace('?', '')
            #                 if stripped == name_guess:
            #                     resp_confidence += 0.8
            #                 else:
            #                     fin_content.append(stripped)
            #
            #             # Talk if it makes sense to
            #             if resp_confidence > 0.9:
            #                 obs = obs.copy()
            #                 args = [
            #                     obs['actors'][0],
            #                     agent_id,
            #                     ' '.join(fin_content)
            #                 ]
            #                 self._last_tell_target[actor] = agent_id
            #                 # Callbacks.handle_callbacks(
            #                 #     TellFunction(), self, args)
            self.get_text(agent_id)
            did_hit = False
            possible_agents = self._node_contains[self.room(agent_id)]
            for other_agent_id in possible_agents:
                if self.get_prop(other_agent_id, 'is_player'):
                    aggression = self.get_prop(agent_id, 'aggression', 0)
                    if random.randint(0, 100) < aggression:
                        act = 'hit {}'.format(other_agent_id)
                        self.parse_exec(agent_id, act)
                        did_hit = True
            if not did_hit:
                # random movement for npcs..
                if random.randint(0, 100) < self.get_prop(agent_id, 'speed'):
                    cur_loc = self.room(agent_id)
                    locs = self.node_path_to(cur_loc)
                    loc = locs[random.randint(0, len(locs) - 1)]
                    act = 'go ' + self.node_to_desc_raw(loc, from_id=cur_loc)
                    self.parse_exec(agent_id, act)
