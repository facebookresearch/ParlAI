#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Worlds are the basic environments which define how agents interact with one another.

    ``World(object)`` provides a generic parent class, including ``__enter__``
    and ``__exit__`` statements which allow you to guarantee that the shutdown
    method is called.

    ``DialogPartnerWorld(World)`` provides a two-agent turn-based dialog setting.

    ``MultiAgentDialogWorld(World)`` provides a multi-agent setting.

    ``MultiWorld(World)`` creates a set of environments (worlds) for the same agent
    to multitask over, a different environment will be chosen per episode.

    ``HogwildWorld(World)`` is a container that creates another world within itself for
    every thread, in order to have separate simulated environments for each one.
    Each world gets its own agents initialized using the ``share()`` parameters
    from the original agents.

    ``BatchWorld(World)`` is a container for doing minibatch training over a world by
    collecting batches of N copies of the environment (each with different state).


All worlds are initialized with the following parameters:

    ``opt`` -- contains any options needed to set up the agent. This generally contains
        all command-line arguments recognized from core.params, as well as other
        options that might be set through the framework to enable certain modes.
    ``agents`` -- the set of agents that should be attached to the world,
        e.g. for DialogPartnerWorld this could be the teacher (that defines the
        task/dataset) and the learner agent. This is ignored in the case of
        sharing, and the shared parameter is used instead to initalize agents.
    ``shared`` (optional) -- if not None, contains any shared data used to construct
        this particular instantiation of the world. This data might have been
        initialized by another world, so that different agents can share the same
        data (possibly in different Processes).
"""

import copy
import queue
import random
import time

from abc import ABC, abstractmethod
from enum import auto, Enum
from functools import lru_cache
from typing import List, Dict, Any, Union, Tuple, Optional

try:
    from torch.multiprocessing import Process, Value, Semaphore, Queue
except ImportError:
    from multiprocessing import Process, Value, Semaphore, Queue  # noqa: F401

from parlai.core.agents import create_agents_from_shared, Agent
from parlai.core.loader import load_task_module, load_world_module
from parlai.core.message import Message
from parlai.core.metrics import aggregate_metrics
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.utils.batch import Batch
from parlai.utils.misc import Timer, display_messages
from parlai.tasks.tasks import ids_to_tasks


def validate(observation):
    """
    Make sure the observation table is valid, or raise an error.
    """
    if observation is not None and isinstance(observation, dict):
        return observation
    else:
        raise RuntimeError('Must return dictionary or Message object from act().')


class World(object):
    """
    Empty parent providing null definitions of API functions for Worlds.

    All children can override these to provide more detailed functionality.
    """

    def __init__(self, opt: Opt, agents=None, shared=None):
        self.id = opt['task']
        self.opt = copy.deepcopy(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            # Add passed in agents to world directly.
            self.agents = agents
        self.max_exs = None
        self.total_exs = 0
        self.total_epochs = 0
        self.total_parleys = 0
        self.time = Timer()

    def parley(self):
        """
        Perform one step of actions for the agents in the world.

        This is empty in the base class.
        """
        # TODO: mark as abstract?
        pass

    def getID(self):
        """
        Return the name of the world, typically the task the world encodes.
        """
        return self.id

    def display(self):
        """
        Return a string describing the current state of the world.

        Useful for monitoring and debugging. By default, display the messages between
        the agents.
        """
        if not hasattr(self, 'acts'):
            return ''
        return display_messages(
            self.acts,
            ignore_fields=self.opt.get('display_ignore_fields', ''),
            prettify=self.opt.get('display_prettify', False),
            max_len=self.opt.get('max_display_len', 1000),
        )

    def episode_done(self):
        """
        Whether the episode is done or not.
        """
        return False

    def epoch_done(self):
        """
        Whether the epoch is done or not.

        Not all worlds have the notion of an epoch, but this is useful for fixed
        training, validation or test sets.
        """
        return False

    def share(self):
        """
        Share the world.
        """
        shared_data = {}
        shared_data['world_class'] = type(self)
        shared_data['opt'] = self.opt
        shared_data['agents'] = self._share_agents()
        return shared_data

    def _share_agents(self):
        """
        Create shared data for agents.

        Allows other classes to create the same agents without duplicating the data
        (i.e. sharing parameters).
        """
        if not hasattr(self, 'agents'):
            return None
        shared_agents = [a.share() for a in self.agents]
        return shared_agents

    def get_agents(self):
        """
        Return the list of agents.
        """
        return self.agents

    def get_task_agent(self):
        """
        Return task agent, if applicable.
        """
        raise NotImplementedError('Implement in subworld')

    def get_model_agent(self):
        """
        Return model agent, if applicable.
        """
        raise NotImplementedError('Implement in subworld')

    def get_acts(self):
        """
        Return the last act of each agent.
        """
        return self.acts

    def get_time(self):
        """
        Return total training time.
        """
        return self.time.time()

    def get_total_exs(self):
        """
        Return total amount of examples seen by world.
        """
        return self.total_exs

    def get_total_epochs(self):
        """
        Return total amount of epochs on which the world has trained.
        """
        return self.total_epochs

    def __enter__(self):
        """
        Empty enter provided for use with ``with`` statement.

        e.g:

        .. code-block:: python

            with World() as world:
                for n in range(10):
                    n.parley()
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        After ``with`` statement, call shutdown.
        """
        self.shutdown()
        return False

    def num_examples(self):
        """
        Return the number of examples.

        Always 0 in the abstract world.
        """
        # TODO: mark as abstract?
        return 0

    def num_episodes(self):
        """
        Return the number of episodes.

        Always 0 in the abstract world.
        """
        # TODO: mark as abstract?
        return 0

    def reset(self):
        """
        Reset all agents in the world, and world statistics.
        """
        for a in self.agents:
            a.reset()
        self.max_exs = None
        self.total_exs = 0
        self.total_epochs = 0
        self.total_parleys = 0
        self.time.reset()

    def reset_metrics(self):
        """
        Reset metrics for all agents.
        """
        for a in self.agents:
            a.reset_metrics()

    def shutdown(self):
        """
        Perform any cleanup, if appropriate.
        """
        pass

    def update_counters(self):
        """
        Update how many epochs have completed.
        """
        self.total_parleys += 1
        if self.max_exs is None:
            if 'num_epochs' in self.opt and self.opt['num_epochs'] > 0:
                if self.num_examples:
                    self.max_exs = self.num_examples() * self.opt['num_epochs']
                else:
                    self.max_exs = -1
            else:
                self.max_exs = -1
        # when we know the size of the data
        if self.max_exs > 0 or self.num_examples():
            self.total_epochs = (
                self.total_parleys * self.opt.get('batchsize', 1) / self.num_examples()
            )
        # when we do not know the size of the data
        else:
            if self.epoch_done():
                self.total_epochs += 1


class DialogPartnerWorld(World):
    """
    Simple world for two agents communicating synchronously.

    This basic world switches back and forth between two agents, giving each agent one
    chance to speak per turn and passing that back to the other one.
    """

    def __init__(self, opt: Opt, agents=None, shared=None):
        if not ((agents is not None) ^ (shared is not None)):
            raise ValueError('You must supply either agents or shared, but not both.')
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            if len(agents) != 2:
                raise RuntimeError(
                    'There must be exactly two agents for this ' 'world.'
                )
            # Add passed in agents directly.
            self.agents = agents
        self.acts = [None] * len(self.agents)
        if self.agents is not None and len(self.agents) > 0:
            # Name the world after the first agent.
            self.id = self.get_task_agent().getID()

    def get_task_agent(self):
        """
        Return task agent.
        """
        return self.get_agents()[0]

    def get_model_agent(self):
        """
        Return model agent, if applicable.
        """
        return self.get_agents()[1]

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        acts = self.acts
        agents = self.agents
        acts[0] = agents[0].act()
        agents[1].observe(validate(acts[0]))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()

    def episode_done(self):
        """
        Only the first agent indicates when the episode is done.
        """
        if self.acts[0] is not None:
            return self.acts[0].get('episode_done', False)
        else:
            return False

    def epoch_done(self):
        """
        Only the first agent indicates when the epoch is done.
        """
        return self.get_task_agent().epoch_done()

    def report(self):
        """
        Report all metrics of all subagents.
        """

        # DEPRECATIONDAY: should we get rid of this option?
        metrics = {}
        for a in self.agents:
            if hasattr(a, 'report'):
                m = a.report()
                for k, v in m.items():
                    if k not in metrics:
                        # first agent gets priority in settings values for keys
                        # this way model can't e.g. override accuracy to 100%
                        metrics[k] = v
        if metrics:
            self.total_exs += metrics.get('exs', 0)
            return metrics

    @lru_cache(maxsize=1)
    def num_examples(self):
        """
        Return number of examples.
        """
        if hasattr(self.get_task_agent(), 'num_examples'):
            return self.get_task_agent().num_examples()
        return 0

    def num_episodes(self):
        """
        Return number of episodes.
        """
        if hasattr(self.get_task_agent(), 'num_episodes'):
            return self.get_task_agent().num_episodes()
        return 0

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for a in self.agents:
            a.shutdown()


class MultiAgentDialogWorld(World):
    """
    Basic world where each agent gets a turn in a round-robin fashion.

    Each agent receives as input the actions of all other agents since its last `act()`.
    """

    def __init__(self, opt: Opt, agents, shared=None):
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            # Add passed in agents directly.
            self.agents = agents
        self.acts = [None] * len(self.agents)

    def parley(self):
        """
        Perform a turn for every agent.

        For each agent, get an observation of the last action each of the other agents
        took. Then take an action yourself.
        """
        acts = self.acts
        for index, agent in enumerate(self.agents):
            acts[index] = agent.act()
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))
        self.update_counters()

    def get_task_agent(self):
        """
        Return task agent.
        """
        return self.get_agents()[0]

    def get_model_agent(self):
        """
        Return model agent.
        """
        return self.get_agents()[1]

    def epoch_done(self):
        """
        Return if the epoch is done for any subagent.
        """
        done = False
        for a in self.agents:
            if a.epoch_done():
                done = True
        return done

    def episode_done(self):
        """
        Return if the episode is done for any subagent.
        """
        done = False
        for a in self.agents:
            if a.episode_done():
                done = True
        return done

    def report(self):
        """
        Report metrics for all subagents.
        """
        metrics = {}
        for a in self.agents:
            if hasattr(a, 'report'):
                m = a.report()
                for k, v in m.items():
                    if k not in metrics:
                        # first agent gets priority in settings values for keys
                        # this way model can't e.g. override accuracy to 100%
                        metrics[k] = v
        if metrics:
            self.total_exs += metrics.get('exs', 0)
            return metrics

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for a in self.agents:
            a.shutdown()


class ExecutableWorld(MultiAgentDialogWorld):
    """
    World where messages from agents can be interpreted as _actions_.

    Actions result in changes in the environment (are executed). Hence a grounded
    simulation can be implemented rather than just dialogue between agents.
    """

    def __init__(self, opt: Opt, agents=None, shared=None):
        super().__init__(opt, agents, shared)
        self.init_world()

    def init_world(self):
        """
        Initialize the world.

        An executable world class should implement this function, otherwise the actions
        do not do anything (and it is the same as MultiAgentDialogWorld).
        """
        # TODO: mark as abstract
        pass

    def execute(self, agent, act):
        """
        Execute an action.

        An executable world class should implement this function, otherwise the actions
        do not do anything (and it is the same as MultiAgentDialogWorld).
        """
        pass

    def observe(self, agent, act):
        """
        Observe an action.

        An executable world class should implement this function, otherwise the
        observations for each agent are just the messages from other agents and not
        confitioned on the world at all (and it is thus the same as
        MultiAgentDialogWorld).
        """
        if agent.id == act['id']:
            return None
        else:
            return act

    def parley(self):
        """
        For each agent: act, execute and observe actions in world.
        """
        acts = self.acts
        for index, agent in enumerate(self.agents):
            # The agent acts.
            acts[index] = agent.act()
            # We execute this action in the world.
            self.execute(agent, acts[index])
            # All agents (might) observe the results.
            for other_agent in self.agents:
                obs = self.observe(other_agent, acts[index])
                if obs is not None:
                    other_agent.observe(obs)
        self.update_counters()


class MultiWorld(World):
    """
    Container for multiple worlds.

    Container for a set of worlds where each world gets a turn in a round-robin fashion.
    The same user_agents are placed in each, though each world may contain additional
    agents according to the task that world represents.
    """

    def __init__(self, opt: Opt, agents=None, shared=None, default_world=None):
        super().__init__(opt)
        self.worlds: List[World] = []
        for index, k in enumerate(opt['task'].split(',')):
            k = k.strip()
            if k:
                opt_singletask = copy.deepcopy(opt)
                opt_singletask['task'] = k
                if shared:
                    # Create worlds based on shared data.
                    s = shared['worlds'][index]
                    self.worlds.append(s['world_class'](s['opt'], None, s))
                else:
                    # Agents are already specified.
                    self.worlds.append(
                        create_task_world(
                            opt_singletask, agents, default_world=default_world
                        )
                    )
        self.world_idx = -1
        self.new_world = True
        self.parleys = -1
        self.random = opt.get('datatype', None) == 'train'
        # Make multi-task task probabilities.
        self.cum_task_weights = [1] * len(self.worlds)
        self.task_choices = range(len(self.worlds))
        weights = self.opt.get('multitask_weights', [1])
        sum = 0
        for i in self.task_choices:
            if len(weights) > i:
                weight = weights[i]
            else:
                weight = 1
            self.cum_task_weights[i] = weight + sum
            sum += weight

    def num_examples(self):
        """
        Return sum of each subworld's number of examples.
        """
        if not hasattr(self, 'num_exs'):
            worlds_num_exs = [w.num_examples() for w in self.worlds]
            if any(num is None for num in worlds_num_exs):
                self.num_exs = None
            else:
                self.num_exs = sum(worlds_num_exs)
        return self.num_exs

    def num_episodes(self):
        """
        Return sum of each subworld's number of episodes.
        """
        if not hasattr(self, 'num_eps'):
            worlds_num_eps = [w.num_episodes() for w in self.worlds]
            if any(num is None for num in worlds_num_eps):
                self.num_eps = None
            else:
                self.num_eps = sum(worlds_num_eps)
        return self.num_eps

    def get_agents(self):
        """
        Return the agents in the *current* subworld.
        """
        return self.worlds[self.world_idx].get_agents()

    def get_task_agent(self):
        """
        Not possible/well-defined in this setting.
        """
        return self.worlds[self.world_idx].get_task_agent()

    def get_model_agent(self):
        """
        Not implemented.
        """
        return self.worlds[self.world_idx].get_model_agent()

    def get_acts(self):
        """
        Return the acts in the *current* subworld.
        """
        return self.worlds[self.world_idx].get_acts()

    def share(self):
        """
        Share all the subworlds.
        """
        shared_data = {}
        shared_data['world_class'] = type(self)
        shared_data['opt'] = self.opt
        shared_data['worlds'] = [w.share() for w in self.worlds]
        return shared_data

    def epoch_done(self):
        """
        Return if *all* the subworlds are done.
        """
        for t in self.worlds:
            if not t.epoch_done():
                return False
        return True

    def parley_init(self):
        """
        Update the current subworld.

        If we are in the middle of an episode, keep the same world and finish this
        episode. If we have finished this episode, pick a new world (either in a random
        or round-robin fashion).
        """
        self.parleys = self.parleys + 1
        if self.world_idx >= 0 and self.worlds[self.world_idx].episode_done():
            self.new_world = True
        if self.new_world:
            self.new_world = False
            self.parleys = 0
            if self.random:
                # select random world
                self.world_idx = random.choices(
                    self.task_choices, cum_weights=self.cum_task_weights
                )[0]
            else:
                # do at most one full loop looking for unfinished world
                for _ in range(len(self.worlds)):
                    self.world_idx = (self.world_idx + 1) % len(self.worlds)
                    if not self.worlds[self.world_idx].epoch_done():
                        # if this world has examples ready, break
                        break

    def parley(self):
        """
        Parley the *current* subworld.
        """
        self.parley_init()
        self.worlds[self.world_idx].parley()
        self.update_counters()

    def display(self):
        """
        Display all subworlds.
        """
        if self.world_idx != -1:
            s = ''
            w = self.worlds[self.world_idx]
            if self.parleys == 0:
                s = '[world ' + str(self.world_idx) + ':' + w.getID() + ']\n'
            s = s + w.display()
            return s
        else:
            return ''

    def report(self):
        """
        Report aggregate metrics across all subworlds.
        """
        metrics = aggregate_metrics(self.worlds)
        self.total_exs += metrics.get('exs', 0)
        return metrics

    def reset(self):
        """
        Reset all subworlds.
        """
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        """
        Reset metrics in all subworlds.
        """
        for w in self.worlds:
            w.reset_metrics()

    def save_agents(self):
        """
        Save agents in all subworlds.
        """
        # Assumes all worlds have same agents, picks first to save.
        self.worlds[0].save_agents()


def _override_opts_in_shared(table, overrides):
    """
    Override all shared dicts.

    Looks recursively for ``opt`` dictionaries within shared dict and overrides any key-
    value pairs with pairs from the overrides dict.
    """
    if 'opt' in table:
        # change values if an 'opt' dict is available
        for k, v in overrides.items():
            table['opt'][k] = v
    for k, v in table.items():
        # look for sub-dictionaries which also might contain an 'opt' dict
        if type(v) == dict and k != 'opt' and 'opt' in v:
            _override_opts_in_shared(v, overrides)
        elif type(v) == list:
            for item in v:
                if type(item) == dict and 'opt' in item:
                    # if this is a list of agent shared dicts, we want to iterate
                    _override_opts_in_shared(item, overrides)
                else:
                    # if this is e.g. list of candidate strings, stop right away
                    break
    return table


class BatchWorld(World):
    """
    BatchWorld contains many copies of the same world.

    Create a separate world for each item in the batch, sharing
    the parameters for each.

    The underlying world(s) it is batching can be either
    ``DialogPartnerWorld``, ``MultiAgentWorld``, ``ExecutableWorld`` or
    ``MultiWorld``.
    """

    def __init__(self, opt: Opt, world):
        super().__init__(opt)
        self.opt = opt
        self.random = opt.get('datatype', None) == 'train'
        self.world = world
        self.worlds: List[World] = []
        for i in range(opt['batchsize']):
            # make sure that any opt dicts in shared have batchindex set to i
            # this lets all shared agents know which batchindex they have,
            # which is needed for ordered data (esp valid/test sets)
            shared = world.share()
            shared['batchindex'] = i
            for agent_shared in shared.get('agents', ''):
                agent_shared['batchindex'] = i
            # TODO: deprecate override_opts
            _override_opts_in_shared(shared, {'batchindex': i})
            self.worlds.append(shared['world_class'](opt, None, shared))
        self.batch_observations = [None] * len(self.world.get_agents())
        self.first_batch = None
        self.acts = [None] * len(self.world.get_agents())

    def batch_observe(self, index, batch_actions, index_acting):
        """
        Observe corresponding actions in all subworlds.
        """
        batch_observations = []
        for i, w in enumerate(self.worlds):
            agents = w.get_agents()
            observation = None
            if batch_actions[i] is None:
                # shouldn't send None, should send empty observations
                batch_actions[i] = [{}] * len(self.worlds)

            if hasattr(w, 'observe'):
                # The world has its own observe function, which the action
                # first goes through (agents receive messages via the world,
                # not from each other).
                observation = w.observe(agents[index], validate(batch_actions[i]))
            else:
                observation = validate(batch_actions[i])

            if index == index_acting:
                # self_observe is distinguished from a normal observe
                if hasattr(agents[index], 'self_observe'):
                    agents[index].self_observe(observation)
            else:
                observation = agents[index].observe(observation)

            # TODO: not so sure about this...
            if observation is None:
                raise ValueError('Agents should return what they observed.')
            batch_observations.append(observation)
        return batch_observations

    def batch_act(self, agent_idx, batch_observation):
        """
        Act in all subworlds.
        """
        # Given batch observation, do update for agents[index].
        # Call update on agent
        a = self.world.get_agents()[agent_idx]
        if hasattr(a, 'batch_act') and not (
            hasattr(a, 'use_batch_act')  # TODO: remove use_batch_act
            and not a.use_batch_act
        ):
            batch_actions = a.batch_act(batch_observation)
            # Store the actions locally in each world.
            for i, w in enumerate(self.worlds):
                acts = w.get_acts()
                acts[agent_idx] = batch_actions[i]
        else:
            # Reverts to running on each individually.
            batch_actions = []
            for w in self.worlds:
                agents = w.get_agents()
                acts = w.get_acts()
                acts[agent_idx] = agents[agent_idx].act()
                batch_actions.append(acts[agent_idx])
        return batch_actions

    def parley(self):
        """
        Parley in all subworlds.

        Usually with ref:`batch_act` and ref:`batch_observe`.
        """
        # Collect batch together for each agent, and do update.
        # Assumes DialogPartnerWorld, MultiAgentWorld, or MultiWorlds of them.
        num_agents = len(self.world.get_agents())
        batch_observations = self.batch_observations

        if hasattr(self.world, 'parley_init'):
            for w in self.worlds:
                w.parley_init()

        for agent_idx in range(num_agents):
            # The agent acts.
            batch_act = self.batch_act(agent_idx, batch_observations[agent_idx])
            self.acts[agent_idx] = batch_act
            # We possibly execute this action in the world.
            if hasattr(self.world, 'execute'):
                for w in self.worlds:
                    w.execute(w.agents[agent_idx], batch_act[agent_idx])
            # All agents (might) observe the results.
            for other_index in range(num_agents):
                obs = self.batch_observe(other_index, batch_act, agent_idx)
                if obs is not None:
                    batch_observations[other_index] = obs
        self.update_counters()

    def display(self):
        """
        Display the full batch.
        """
        s = "[--batchsize " + str(len(self.worlds)) + "--]\n"
        for i, w in enumerate(self.worlds):
            s += "[batch world " + str(i) + ":]\n"
            s += w.display() + '\n'
        s += "[--end of batch--]"
        return s

    def num_examples(self):
        """
        Return the number of examples for the root world.
        """
        return self.world.num_examples()

    def num_episodes(self):
        """
        Return the number of episodes for the root world.
        """
        return self.world.num_episodes()

    def get_total_exs(self):
        """
        Return the total number of processed episodes in the root world.
        """
        return self.world.get_total_exs()

    def getID(self):
        """
        Return the ID of the root world.
        """
        return self.world.getID()

    def get_agents(self):
        """
        Return the agents of the root world.
        """
        return self.world.get_agents()

    def get_task_agent(self):
        """
        Return task agent of the root world.
        """
        return self.world.get_task_agent()

    def get_model_agent(self):
        """
        Return model agent of the root world.
        """
        return self.world.get_model_agent()

    def episode_done(self):
        """
        Return whether the episode is done.

        A batch world is never finished, so this always returns `False`.
        """
        return False

    def epoch_done(self):
        """
        Return if the epoch is done in the root world.
        """
        # first check parent world: if it says it's done, we're done
        if self.world.epoch_done():
            return True
        # otherwise check if all shared worlds are done
        for world in self.worlds:
            if not world.epoch_done():
                return False
        return True

    def report(self):
        """
        Report metrics for the root world.
        """
        return self.world.report()

    def reset(self):
        """
        Reset the root world, and all copies.
        """
        self.world.reset()
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        """
        Reset metrics in the root world.
        """
        self.world.reset_metrics()

    def save_agents(self):
        """
        Save the agents in the root world.
        """
        # Because all worlds share the same parameters through sharing, saving
        # one copy would suffice
        self.world.save_agents()

    def shutdown(self):
        """
        Shutdown each world.
        """
        for w in self.worlds:
            w.shutdown()
        self.world.shutdown()


class DynamicBatchWorld(World):
    def __init__(self, opt: Opt, world: Union[DialogPartnerWorld, MultiWorld]):
        super().__init__(opt)
        self.opt = opt

        # agents is a placeholder just for super.reset
        self.agents = []

        # check some assumptions
        if isinstance(
            world, (ExecutableWorld, BatchWorld, HogwildWorld, MultiAgentDialogWorld)
        ):
            raise TypeError(
                'World must be a DialogPartnerWorld or a '
                'MultiWorld of DialogPartnerWorld'
            )

        if len(world.get_agents()) != 2:
            raise AssertionError(
                "Dynamic batch only works in a fixed dialog world with two agents."
            )

        if not hasattr(world.get_model_agent(), 'batch_act'):
            raise TypeError("Model agent doesn't have batch_act.")

        self.truncate = opt.get('text_truncate', None) or opt.get('truncate', None)
        self.l_truncate = opt.get('label_truncate', None) or opt.get('truncate', None)
        if self.truncate is None or self.truncate < 0:
            raise ValueError(
                'You must use --text-truncate or --truncate in order to use '
                'dynamic batching.'
            )

        # size of the buffer we will use to find worlds
        self._BUFFER_SIZE = 1021  # chosen as a prime number

        if opt['dynamic_batching'] == 'full':
            # full dynamic batching, we can grow our batchsize
            self.max_batch_size = self._BUFFER_SIZE
        else:
            # simple batchsort
            self.max_batch_size = opt['batchsize']

        # TODO: check to ensure the agent has self_observe
        shared = world.share()
        self.world = world
        # TODO: maybe generalize this
        self.max_words = (self.l_truncate + self.truncate) * opt['batchsize']

        # buffer worlds
        self.worlds = [
            shared['world_class'](opt, shared=shared) for _ in range(self._BUFFER_SIZE)
        ]

        self.reset()

    def reset(self):
        super().reset()
        self._obs = [None for _ in range(self._BUFFER_SIZE)]
        self._scores = [None for _ in range(self._BUFFER_SIZE)]

        self.number_parleys = 0
        self.total_exs = 0
        self.world.reset()
        self.rng = random.Random(4)
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        super().reset_metrics()
        self.world.reset_metrics()
        for w in self.worlds:
            w.reset_metrics()

    def epoch_done(self):
        return (
            self.world.epoch_done()
            or all(w.epoch_done() for w in self.worlds)
            and all(s is None for s in self._scores)
        )

    def num_examples(self):
        return self.world.num_examples()

    def num_episodes(self):
        return self.world.num_episodes()

    def _ceil(self, n):
        """
        Round to the nearest multiple of 8.

        TensorCores only work when a tensor is a multiple of 8 in almost all
        dimensions. This means all examples cost is related to their nearest
        multiple of 8.

        See https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/ for
        more information.
        """
        # round up to r, all things are equal
        from parlai.utils.torch import FP16_PAD_SIZE

        return ((n + FP16_PAD_SIZE - 1) // FP16_PAD_SIZE) * FP16_PAD_SIZE

    def _score(self, obs):
        if 'text_vec' in obs:
            # note that all examples have a cost that is based on their
            # nearest multiple of 4. We can therefore mix-and-match
            # anything with the same cost for increased stochasticity,
            # while not really wasting much padding.
            return tuple(
                self._ceil(len(obs[key]))
                for key in ['text_vec', 'labels_vec', 'eval_labels_vec']
                if key in obs
            )
        else:
            return None

    def _build_batch(self) -> Tuple[List[int], List[Message]]:
        """
        Build the batch of observations for the model.

        Return the indices into `self._obs`, which is a `self._BUFFER_SIZE` length array
        of observations for each subworld, as well as the list of messages

        :return: (batch, obs)
        """
        # first make sure that all the worlds are processed in the queue
        indices = []
        for i in range(self._BUFFER_SIZE):
            if self._scores[i] is not None:
                indices.append(i)
                continue
            if self.worlds[i].epoch_done():
                continue

            if hasattr(self.world, 'parley_init'):
                self.worlds[i].parley_init()

            act = self.worlds[i].get_task_agent().act()
            obs = self.worlds[i].get_model_agent().observe(act)
            self._obs[i] = obs

            self._scores[i] = self._score(obs)
            if self._scores[i] is not None:
                indices.append(i)
        # quick invariant checks
        assert len(indices) != 0, "DynamicBatchWorld ran out of data!"
        assert not any(self._scores[i] is None for i in indices)

        # sort all the indices by their score, so that we can find similarly lengthed
        # items in O(1)
        indices = sorted(indices, key=lambda i: self._scores[i] + (self.rng.random(),))

        # now let's build the batch
        batch: List[int] = []

        # start with a random item. indices_idx is the lookup into indices, but
        # index is the actual world.
        width = 0
        indices_idx = random.randint(0, len(indices) - 1)

        # we picked a random spot, but we can get better packing if we start at
        # the last example with the same score, since we always move down to
        # smaller examples.
        while indices_idx < len(indices) - 1 and (
            sum(self._scores[indices[indices_idx]])
            == sum(self._scores[indices[indices_idx + 1]])
        ):
            indices_idx += 1

        # quit early if we eat our full buffer
        while indices:
            index = indices[indices_idx]
            this_width = self._ceil(sum(self._scores[index]))
            new_width = max(width, this_width)
            # compute the cost of the new batch
            new_bsz = self._ceil(len(batch) + 1)
            new_words = new_width * new_bsz
            if new_words <= self.max_words and new_bsz <= self.max_batch_size:
                # cool, this one fits, let's add it
                width = new_width
                batch.append(index)
                indices.pop(indices_idx)
                indices_idx = max(indices_idx - 1, 0)
            else:
                # we'd overfill our buffer, give up
                break

        # Always have a batch size that's a multiple of 4, for fp16's sake.
        while len(batch) > 4 and len(batch) % 4 != 0:
            # pop off the shortest one. it's easiest to pack in later
            batch.pop(-1)

        # double check our assumed invariant
        assert self._ceil(width) * self._ceil(len(batch)) <= self.max_words
        assert self._ceil(len(batch)) <= self.max_batch_size

        return batch, [self._obs[i] for i in batch]

    def _observe_batch_act(self, batch: List[int], acts: List[Message]):
        """
        Observe the results of the model's batch act.

        :param batch:
            list of indices into `self._obs`

        :param acts:
            list of acts returned from the model's batch act
        """
        # broadcast the results back to all the models
        for i, act in zip(batch, acts):
            # we need to make sure that the teachers saw the result
            self.worlds[i].get_task_agent().observe(act)
            # and that the agent copies saw their own voice
            self.worlds[i].get_model_agent().self_observe(act)

            # move these worlds forward
            act = self.worlds[i].get_task_agent().act()
            obs = self.worlds[i].get_model_agent().observe(act)
            self._scores[i] = self._score(obs)
            self._obs[i] = obs

    def parley(self):
        # get the batch of observations
        batch, batch_obs = self._build_batch()
        # model batch act
        acts = self.world.get_model_agent().batch_act(batch_obs)
        # observe/self-observe
        self._observe_batch_act(batch, acts)
        # update metrics
        self.total_parleys += 1
        self.total_exs += len(batch)

    def get_total_epochs(self):
        return self.total_exs / self.num_examples()

    def report(self):
        return self.world.report()


class HogwildProcess(Process):
    """
    Process child used for ``HogwildWorld``.

    Each ``HogwildProcess`` contain its own unique ``World``.
    """

    def __init__(self, tid, opt: Opt, shared, sync):
        self.numthreads = opt['numthreads']
        opt = copy.deepcopy(opt)
        opt['numthreads'] = 1  # don't let threads create more threads!
        self.opt = opt
        self.shared = shared
        self.shared['threadindex'] = tid
        if 'agents' in self.shared:
            for a in self.shared['agents']:
                a['threadindex'] = tid
        self.sync = sync
        super().__init__(daemon=True)

    def run(self):
        """
        Run a parley loop.

        Runs normal parley loop for as many examples as this thread can get ahold of via
        the semaphore ``queued_sem``.
        """
        world = self.shared['world_class'](self.opt, None, self.shared)
        if self.opt.get('batchsize', 1) > 1:
            world = BatchWorld(self.opt, world)
        self.sync['threads_sem'].release()
        with world:
            while True:
                if self.sync['term_flag'].value:
                    break  # time to close
                self.sync['queued_sem'].acquire()
                self.sync['threads_sem'].release()

                # check if you need to reset before moving on
                if self.sync['epoch_done_ctr'].value < 0:
                    with self.sync['epoch_done_ctr'].get_lock():
                        # increment the number of finished threads
                        self.sync['epoch_done_ctr'].value += 1
                        if self.sync['epoch_done_ctr'].value == 0:
                            # make sure reset sem is clean
                            for _ in range(self.numthreads):
                                self.sync['reset_sem'].acquire(block=False)
                        world.reset()  # keep lock for this!

                while self.sync['epoch_done_ctr'].value < 0:
                    # only move forward once other threads have finished reset
                    time.sleep(0.1)

                # process an example or wait for reset
                if not world.epoch_done() or self.opt.get('datatype').startswith(
                    'train', False
                ):
                    # do one example if any available
                    world.parley()
                    with self.sync['total_parleys'].get_lock():
                        self.sync['total_parleys'].value += 1
                else:
                    # during valid/test, we stop parleying once at end of epoch
                    with self.sync['epoch_done_ctr'].get_lock():
                        # increment the number of finished threads
                        self.sync['epoch_done_ctr'].value += 1
                    # send control back to main thread
                    self.sync['threads_sem'].release()
                    # we didn't process anything
                    self.sync['queued_sem'].release()
                    # wait for reset signal
                    self.sync['reset_sem'].acquire()


class HogwildWorld(World):
    """
    Creates a separate world for each thread (process).

    Maintains a few shared objects to keep track of state:

    - A Semaphore which represents queued examples to be processed. Every call
      of parley increments this counter; every time a Process claims an
      example, it decrements this counter.

    - A Condition variable which notifies when there are no more queued
      examples.

    - A boolean Value which represents whether the inner worlds should shutdown.

    - An integer Value which contains the number of unprocessed examples queued
      (acquiring the semaphore only claims them--this counter is decremented
      once the processing is complete).
    """

    def __init__(self, opt: Opt, world):
        super().__init__(opt)
        self.inner_world = world
        self.numthreads = opt['numthreads']

        self.sync: Dict[str, Any] = {  # syncronization primitives
            # semaphores for counting queued examples
            'queued_sem': Semaphore(0),  # counts num exs to be processed
            'threads_sem': Semaphore(0),  # counts threads
            'reset_sem': Semaphore(0),  # allows threads to reset
            # flags for communicating with threads
            'reset_flag': Value('b', False),  # threads should reset
            'term_flag': Value('b', False),  # threads should terminate
            # counters
            'epoch_done_ctr': Value('i', 0),  # number of done threads
            'total_parleys': Value('l', 0),  # number of parleys in threads
        }

        self.threads: List[HogwildProcess] = []
        for i in range(self.numthreads):
            self.threads.append(HogwildProcess(i, opt, world.share(), self.sync))
            time.sleep(0.05)  # delay can help prevent deadlock in thread launches
        for t in self.threads:
            t.start()

        for _ in self.threads:
            # wait for threads to launch
            # this makes sure that no threads get examples before all are set up
            # otherwise they might reset one another after processing some exs
            self.sync['threads_sem'].acquire()  # type: ignore

        print(f'[ {self.numthreads} threads initialized ]')

    def display(self):
        """
        Unsupported operation.

        Raises a `NotImplementedError`.
        """
        self.shutdown()
        raise NotImplementedError(
            'Hogwild does not support displaying in-run'
            ' task data. Use `--numthreads 1`.'
        )

    def episode_done(self):
        """
        Unsupported operation.

        Raises a `RuntimeError`.
        """
        self.shutdown()
        raise RuntimeError('episode_done() undefined for hogwild')

    def epoch_done(self):
        """
        Return whether the epoch is finished.
        """
        return self.sync['epoch_done_ctr'].value == self.numthreads

    def parley(self):
        """
        Queue one item to be processed.
        """
        # schedule an example
        self.sync['queued_sem'].release()
        # keep main process from getting too far ahead of the threads
        # this way it can only queue up to numthreads unprocessed examples
        self.sync['threads_sem'].acquire()
        self.update_counters()

    def getID(self):
        """
        Return the inner world's ID.
        """
        return self.inner_world.getID()

    @lru_cache(maxsize=1)
    def num_examples(self):
        """
        Return the number of examples.
        """
        return self.inner_world.num_examples()

    def num_episodes(self):
        """
        Return the number of episodes.
        """
        return self.inner_world.num_episodes()

    def get_task_agent(self):
        """
        Return task agent of inner world.
        """
        return self.inner_world.get_task_agent()

    def get_model_agent(self):
        """
        Return model agent of inner world.
        """
        return self.inner_world.get_model_agent()

    def get_total_exs(self):
        """
        Return the number of processed examples.
        """
        return self.inner_world.get_total_exs()

    def get_total_epochs(self):
        """
        Return total amount of epochs on which the world has trained.
        """
        if self.max_exs is None:
            if 'num_epochs' in self.opt and self.opt['num_epochs'] > 0:
                if self.num_examples():
                    self.max_exs = self.num_examples() * self.opt['num_epochs']
                else:
                    self.max_exs = -1
            else:
                self.max_exs = -1
        if self.max_exs > 0:
            return (
                self.sync['total_parleys'].value
                * self.opt.get('batchsize', 1)
                / self.num_examples()
            )
        else:
            return self.total_epochs

    def report(self):
        """
        Report the inner world's metrics.
        """
        return self.inner_world.report()

    def save_agents(self):
        """
        Save the inner world's agents.
        """
        self.inner_world.save_agents()

    def reset(self):
        """
        Reset the inner world.
        """
        # set epoch done counter negative so all threads know to reset
        with self.sync['epoch_done_ctr'].get_lock():
            threads_asleep = self.sync['epoch_done_ctr'].value > 0
            self.sync['epoch_done_ctr'].value = -len(self.threads)
        if threads_asleep:
            # release reset semaphore only if threads had reached epoch_done
            for _ in self.threads:
                self.sync['reset_sem'].release()

    def reset_metrics(self):
        """
        Reset metrics for the inner world.
        """
        self.inner_world.reset_metrics()

    def shutdown(self):
        """
        Set shutdown flag and wake threads up to close themselves.
        """
        # set shutdown flag
        with self.sync['term_flag'].get_lock():
            self.sync['term_flag'].value = True

        # wake up each thread by queueing fake examples or setting reset flag
        for _ in self.threads:
            self.sync['queued_sem'].release()
            self.sync['reset_sem'].release()

        # make sure epoch counter is reset so threads aren't waiting for it
        with self.sync['epoch_done_ctr'].get_lock():
            self.sync['epoch_done_ctr'].value = 0

        # wait for threads to close
        for t in self.threads:
            t.join()
        self.inner_world.shutdown()


class QueueSignal(Enum):
    """
    Command signal used in the Queues for QueueWorld.

    BATCH:                normal result; a batch is in the Queue
    WORKER_FINISHED:      a worker is finished sending batch
    RESET:                tell a worker to reset
    RESET_METRICS:        tell a worker to reset_metrics
    REPORT:               tell a worker to report
    BEGIN:                tell a worker to begin again (valid worlds)
    TERMINATE:            tell a worker to terminate
    """

    BATCH = auto()
    WORKER_FINISHED = auto()
    RESET = auto()
    RESET_METRICS = auto()
    REPORT = auto()
    BEGIN = auto()
    TERMINATE = auto()


class PWorldProcess(Process):
    """
    Process that runs a PWorld.
    """

    def __init__(
        self,
        opt: Opt = None,
        world: Union[DialogPartnerWorld, MultiWorld] = None,
        pworld_class: type = None,
        produce_queue: Queue = None,
        report_queue: Queue = None,
        consume_queue: Queue = None,
        worker_idx: int = None,
    ):
        super().__init__(daemon=True)
        self.p_world: PWorld = pworld_class(
            opt, world, produce_queue, report_queue, consume_queue, worker_idx
        )
        self.produce_queue = produce_queue
        self.consume_queue = consume_queue
        self.worker_idx = worker_idx

    def run(self):
        """
        Run a PWorld to completion.

        The function first runs a world until `epoch_done()` is True, and then awaits
        further instruction
        """
        while True:
            while not self.p_world.epoch_done():
                self.p_world.parley()
            # one more time to observe
            self.produce_queue.put((QueueSignal.WORKER_FINISHED, None, self.worker_idx))
            while True:
                signal, batch = self.consume_queue.get()
                if signal == QueueSignal.BATCH:
                    # TODO: determine why this happens
                    self.p_world.consume_act(batch)
                elif signal == QueueSignal.REPORT:
                    self.p_world.enqueue_report()
                elif signal == QueueSignal.BEGIN:
                    self.p_world.begin()
                    break
                elif signal == QueueSignal.RESET:
                    self.p_world.reset()
                elif signal == QueueSignal.TERMINATE:
                    return


class PWorld(ABC, World):
    """
    A PWorld is a world that can be used in a subprocess of the QueueWorld.

    A subclass must implement two methods:
        1. ``produce_observation()``:
            returns an observation for an agent; this can
            be either a List[Message] or a Batch, depending on the agent
        2. ``consume_act(act)``:
            Consumes a model's action, resulting from the observation in
            ``produce_observation``

    The PWorld places produced observations on its ``produe_queue``;
    it obtains actions from its ``consume_queue``. Finally, it places
    reports on its ``report_queue``.
    """

    def __init__(
        self,
        opt: Opt,
        world: Union[DialogPartnerWorld, MultiWorld],
        produce_queue: Queue,
        report_queue: Queue,
        consume_queue: Queue,
        worker_idx: int,
    ):
        from torch import Tensor

        World.__init__(self, opt)
        self.world = world
        self.produce_queue = produce_queue
        self.report_queue = report_queue
        self.consume_queue = consume_queue
        self.worker_idx = worker_idx
        self.produce_batch: bool = True
        self.buffers: Dict[str, Tensor] = {}
        self.num_prod = 0
        self.num_consume = 0
        self.begun = False

    @abstractmethod
    def produce_observation(self) -> Union[List[Message], Batch]:
        """
        Produce an observation.

        Rtype depends on the subclassing world

        :return: an observation to put on the produce_queue
        """

    @abstractmethod
    def consume_act(self, act: List[Message]):
        """
        Consume an act from the consume_queue.

        :param act:
            the action from
        """

    def _producing(self):
        """
        Determine whether this world should produce.
        """
        return self.num_prod == self.num_consume and self.begun

    def parley(self):
        """
        Produce a batch observation.

        Consume a batch act.
        """
        if self._producing():
            self.num_prod += 1
            obs = self.produce_observation()
            self.produce_queue.put_nowait((QueueSignal.BATCH, obs, self.worker_idx))
        queue_signal, act = self.consume_queue.get()
        if queue_signal == QueueSignal.RESET:
            # RESET
            self.reset()
        elif queue_signal == QueueSignal.RESET_METRICS:
            # RESET
            self.reset_metrics()
        elif queue_signal == QueueSignal.REPORT:
            self.enqueue_report()
        elif queue_signal == QueueSignal.TERMINATE:
            self.shutdown()
        elif queue_signal == QueueSignal.BEGIN:
            # no op
            self.begin()
        else:  # QueueSignal.BATCH
            self.num_consume += 1
            self.consume_act(act)
            self.update_counters()

    def _handle_buffers(self, batch_obs: List[Message]) -> Batch:
        """
        Batchify list of observations and handle tensor buffers.

        :param batch_obs: a list of Messages

        :return: batch
        """
        from torch import Tensor

        updated_buffers = {}
        resized = {}
        batch: Batch = self.world.get_model_agent().batchify(batch_obs)
        for k, v in batch.items():
            if isinstance(v, Tensor):
                try:
                    self.buffers[k].copy_(v)
                    self.buffers[k].resize_(v.size())
                    resized[k] = v.size()
                    setattr(batch, k, None)
                except (KeyError, RuntimeError):
                    # either not broadcastable
                    # or not in self.buffers
                    self.buffers[k] = v
                    self.buffers[k].share_memory_()
                    updated_buffers[k] = v
        batch.updated_buffers = updated_buffers
        batch.resized = resized
        return batch

    def enqueue_report(self):
        self.report_queue.put_nowait(self.report())

    def reset(self):
        super().reset()
        self.buffers = {}
        self.num_prod = 0
        self.num_consume = 0

    def begin(self):
        """
        Signal world to begin.

        World is ready to produce batches
        """
        self.begun = True


class PBatchWorld(PWorld, BatchWorld):
    """
    P Batch World.
    """

    def __init__(
        self,
        opt: Opt,
        world: Union[DialogPartnerWorld, MultiWorld],
        produce_queue: Queue,
        report_queue: Queue,
        consume_queue: Queue,
        worker_idx: int,
    ):
        PWorld.__init__(
            self, opt, world, produce_queue, report_queue, consume_queue, worker_idx
        )
        BatchWorld.__init__(self, opt, world)

    def produce_observation(self) -> Union[List[Message], Batch]:
        teacher_idx = 0
        agent_idx = 1
        if hasattr(self.world, 'parley_init'):
            for w in self.worlds:
                w.parley_init()
        batch_act = self.batch_act(teacher_idx, None)
        batch_obs = self.batch_observe(agent_idx, batch_act, teacher_idx)
        if hasattr(self.world.get_model_agent(), 'batchify'):
            batch_obs = self._handle_buffers(batch_obs)
        return batch_obs

    def consume_act(self, act: List[Message]):
        agent_idx = 1
        # Agent does not see *all* observations, only valid ones.
        # So, need to pad the act with some empty messages
        act += [
            Message({'id': self.get_model_agent().getID(), 'episode_done': False})
            for _ in range(self.opt['batchsize'] - len(act))
        ]
        for other_idx in range(len(self.world.get_agents())):
            self.batch_observe(other_idx, act, agent_idx)

    def reset(self):
        PWorld.reset(self)
        BatchWorld.reset(self)


class PDynamicBatchWorld(PWorld, DynamicBatchWorld):
    """
    P Dynamic Batchworld.
    """

    def __init__(
        self,
        opt: Opt,
        world: Union[DialogPartnerWorld, MultiWorld],
        produce_queue: Queue,
        report_queue: Queue,
        consume_queue: Queue,
        worker_idx: int,
    ):
        PWorld.__init__(
            self, opt, world, produce_queue, report_queue, consume_queue, worker_idx
        )
        DynamicBatchWorld.__init__(self, opt, world)
        self.batch_indices: Optional[List[int]] = None

    def produce_observation(self) -> Union[List[Message], Batch]:
        assert self.batch_indices is None
        batch, batch_obs = self._build_batch()
        self.batch_indices = batch
        if hasattr(self.world.get_model_agent(), 'batchify'):
            batch_obs = self._handle_buffers(batch_obs)
        return batch_obs

    def consume_act(self, act: List[Message]):
        assert self.batch_indices is not None
        self._observe_batch_act(self.batch_indices, act)
        self.batch_indices = None

    def reset(self):
        DynamicBatchWorld.reset(self)
        PWorld.reset(self)
        self.batch_indices = None


class QueueWorld(World):
    """
    A QueueWorld for background preprocessing.

    The QueueWorld spawns ``opt['num_workers']`` subprocesses, which handle
    preprocessing of batches off the main thread.

    The QueueWorld has a number of important structures:
        1. ``consume_queues`` - each subprocess has its own consume queue
            for model's actions.
        2. ``produce_queue`` - the queue from which QueueWorld pulls batches
        3. ``report_queue`` - the queue from which QueueWorld pulls reports
        4. ``buffers`` - tensor buffers for each attr in a Batch that is a tensor.
            This allows buffers to be reused amongst processes and signficantly reduces
            overhead. The reason for this is that copying tensors to/from queues is
            _costly_

    Most importantly - please mind your Ps and Qs :)
    """

    def __init__(self, opt: Opt, world: World, pworld_class: type):
        super().__init__(opt)
        # QueueWorld init
        from torch import Tensor

        self.world: World = world
        self.agents: List[Agent] = world.get_agents()
        self.processes: List[Process] = []
        self.consume_queues: List[Queue] = []
        self.worlds: List[World] = []
        self.produce_queue: Queue = Queue()
        self.report_queue: Queue = Queue()
        self.buffers: List[Dict[str, Tensor]] = []
        self.finished_workers: List[int] = []
        self.training = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.num_workers = opt['num_workers'] if self.training else 1
        self.init_parley = True
        self.num_consume = 0
        self._init_workers(opt, world, pworld_class)

    def num_examples(self):
        """
        Return the number of examples for the root world.
        """
        return self.world.num_examples()

    def num_episodes(self):
        """
        Return the number of episodes for the root world.
        """
        return self.world.num_episodes()

    def get_total_exs(self):
        """
        Return the total number of processed episodes in the root world.
        """
        return self.world.get_total_exs()

    def getID(self):
        """
        Return the ID of the root world.
        """
        return self.world.getID()

    def get_agents(self):
        """
        Return the agents of the root world.
        """
        return self.world.get_agents()

    def get_task_agent(self):
        """
        Return task agent of the root world.
        """
        return self.world.get_task_agent()

    def get_model_agent(self):
        """
        Return model agent of the root world.
        """
        return self.world.get_model_agent()

    def episode_done(self):
        """
        Return whether the episode is done.

        A batch world is never finished, so this always returns `False`.
        """
        return False

    def epoch_done(self):
        """
        Override to account for finished workers.
        """
        subworlds_done = self.world.epoch_done() or all(
            w.epoch_done() for w in self.worlds
        )
        return subworlds_done or len(self.finished_workers) == self.num_workers

    def save_agents(self):
        """
        Save the agents in the root world.
        """
        self.world.save_agents()

    def _init_workers(self, opt: Opt, world: World, pworld_class: type):
        """
        Initialize the processes.

        :param opt:
            Opt dictionary

        :param world:
            Master world

        :param pworld_class:
            Which PWorld to use.
        """
        for worker_idx in range(self.num_workers):
            consume_queue: Queue = Queue()
            shared = world.share()
            subworld = shared['world_class'](opt, None, shared)
            self.processes.append(
                PWorldProcess(
                    opt=opt,
                    world=subworld,
                    pworld_class=pworld_class,
                    produce_queue=self.produce_queue,
                    report_queue=self.report_queue,
                    consume_queue=consume_queue,
                    worker_idx=worker_idx,
                )
            )
            self.worlds.append(subworld)
            self.consume_queues.append(consume_queue)
            self.buffers.append({})
        for p in self.processes:
            p.start()

    def _maybe_begin_processes(self):
        """
        Send BEGIN signal to workers.

        Useful for validation worlds used during training.
        """
        if self.init_parley:
            self.init_parley = False
            for q in self.consume_queues:
                q.put_nowait((QueueSignal.BEGIN, None))

    def _maybe_parley_init(self):
        """
        Call ``parley_init`` if necessary.
        """
        if hasattr(self.world, 'parley_init'):
            for w in self.worlds:
                w.parley_init()

    def _poll_produce_queue(
        self
    ) -> Tuple[QueueSignal, Union[List[Message], Batch], int]:
        """
        Poll the QueueWorld's produce queue.

        Continuously poll the queue until we receive a Batch. Mark finished workers along the way.

        :return: (signal, batch_obs, idx)
            signal:    the command signal from the queue
            batch_obs: a batch from one of the producers
            idx:       the worker index that produced the batch
        """
        batch_obs: Union[List[Message], Batch] = None
        queue_signal: QueueSignal = None
        worker_idx: int = None
        while queue_signal != QueueSignal.BATCH and not self.epoch_done():
            queue_signal, batch_obs, worker_idx = self.produce_queue.get()
            if queue_signal == QueueSignal.WORKER_FINISHED:
                self.finished_workers.append(worker_idx)
        if queue_signal == QueueSignal.BATCH:
            self.num_consume += 1
        return queue_signal, batch_obs, worker_idx

    def _maybe_handle_buffers(self, batch_obs: Batch, worker_idx: int) -> Batch:
        """
        Handle batch_obs buffers.

        Essentially, housekeeping on the buffers used for the given `worker_idx`.
            1. Update QueueWorld buffers for `worker_idx`
            2. Resize QueueWorld buffers for `worker_idx`
            3. Set appropriate batch attrs with QueueWorld buffers

        :param batch_obs:
            The batch of observations

        :param worker_idx:
            The worker from which the batch came

        :return: batch_obs, a batch post-buffer-housekeeping
        """
        if hasattr(self.world.get_model_agent(), 'batchify'):
            for key, buff in batch_obs.updated_buffers.items():
                # Update
                self.buffers[worker_idx][key] = buff
            for key, new_size in batch_obs.resized.items():
                self.buffers[worker_idx][key].resize_(new_size)
            for key, buff in self.buffers[worker_idx].items():
                setattr(batch_obs, key, buff)
        return batch_obs

    def parley(self):
        """
        QueueWorld parley.

        QueueWorld receives a batch from its produce queue, gives it to an agent to act,
        then puts it back on the appropriate consume queue.
        """
        self._maybe_begin_processes()
        self._maybe_parley_init()
        queue_signal, batch_obs, worker_idx = self._poll_produce_queue()
        if self.epoch_done():
            return
        batch_obs = self._maybe_handle_buffers(batch_obs, worker_idx)
        batch_acts = self.batch_act(batch_obs)
        self.consume_queues[worker_idx].put_nowait((QueueSignal.BATCH, batch_acts))
        self.update_counters(batch_obs)

    def batch_act(
        self, batch_observation: Union[List[Message], Batch]
    ) -> List[Message]:
        """
        Batch act on agent.

        :param batch_observation:
            The observation that the agent will act upon

        :return: batch_actions
        """
        a = self.world.get_model_agent()
        if hasattr(a, 'batch_act'):
            batch_actions = a.batch_act(batch_observation)
        else:
            batch_actions = []
            for w in self.worlds:
                agent = w.get_model_agent()
                acts = w.get_acts()
                acts[1] = agent.act()
                batch_actions.append(acts[1])
        return batch_actions

    def shutdown(self):
        """
        Shutdown subworlds and close all queues.
        """
        for w in self.worlds:
            w.shutdown()
        self.world.shutdown()
        for q in self.consume_queues:
            q.put_nowait((QueueSignal.TERMINATE, None))
        for q in [self.produce_queue, self.report_queue] + self.consume_queues:
            q.close()
        for p in self.processes:
            p.terminate()

    def reset(self):
        """
        Reset each subworld and additionally each individual process.
        """
        super().reset()
        self.world.reset()
        for w in self.worlds:
            w.reset()
        while True:
            try:
                self.produce_queue.get_nowait()
            except queue.Empty:
                break
        for q in self.consume_queues:
            q.put((QueueSignal.RESET, None))
        self.finished_workers = []
        self.init_parley = True

    def reset_metrics(self):
        """
        Reset metrics.

        (TODO possibly reset process metrics)
        """
        super().reset_metrics()
        self.world.reset_metrics()
        for q in self.consume_queues:
            q.put_nowait((QueueSignal.RESET_METRICS, None))

    def report(self) -> Dict[str, Union[str, int, Dict]]:
        """
        Aggregate reports from each process.

        :return: report
        """
        for q in self.consume_queues:
            q.put((QueueSignal.REPORT, None))
        reports: List[Dict] = []
        while len(reports) < self.num_workers:
            reports.append(self.report_queue.get())
        # TODO: Aggregate correctly
        report = {'exs': sum(r.get('exs', 0) for r in reports)}
        report.update(self.world.report())
        return report

    def get_total_epochs(self) -> float:
        """
        Return total number of epochs processed.
        """
        return self.total_exs / self.num_examples()

    def update_counters(self, batch: Batch):
        """
        Update world counters.

        Must account for dynamic batching, where number exs processed != batchsize

        :param batch:
            batch of observations
        """
        self.total_exs += batch.batchsize
        self.total_parleys += 1


################################################################################
# Functions for creating tasks/worlds given options.
################################################################################
def _create_task_agents(opt: Opt):
    """
    Create task agent(s) for the given task name.

    It does this by calling the create_agent function in agents.py of the given task. If
    create_agents function does not exist, it just looks for the teacher (agent) class
    defined by the task name directly.  (This saves the task creator bothering to define
    the create_agents function when it is not needed.)
    """
    my_module = load_task_module(opt['task'])
    try:
        # Tries to call the create_agent function in agents.py
        task_agents = my_module.create_agents(opt)  # type: ignore

    except AttributeError:
        # Create_agent not found, so try to create the teacher directly.
        return create_task_agent_from_taskname(opt)
    if type(task_agents) != list:
        task_agents = [task_agents]
    return task_agents


def create_task_world(opt: Opt, user_agents, default_world=None):
    """
    Instantiate a world with the supplied options and user agents.

    (A world factory.)
    """
    task_agents = _create_task_agents(opt)
    world_class = load_world_module(
        opt['task'],
        opt.get('interactive_task', False),
        len(user_agents + task_agents),
        default_world=default_world,
    )

    return world_class(opt, task_agents + user_agents)


def create_task(opt: Opt, user_agents, default_world=None):
    """
    Create a world + task_agents (aka a task).

    Assuming ``opt['task']="task_dir:teacher_class:options"`` e.g. ``"babi:Task1k:1"``
    or ``"#babi-1k"`` or ``"#QA"``, see ``parlai/tasks/tasks.py`` and see
    ``parlai/tasks/task_list.py`` for list of tasks.
    """
    task = opt.get('task')
    if not task:
        raise RuntimeError(
            'No task specified. Please select a task with ' + '--task {task_name}.'
        )
    if type(user_agents) != list:
        user_agents = [user_agents]

    # Convert any hashtag task labels to task directory path names.
    # (e.g. "#QA" to the list of tasks that are QA tasks).
    opt = copy.deepcopy(opt)
    opt['task'] = ids_to_tasks(opt['task'])
    print('[creating task(s): ' + opt['task'] + ']')

    # check if single or multithreaded, and single-example or batched examples
    if ',' not in opt['task']:
        # Single task
        world = create_task_world(opt, user_agents, default_world=default_world)
    else:
        # Multitask teacher/agent
        # TODO: remove and replace with multiteachers only?
        world = MultiWorld(opt, user_agents, default_world=default_world)

    if opt.get('numthreads', 1) > 1:
        # use hogwild world if more than one thread requested
        # hogwild world will create sub batch worlds as well if bsz > 1
        world = HogwildWorld(opt, world)
    elif opt.get('num_workers', 1) > 1:
        if opt.get('dynamic_batching') and opt.get('batchsize', 1) > 1:
            world = QueueWorld(opt, world, PDynamicBatchWorld)
        else:
            world = QueueWorld(opt, world, PBatchWorld)
    elif opt.get('batchsize', 1) > 1 and opt.get('dynamic_batching'):
        world = DynamicBatchWorld(opt, world)
    elif opt.get('batchsize', 1) > 1:
        # otherwise check if should use batchworld
        world = BatchWorld(opt, world)

    return world
