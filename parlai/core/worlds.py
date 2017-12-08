# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""This class defines the basic environments that define how agents interact
with one another.

    ``World(object)`` provides a generic parent class, including ``__enter__``
    and ``__exit__`` statements which allow you to guarantee that the shutdown
    method is called and KeyboardInterrupts are less noisy (if desired).

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
import importlib
import random
from tqdm import tqdm

try:
    from torch.multiprocessing import Process, Value, Condition, Semaphore
except ImportError:
    from multiprocessing import Process, Value, Condition, Semaphore
from parlai.core.agents import _create_task_agents, create_agents_from_shared
from parlai.tasks.tasks import ids_to_tasks


def validate(observation):
    """Make sure the observation table is valid, or raise an error."""
    if observation is not None and type(observation) == dict:
        return observation
    else:
        raise RuntimeError('Must return dictionary from act().')


def display_messages(msgs):
    """Returns a string describing the set of messages provided"""
    lines = []
    episode_done = False
    for index, msg in enumerate(msgs):
        if msg is None:
            continue
        if msg.get('episode_done'):
            episode_done = True
        # Possibly indent the text (for the second speaker, if two).
        space = ''
        if len(msgs) == 2 and index == 1:
            space = '   '
        if msg.get('reward', None) is not None:
            lines.append(space + '[reward: {r}]'.format(r=msg['reward']))
        if type(msg.get('image')) == str:
            lines.append(msg['image'])
        if msg.get('text', ''):
            ID = '[' + msg['id'] + ']: ' if 'id' in msg else ''
            lines.append(space + ID + msg['text'])
        if msg.get('labels'):
            lines.append(space + ('[labels: {}]'.format(
                        '|'.join(msg['labels']))))
        if msg.get('eval_labels'):
            lines.append(space + ('[eval_labels: {}]'.format(
                        '|'.join(msg['eval_labels']))))
        if msg.get('label_candidates'):
            cand_len = len(msg['label_candidates'])
            if cand_len <= 10:
                lines.append(space + ('[cands: {}]'.format(
                        '|'.join(msg['label_candidates']))))
            else:
                # select five label_candidates from the candidate set,
                # can't slice in because it's a set
                cand_iter = iter(msg['label_candidates'])
                display_cands = (next(cand_iter) for _ in range(5))
                # print those cands plus how many cands remain
                lines.append(space + ('[cands: {}{}]'.format(
                        '|'.join(display_cands),
                        '| ...and {} more'.format(cand_len - 5)
                        )))
    if episode_done:
        lines.append('- - - - - - - - - - - - - - - - - - - - -')
    return '\n'.join(lines)


class World(object):
    """Empty parent providing null definitions of API functions for Worlds.
    All children can override these to provide more detailed functionality."""

    def __init__(self, opt, agents=None, shared=None):
        self.id = opt['task']
        self.opt = copy.deepcopy(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            # Add passed in agents to world directly.
            self.agents = agents

    def parley(self):
        """The main method, that does one step of actions for the agents
        in the world. This is empty in the base class.
        """
        pass

    def getID(self):
        """Return the name of the world, typically the task the world encodes."""
        return self.id

    def display(self):
        """Returns a string describing the current state of the world.

        Useful for monitoring and debugging.
        By default, display the messages between the agents."""
        if not hasattr(self, 'acts'):
            return ''
        return display_messages(self.acts)

    def episode_done(self):
        """Whether the episode is done or not."""
        return False

    def epoch_done(self):
        """Whether the epoch is done or not.

        Not all worlds have the notion of an epoch, but this is useful
        for fixed training, validation or test sets.
        """
        return False

    def share(self):
        shared_data = {}
        shared_data['world_class'] = type(self)
        shared_data['opt'] = self.opt
        shared_data['agents'] = self._share_agents()
        return shared_data

    def _share_agents(self):
        """Create shared data for agents so other classes can create the same
        agents without duplicating the data (i.e. sharing parameters).
        """
        if not hasattr(self, 'agents'):
            return None
        shared_agents = [a.share() for a in self.agents]
        return shared_agents

    def get_agents(self):
        """Return the list of agents."""
        return self.agents

    def get_acts(self):
        """Return the last act of each agent."""
        return self.acts

    def __enter__(self):
        """Empty enter provided for use with ``with`` statement.

        e.g:

        .. code-block:: python

            with World() as world:
                for n in range(10):
                    n.parley()
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """After ``with`` statement, call shutdown."""
        silent_exit = isinstance(exc_value, KeyboardInterrupt)
        self.shutdown()
        return silent_exit

    def num_examples(self):
        return 0

    def num_episodes(self):
        return 0

    def reset(self):
        for a in self.agents:
            a.reset()

    def reset_metrics(self):
        for a in self.agents:
            a.reset_metrics()

    def save_agents(self):
        """Saves all of the agents in the world by calling their respective
        save() methods.
        """
        for a in self.agents:
            a.save()

    def shutdown(self):
        """Perform any cleanup, if appropriate."""
        pass

    def auto_execute(self, use_tqdm=True, maxes=0):
        with self:
            cnt = 0
            for _ in self:
                if maxes:
                    cnt += 1
                    if cnt > maxes:
                        break
                self.parley()


class DialogPartnerWorld(World):
    """Simple world for two agents communicating synchronously.

    This basic world switches back and forth between two agents, giving each
    agent one chance to speak per turn and passing that back to the other one.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            if len(agents) != 2:
                raise RuntimeError('There must be exactly two agents for this '
                                   'world.')
            # Add passed in agents directly.
            self.agents = agents
        self.acts = [None] * len(self.agents)

    def parley(self):
        """Agent 0 goes first. Alternate between the two agents."""
        acts = self.acts
        agents = self.agents
        acts[0] = agents[0].act()
        agents[1].observe(validate(acts[0]))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))

    def episode_done(self):
        """Only the first agent indicates when the episode is done."""
        if self.acts[0] is not None:
            return self.acts[0].get('episode_done', False)
        else:
            return False

    def epoch_done(self):
        """Only the first agent indicates when the epoch is done."""
        return (self.agents[0].epoch_done()
                if hasattr(self.agents[0], 'epoch_done') else False)

    def report(self):
        if hasattr(self.agents[0], 'report'):
            return self.agents[0].report()

    def num_examples(self):
        return self.agents[0].num_examples()

    def num_episodes(self):
        return self.agents[0].num_episodes()

    def shutdown(self):
        """Shutdown each agent."""
        for a in self.agents:
            a.shutdown()


class MultiAgentDialogWorld(World):
    """Basic world where each agent gets a turn in a round-robin fashion,
    receiving as input the actions of all other agents since that agent last
    acted.
    """
    def __init__(self, opt, agents=None, shared=None):
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            # Add passed in agents directly.
            self.agents = agents
            self.acts = [None] * len(agents)
        super().__init__(opt, agents, shared)

    def parley(self):
        """For each agent, get an observation of the last action each of the
        other agents took. Then take an action yourself.
        """
        acts = self.acts
        for index, agent in enumerate(self.agents):
            acts[index] = agent.act()
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))

    def epoch_done(self):
        done = False
        for a in self.agents:
            if a.epoch_done():
                done = True
        return done

    def episode_done(self):
        done = False
        for a in self.agents:
            if a.episode_done():
                done = True
        return done

    def report(self):
        return self.agents[0].report()

    def shutdown(self):
        """Shutdown each agent."""
        for a in self.agents:
            a.shutdown()


class ExecutableWorld(MultiAgentDialogWorld):
    """A world where messages from agents can be interpreted as _actions_ in the
    world which result in changes in the environment (are executed). Hence a grounded
    simulation can be implemented rather than just dialogue between agents.
    """
    def __init__(self, opt, agents=None, shared=None):
        super().__init__(opt, agents, shared)
        self.init_world()

    def init_world(self):
        """An executable world class should implement this function, otherwise
        the actions do not do anything (and it is the same as MultiAgentDialogWorld).
        """
        pass

    def execute(self, agent, act):
        """An executable world class should implement this function, otherwise
        the actions do not do anything (and it is the same as MultiAgentDialogWorld).
        """
        pass

    def observe(self, agent, act):
        """An executable world class should implement this function, otherwise
        the observations for each agent are just the messages from other agents
        and not confitioned on the world at all (and it is thus the same as
        MultiAgentDialogWorld). """
        if agent.id == act['id']:
            return None
        else:
            return act

    def parley(self):
        """For each agent: act, execute and observe actions in world
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


class MultiWorld(World):
    """Container for a set of worlds where each world gets a turn
    in a round-robin fashion. The same user_agents are placed in each,
    though each world may contain additional agents according to the task
    that world represents.
    """

    def __init__(self, opt, agents=None, shared=None):
        opt['batch_sort'] = False
        print('WARNING: batch_sort disabled for multitasking')
        super().__init__(opt)
        self.worlds = []
        for index, k in enumerate(opt['task'].split(',')):
            k = k.strip()
            if k:
                print("[creating world: " + k + "]")
                opt_singletask = copy.deepcopy(opt)
                opt_singletask['task'] = k
                if shared:
                    # Create worlds based on shared data.
                    s = shared['worlds'][index]
                    self.worlds.append(s['world_class'](s['opt'], None, s))
                else:
                    # Agents are already specified.
                    self.worlds.append(create_task_world(opt_singletask, agents))
        self.world_idx = -1
        self.new_world = True
        self.parleys = -1
        self.random = opt.get('datatype', None) == 'train'

    def num_examples(self):
        if not hasattr(self, 'num_exs'):
            worlds_num_exs = [w.num_examples() for w in self.worlds]
            if any(num is None for num in worlds_num_exs):
                self.num_exs = None
            else:
                self.num_exs = sum(worlds_num_exs)
        return self.num_exs

    def num_episodes(self):
        if not hasattr(self, 'num_eps'):
            worlds_num_eps = [w.num_episodes() for w in self.worlds]
            if any(num is None for num in worlds_num_eps):
                self.num_eps = None
            else:
                self.num_eps = sum(worlds_num_eps)
        return self.num_eps

    def get_agents(self):
        return self.worlds[self.world_idx].get_agents()

    def get_acts(self):
        return self.worlds[self.world_idx].get_acts()

    def share(self):
        shared_data = {}
        shared_data['world_class'] = type(self)
        shared_data['opt'] = self.opt
        shared_data['worlds'] = [w.share() for w in self.worlds]
        return shared_data

    def epoch_done(self):
        for t in self.worlds:
            if not t.epoch_done():
                return False
        return True

    def parley_init(self):
        self.parleys = self.parleys + 1
        if self.world_idx >= 0 and self.worlds[self.world_idx].episode_done():
            self.new_world = True
        if self.new_world:
            self.new_world = False
            self.parleys = 0
            if self.random:
                # select random world
                self.world_idx = random.randrange(len(self.worlds))
            else:
                # do at most one full loop looking for unfinished world
                for _ in range(len(self.worlds)):
                    self.world_idx = (self.world_idx + 1) % len(self.worlds)
                    if not self.worlds[self.world_idx].epoch_done():
                        # if this world has examples ready, break
                        break

    def parley(self):
        self.parley_init()
        self.worlds[self.world_idx].parley()

    def display(self):
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
        # TODO: static method in metrics, "aggregate metrics"
        m = {}
        m['tasks'] = {}
        sum_accuracy = 0
        num_tasks = 0
        total = 0
        for i in range(len(self.worlds)):
            wid = self.worlds[i].getID()
            mt = self.worlds[i].report()
            while wid in m['tasks']:
                # prevent name cloberring if using multiple tasks with same ID
                wid += '_'
            m['tasks'][wid] = mt
            total += mt['total']
            if 'accuracy' in mt:
                sum_accuracy += mt['accuracy']
                num_tasks += 1
        if num_tasks > 0:
            m['accuracy'] = sum_accuracy / num_tasks
            m['total'] = total
        return m

    def reset(self):
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        for w in self.worlds:
            w.reset_metrics()

    def save_agents(self):
        # Assumes all worlds have same agents, picks first to save.
        self.worlds[0].save_agents()


def override_opts_in_shared(table, overrides):
    """Looks recursively for ``opt`` dictionaries within shared dict and overrides
    any key-value pairs with pairs from the overrides dict.
    """
    if 'opt' in table:
        # change values if an 'opt' dict is available
        for k, v in overrides.items():
            table['opt'][k] = v
    for k, v in table.items():
        # look for sub-dictionaries which also might contain an 'opt' dict
        if type(v) == dict and k != 'opt':
            override_opts_in_shared(v, overrides)
        elif type(v) == list:
            for item in v[:1]:  # don't actually want to iterate over all items
                if type(item) == dict:
                    override_opts_in_shared(item, overrides)
    return table


class BatchWorld(World):
    """Creates a separate world for each item in the batch, sharing
    the parameters for each.
    The underlying world(s) it is batching can be either
    ``DialogPartnerWorld``, ``MultiAgentWorld``, ``ExecutableWorld`` or
    ``MultiWorld``.
    """

    def __init__(self, opt, world):
        self.opt = opt
        self.random = opt.get('datatype', None) == 'train'
        self.world = world
        shared = world.share()
        self.worlds = []
        for i in range(opt['batchsize']):
            # make sure that any opt dicts in shared have batchindex set to i
            # this lets all shared agents know which batchindex they have,
            # which is needed for ordered data (esp valid/test sets)
            override_opts_in_shared(shared, {'batchindex': i})
            self.worlds.append(shared['world_class'](opt, None, shared))
        self.batch_observations = [None] * len(self.world.get_agents())
        self.first_batch = None

    def batch_observe(self, index, batch_actions, index_acting):
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
                if index == index_acting: return None # don't observe yourself talking
                observation = validate(batch_actions[i])
            observation = agents[index].observe(observation)
            if observation is None:
                raise ValueError('Agents should return what they observed.')
            batch_observations.append(observation)
        return batch_observations

    def batch_act(self, agent_idx, batch_observation):
        # Given batch observation, do update for agents[index].
        # Call update on agent
        a = self.world.get_agents()[agent_idx]
        if hasattr(a, 'batch_act') and not (hasattr(a, 'use_batch_act') and not a.use_batch_act):
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
        # Collect batch together for each agent, and do update.
        # Assumes DialogPartnerWorld, MultiAgentWorld, or MultiWorlds of them.
        num_agents = len(self.world.get_agents())
        batch_observations = self.batch_observations

        for w in self.worlds:
            if hasattr(w, 'parley_init'):
                w.parley_init()

        for agent_idx in range(num_agents):
            # The agent acts.
            batch_act = self.batch_act(agent_idx, batch_observations[agent_idx])
            # We possibly execute this action in the world.
            for i, w in enumerate(self.worlds):
                if hasattr(w, 'execute'):
                    w.execute(w.agents[i], batch_act[i])
            # All agents (might) observe the results.
            for other_index in range(num_agents):
                obs = self.batch_observe(other_index, batch_act, agent_idx)
                if obs is not None:
                    batch_observations[other_index] = obs

    def display(self):
        s = ("[--batchsize " + str(len(self.worlds)) + "--]\n")
        for i, w in enumerate(self.worlds):
            s += ("[batch world " + str(i) + ":]\n")
            s += (w.display() + '\n')
        s += ("[--end of batch--]")
        return s

    def num_examples(self):
        return self.world.num_examples()

    def num_episodes(self):
        return self.world.num_episodes()

    def getID(self):
        return self.world.getID()

    def episode_done(self):
        return False

    def epoch_done(self):
        # first check parent world: if it says it's done, we're done
        if self.world.epoch_done():
            return True
        # otherwise check if all shared worlds are done
        for world in self.worlds:
            if not world.epoch_done():
                return False
        return True

    def report(self):
        return self.world.report()

    def reset(self):
        self.world.reset()
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        self.world.reset_metrics()

    def save_agents(self):
        # Because all worlds share the same parameters through sharing, saving
        # one copy would suffice
        self.world.save_agents()

    def shutdown(self):
        """Shutdown each world."""
        for w in self.worlds:
            w.shutdown()
        self.world.shutdown()


class HogwildProcess(Process):
    """Process child used for ``HogwildWorld``.
    Each ``HogwildProcess`` contain its own unique ``World``.
    """


    def __init__(self, tid, world, opt, agents, sync):
        self.threadId = tid
        self.world_type = world
        self.opt = opt
        self.agent_shares = [a.share() for a in agents]
        self.sync = sync
        super().__init__()

    def run(self):
        """Runs normal parley loop for as many examples as this thread can get
        ahold of via the semaphore ``queued_sem``.
        """
        shared_agents = create_agents_from_shared(self.agent_shares, self.threadId)
        world = self.world_type(self.opt, shared_agents)
        self.sync['threads_sem'].release()
        with world:
            while True:
                if self.sync['term_flag'].value:
                    break  # time to close
                self.sync['queued_sem'].acquire()
                self.sync['threads_sem'].release()
                if not world.epoch_done():
                    # do one example if any available
                    world.parley()
                else:
                    with self.sync['epoch_done_flag'].get_lock():
                        # increment the number of finished threads
                        self.sync['epoch_done_flag'].value = True
                    self.sync['threads_sem'].release()  # send control back to main thread
                    self.sync['queued_sem'].release()  # we didn't process anything
                    self.sync['reset_sem'].acquire()
                    world.reset()


class HogwildWorld(World):
    """Creates a separate world for each thread (process).

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

    def __init__(self, world_class, opt, agents):
        self.inner_world = world_class(opt, agents)
        self.numthreads = opt['numthreads']

        self.sync = {  # syncronization primitives
            # semaphores for counting queued examples
            'queued_sem': Semaphore(0),  # counts num exs to be processed
            'threads_sem': Semaphore(0),  # counts threads
            'reset_sem': Semaphore(0),  # allows threads to reset

            # flags for communicating with threads
            'epoch_done_flag': Value('b', False),  # epoch is done
            'reset_flag': Value('b', False),  # threads should reset
            'term_flag': Value('b', False),  # threads should terminate
        }

        # don't let threads create more threads!
        opt = copy.deepcopy(opt)
        opt['numthreads'] = 1
        self.threads = []
        for i in range(self.numthreads):
            self.threads.append(HogwildProcess(i, world_class, opt, agents,
                                               self.sync))
        for t in self.threads:
            t.start()

        for _ in self.threads:
            self.sync['threads_sem'].acquire()


    def display(self):
        return ''
        self.shutdown()
        raise NotImplementedError('Hogwild does not support displaying in-run'
                                  ' task data. Use `--numthreads 1`.')

    def episode_done(self):
        self.shutdown()
        raise RuntimeError('episode_done() undefined for hogwild')

    def epoch_done(self):
        return self.sync['epoch_done_flag'].value

    def parley(self):
        """Queue one item to be processed."""
        # schedule an example
        self.sync['queued_sem'].release()
        # keep main process from getting too far ahead of the threads
        # this way it can only queue up to numthreads unprocessed examples
        self.sync['threads_sem'].acquire()

    def getID(self):
        return self.inner_world.getID()

    def report(self):
        return self.inner_world.report()

    def save_agents(self):
        self.inner_world.save_agents()

    def reset(self):
        # release reset semaphore
        for _ in self.threads:
            self.sync['reset_sem'].release()

    def reset_metrics(self):
        self.inner_world.reset_metrics()

    def shutdown(self):
        """Set shutdown flag and wake threads up to close themselves."""
        # set shutdown flag
        with self.sync['term_flag'].get_lock():
            self.sync['term_flag'].value = True
        # wake up each thread by queueing fake examples or setting reset flag
        for _ in self.threads:
            self.sync['queued_sem'].release()
            self.sync['reset_sem'].release()
        # wait for threads to close
        for t in self.threads:
            t.join()
        self.inner_world.shutdown()


### Functions for creating tasks/worlds given options.

def _get_task_world(opt):
    sp = opt['task'].strip().split(':')
    if '.' in sp[0]:
        # The case of opt['task'] = 'parlai.tasks.squad.agents:DefaultTeacher'
        # (i.e. specifying your own path directly, assumes DialogPartnerWorld)
        world_class = DialogPartnerWorld
    else:
        task = sp[0].lower()
        if len(sp) > 1:
            sp[1] = sp[1][0].upper() + sp[1][1:]
            world_name = sp[1] + "World"
        else:
            world_name = "DefaultWorld"
        module_name = "parlai.tasks.%s.worlds" % (task)
        try:
            my_module = importlib.import_module(module_name)
            world_class = getattr(my_module, world_name)
        except Exception:
            # Defaults to this if you did not specify a world for your task.
            world_class = DialogPartnerWorld
    task_agents = _create_task_agents(opt)
    return world_class, task_agents


def create_task_world(opt, user_agents):
    world_class, task_agents = _get_task_world(opt)
    return world_class(opt, task_agents + user_agents)

def create_task(opt, user_agents):
    """Creates a world + task_agents (aka a task)
    assuming ``opt['task']="task_dir:teacher_class:options"``
    e.g. ``"babi:Task1k:1"`` or ``"#babi-1k"`` or ``"#QA"``,
    see ``parlai/tasks/tasks.py`` and see ``parlai/tasks/task_list.py``
    for list of tasks.
    """
    if not opt.get('task'):
        raise RuntimeError('No task specified. Please select a task with ' +
                           '--task {task_name}.')
    if type(user_agents) != list:
        user_agents = [user_agents]

    # Convert any hashtag task labels to task directory path names.
    # (e.g. "#QA" to the list of tasks that are QA tasks).
    opt = copy.deepcopy(opt)
    opt['task'] = ids_to_tasks(opt['task'])
    print('[creating task(s): ' + opt['task'] + ']')

    # Single threaded or hogwild task creation (the latter creates multiple threads).
    # Check datatype for train, because we need to do single-threaded for
    # valid and test in order to guarantee exactly one epoch of training.
    # If batchsize > 1, default to BatchWorld, as numthreads can be used for
    # multithreading in batch mode, e.g. multithreaded data loading
    if opt.get('numthreads', 1) == 1 or opt.get('batchsize', 1) > 1:
        if ',' not in opt['task']:
            # Single task
            world = create_task_world(opt, user_agents)
        else:
            # Multitask teacher/agent
            # TODO: remove and replace with multiteachers only?
            world = MultiWorld(opt, user_agents)

        if opt.get('batchsize', 1) > 1:
            return BatchWorld(opt, world)
        else:
            return world
    else:
        # more than one thread requested: do hogwild training
        if ',' not in opt['task']:
            # Single task
            # TODO(ahm): fix metrics for multiteacher hogwild training
            world_class, task_agents = _get_task_world(opt)
            return HogwildWorld(world_class, opt, task_agents + user_agents)
        else:
            # TODO(ahm): fix this
            raise NotImplementedError('hogwild multiworld not supported yet')
