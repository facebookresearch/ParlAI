# Copyright 2004-present Facebook. All Rights Reserved.
"""This class defines the basic environments that define how agents interact
with one another.

World(object) provides a generic parent class, including __enter__ and __exit__
    statements which allow you to guarantee that the shutdown method is called
    and KeyboardInterrupts are less noisy (if desired).

DialogPartnerWorld(World) provides a two-agent turn-based dialog setting
MultiAgentDialogWorld(World) provides a two-plus turn-based round-robin dialog

HogwildWorld(World) creates another world within itself for every thread, in
    order to have separate simulated environments for each one. Each world gets
    its own agents initialized using the "share()" parameters from the original
    agents.
"""

import copy
import importlib
import random

from multiprocessing import Process, Value, Condition, Semaphore
from collections import deque
from parlai.core.agents import create_task_agents

def validate(observation):
    """Make sure the observation table is valid, or raise an error."""
    if observation is not None and type(observation) == dict:
        return observation
    else:
        raise RuntimeError('Must return dictionary from act().')


class World(object):
    """Empty parent providing null definitions of API functions for Worlds.
    All children can override these to provide more detailed functionality."""

    def __init__(self, opt):
        self.is_done = False

    def __enter__(self):
        """Empty enter provided for use with `with` statement.
        e.g:
        with World() as world:
            for n in range(10):
                n.parley()
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """After `with` statement, call shutdown."""
        silent_exit = isinstance(exc_value, KeyboardInterrupt)
        self.shutdown()
        return silent_exit

    def __iter__(self):
        raise NotImplementedError('Subclass did not implement this.')

    def __len__(self):
        return 0

    def parley(self):
        pass

    def display(self):
        """Returns a string describing the current state of the world.
        Useful for monitoring and debugging. """
        return ''

    def done(self):
        """Whether the episode is done or not. """
        return self.is_done

    def shutdown(self):
        """Performs any cleanup, if appropriate."""
        pass

    def synchronize(self):
        """Can be used to synchronize processes."""
        pass


def create_task_world(opt, user_agents):
    sp = opt['task'].strip().split(':')
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
    except:
        # Defaults to this if you did not specify a world for your task.
        world_class = DialogPartnerWorld
    task_agents = create_task_agents(opt)
    if opt.get('numthreads', 1) == 1:
        world = world_class(opt, task_agents + user_agents)
    else:
        world = HogwildWorld(opt, world_class, task_agents + user_agents)
    return world


def create_task(opt, user_agents):
    """Creates a world + task_agents (aka a task)
    assuming opt['task']="task_dir:teacher_class:options"
    e.g. "babi:Task1k:1"
    """
    if type(user_agents) != list:
        user_agents = [user_agents]

    if not ',' in opt['task']:
        # Single task
        world = create_task_world(opt, user_agents)
        return world
    else:
        # Multitask teacher/agent
        worlds = MultiWorld(opt, user_agents)
        return worlds


class DialogPartnerWorld(World):
    """This basic world switches back and forth between two agents, giving each
    agent one chance to speak per turn and passing that back to the other agent.
    """

    def __init__(self, opt, agents):
        if len(agents) != 2:
            raise RuntimeError('There must be exactly two agents for this ' +
                               'world.')
        self.teacher = agents[0]
        self.agent = agents[1]
        self.reply = {}

    def __iter__(self):
        return iter(self.teacher)

    def parley(self):
        """Teacher goes first. Alternate between the teacher and the agent."""
        self.query = validate(self.teacher.act(self.reply))
        self.reply = validate(self.agent.act(self.query))
        self.is_done = self.query['done']

    def report(self):
        return self.teacher.report()

    def display(self):
        lines = []
        if self.query.get('reward', None) is not None:
            lines.append('   [reward: {r}]'.format(r=self.query['reward']))
        if self.query.get('text', ''):
            lines.append(self.query['text'])
        if self.query.get('candidates', False):
            cand_len = len(self.query['candidates'])
            if cand_len <= 10:
                lines.append('[cands: {}{}]'.format(
                    '|'.join(self.query['candidates'])))
            else:
                # select five candidates from the candidate set, can't slice in
                # because it's a set
                cand_iter = iter(self.query['candidates'])
                display_cands = (next(cand_iter) for _ in range(5))
                # print those cands plus how many cands remain
                lines.append('[cands: {}{}]'.format(
                    '|'.join(display_cands),
                    '| ...and {} more'.format(cand_len - 5)
                ))
        if self.reply.get('text', ''):
            lines.append('   A: ' + self.reply['text'])
        if self.done():
            lines.append('- - - - - - - - - - - - - - - - - - - - -')
        return '\n'.join(lines)

    def __len__(self):
        return len(self.teacher)

    def shutdown(self):
        """Shutdown each agent."""
        self.teacher.shutdown()
        self.agent.shutdown()


class MultiWorld(World):
    """Container for a set of worlds where each world gets a turn
    in a round-robin fashion. The same user_agents are placed in each,
    though each world may contain additional agents according to the task
    that world represents.
    """

    def __init__(self, opt, user_agents):
        self.worlds = []
        tasks = opt['task'].split(',')
        for k in tasks:
            print("[creating world: " + k + "]")
            opt_singletask = copy.deepcopy(opt)
            opt_singletask['task'] = k
            self.worlds.append(
                create_task(opt_singletask, user_agents))
        self.world_idx = -1
        self.new_world = True
        self.parleys = 0
        self.random = opt.get('datatype') == 'train'
        super().__init__(opt)

    def __len__(self):
        if not hasattr(self, 'len'):
            self.len = 0
            # length is sum of all world lengths
            for _ind, t in enumerate(self.worlds):
                self.len += len(t)
        return self.len

    def parley(self):
        if self.new_world:
            self.new_world = False
            self.parleys = 0
            if self.random:
                self.world_idx = random.randrange(len(self.worlds))
            else:
                self.world_idx = (self.world_idx + 1) % len(self.worlds)
        t = self.worlds[self.world_idx]
        t.parley()
        self.parleys = self.parleys + 1
        if t.done():
            self.new_world = True

    def display(self):
        if self.world_idx != -1:
            s = ''
            if self.parleys == 1:
                s = '[world ' + str(self.world_idx) + ']\n'
            s = s + self.worlds[self.world_idx].display()
            return s
        else:
            return ''

class MultiAgentDialogWorld(World):
    """Basic world where each agent gets a turn in a round-robin fashion,
    recieving as input the actions of all other agents since that agent last
    acted.
    """

    def __init__(self, opt, agents):
        # list of actions that each other agent previously took
        self.observations = deque()
        for _ in range(len(agents) - 1):
            # pad with empty observations to start
            self.observations.append({})
        self.agents = agents

    def parley(self):
        """For each agent, get an observation of the last action each of the
        other agents took. Then take an action yourself.
        """
        for agent in self.agents:
            obs = self.observations
            act = agent.act(obs)
            self.step(agent, act)

    def observe(self):
        """Default behavior: concatenate text and rewards from all other agents,
        but use the labels and candidates from only the most recent.
        """
        t = {}
        t['text'] = '\n'.join(obs['text'] for obs in self.observations)
        t['labels'] = self.observations[-1]['labels']
        t['reward'] = '\n'.join(obs['reward'] for obs in self.observations)
        t['candidates'] = self.observations[-1]['candidates']

    def step(self, agent, action):
        """Pop the oldest observation and append this one."""
        self.observations.popleft()
        self.observations.append(action)

    def shutdown(self):
        for a in self.agents:
            a.shutdown()


class HogwildProcess(Process):
    """Process child used for HogwildWorld.
    Each HogwildProcess contain its own unique World
    """

    def __init__(self, tid, world, opt, agents, sem, fin, term, cnt):
        self.threadId = tid
        self.world_type = world
        self.opt = opt
        self.agent_types = [type(a) for a in agents]
        self.agent_shares = [a.share(self.opt) for a in agents]
        self.queued_items = sem
        self.finished = fin
        self.terminate = term
        self.cnt = cnt
        super().__init__()

    def run(self):
        """Runs normal parley loop for as many examples as this thread can get
        ahold of via the semaphore queued_items.
        """
        shared_agents = []
        # reinitialize world/agents using the shared data within this Process
        for i, agent_type in enumerate(self.agent_types):
            opt, shared = self.agent_shares[i]
            agent = agent_type(opt, shared)
            shared_agents.append(agent)
        world = self.world_type(self.opt, shared_agents)

        with world:
            while True:
                self.queued_items.acquire()
                if self.terminate.value:
                    break  # time to close
                world.parley()
                with self.cnt.get_lock():
                    self.cnt.value -= 1
                    if self.cnt.value == 0:
                        # let main thread know that all the examples are done
                        with self.finished:
                            self.finished.notify_all()


class HogwildWorld(World):
    """Creates a separate world for each thread.
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

    def __init__(self, opt, world_class, agents):
        self.inner_world = world_class(opt, agents)

        self.queued_items = Semaphore(0)  # counts num exs to be processed
        self.finished = Condition()  # notifies when exs are done
        self.terminate = Value('b', False)  # tells threads when to shut down
        self.cnt = Value('i', 0)  # number of exs that remain to be processed

        self.threads = []
        for i in range(opt['numthreads']):
            self.threads.append(HogwildProcess(i, world_class, opt,
                                               agents, self.queued_items,
                                               self.finished, self.terminate,
                                               self.cnt))
        for t in self.threads:
            t.start()

    def display(self):
        self.shutdown()
        raise NotImplementedError('Hogwild does not support displaying in-run' +
                                  ' task data. Use `--numthreads 1`.')

    def done(self):
        return False

    def parley(self):
        """Queue one item to be processed."""
        with self.cnt.get_lock():
            self.cnt.value += 1
        self.queued_items.release()

    def report(self):
        return self.inner_world.report()

    def synchronize(self):
        """Sync barrier: will wait until all queued examples are processed."""
        with self.finished:
            self.finished.wait_for(lambda: self.cnt.value == 0)

    def shutdown(self):
        """Set shutdown flag and wake threads up to close themselves"""
        # set shutdown flag
        with self.terminate.get_lock():
            self.terminate.value = True
        # wake up each thread by queueing fake examples
        for _ in self.threads:
            self.queued_items.release()
        # wait for threads to close
        for t in self.threads:
            t.join()
