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
from parlai.core.agents import _create_task_agents
from parlai.tasks.tasks import ids_to_tasks

def validate(observation):
    """Make sure the observation table is valid, or raise an error."""
    if observation is not None and type(observation) == dict:
        if ('text_candidates' in observation and
            'text' in observation and
            observation['text'] != observation['text_candidates'][0]):
            raise RuntimeError('If text and text_candidates fields are both ' +
                               'filled, top text candidate should be the same' +
                               ' as text.')
        return observation
    else:
        raise RuntimeError('Must return dictionary from act().')


class World(object):
    """Empty parent providing null definitions of API functions for Worlds.
    All children can override these to provide more detailed functionality."""

    def __init__(self, opt):
        self.is_done = False
        self.id = opt['task']

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

    def getID(self):
        """Return the name of the world, typically the task the world encodes."""
        return self.id

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


def _get_task_world(opt):
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
    task_agents = _create_task_agents(opt)
    return world_class, task_agents


def create_task_world(opt, user_agents):
    world_class, task_agents = _get_task_world(opt)
    return world_class(opt, task_agents + user_agents)

def create_task(opt, user_agents):
    """Creates a world + task_agents (aka a task)
    assuming opt['task']="task_dir:teacher_class:options"
    e.g. "babi:Task1k:1" or "#babi-1k" or "#QA",
    see parlai/tasks/tasks.py and see parlai/tasks/tasks.json
    for list of tasks.
    """
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
    if opt.get('numthreads', 1) == 1 or opt['datatype'] != 'train':
        if ',' not in opt['task']:
            # Single task
            world = create_task_world(opt, user_agents)
            return world
        else:
            # Multitask teacher/agent
            worlds = MultiWorld(opt, user_agents)
            return worlds
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


class DialogPartnerWorld(World):
    """This basic world switches back and forth between two agents, giving each
    agent one chance to speak per turn and passing that back to the other agent.
    """

    def __init__(self, opt, agents):
        if len(agents) != 2:
            raise RuntimeError('There must be exactly two agents for this ' +
                               'world.')
        super().__init__(opt)
        self.teacher = agents[0]
        self.agent = agents[1]
        self.reply = {}

    def __iter__(self):
        return iter(self.teacher)

    def finished(self):
        return self.teacher.finished()

    def parley(self):
        """Teacher goes first. Alternate between the teacher and the agent."""
        self.teacher.observe(validate(self.reply))
        self.query = self.teacher.act()
        self.agent.observe(validate(self.query))
        self.reply = self.agent.act()
        self.is_done = self.query['done']

    def report(self):
        return self.teacher.report()

    def display(self):
        lines = []
        if self.query.get('reward', None) is not None:
            lines.append('   [reward: {r}]'.format(r=self.query['reward']))
        if self.query.get('text', ''):
            id = '[' + self.query['id'] + ']: ' if 'id' in self.query else ''
            lines.append(id + self.query['text'])
        if self.query.get('label_candidates', False):
            cand_len = len(self.query['label_candidates'])
            if cand_len <= 10:
                lines.append('[cands: {}]'.format(
                    '|'.join(self.query['label_candidates'])))
            else:
                # select five label_candidates from the candidate set, can't slice in
                # because it's a set
                cand_iter = iter(self.query['label_candidates'])
                display_cands = (next(cand_iter) for _ in range(5))
                # print those cands plus how many cands remain
                lines.append('[cands: {}{}]'.format(
                    '|'.join(display_cands),
                    '| ...and {} more'.format(cand_len - 5)
                ))
        if self.reply.get('text', ''):
            id = '[' + self.reply['id'] + ']: ' if 'id' in self.reply else ''
            lines.append('   ' + id + self.reply['text'])
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
        super().__init__(opt)
        self.worlds = []
        for k in opt['task'].split(','):
            k = k.strip()
            if k:
                print("[creating world: " + k + "]")
                opt_singletask = copy.deepcopy(opt)
                opt_singletask['task'] = k
                self.worlds.append(create_task_world(opt_singletask, user_agents))
        self.world_idx = -1
        self.new_world = True
        self.parleys = 0
        self.random = opt.get('datatype') == 'train'

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished():
            raise StopIteration()

    def __len__(self):
        if not hasattr(self, 'len'):
            self.len = 0
            # length is sum of all world lengths
            for _ind, t in enumerate(self.worlds):
                self.len += len(t)
        return self.len

    def finished(self):
        for t in self.worlds:
            if not t.finished():
                return False
        return True

    def parley(self):
        if self.new_world:
            self.new_world = False
            self.parleys = 0
            if self.random:
                self.world_idx = random.randrange(len(self.worlds))
            else:
                start_idx = self.world_idx
                keep_looking = True
                while keep_looking:
                    self.world_idx = (self.world_idx + 1) % len(self.worlds)
                    keep_looking = (self.worlds[self.world_idx].finished() and
                                    start_idx != self.world_idx)
                if start_idx == self.world_idx:
                    return {'text': 'There are no more examples remaining.'}
        t = self.worlds[self.world_idx]
        t.parley()
        self.parleys = self.parleys + 1
        if t.done():
            self.new_world = True

    def display(self):
        if self.world_idx != -1:
            s = ''
            w = self.worlds[self.world_idx]
            if self.parleys == 1:
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
            mt = self.worlds[i].report()
            m['tasks'][self.worlds[i].getID()] = mt
            total += mt['total']
            if 'accuracy' in mt:
                sum_accuracy += mt['accuracy']
                num_tasks += 1
        if num_tasks > 0:
            m['accuracy'] = sum_accuracy / num_tasks
            m['total'] = total
        return m

class MultiAgentDialogWorld(World):
    """Basic world where each agent gets a turn in a round-robin fashion,
    recieving as input the actions of all other agents since that agent last
    acted.
    """

    def __init__(self, opt, agents):
        id = self.opt['task']
        self.agents = agents

    def parley(self):
        """For each agent, get an observation of the last action each of the
        other agents took. Then take an action yourself.
        """
        for agent in self.agents:
            act = agent.act()
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(act)

    def shutdown(self):
        for a in self.agents:
            a.shutdown()


class HogwildProcess(Process):
    """Process child used for HogwildWorld.
    Each HogwildProcess contain its own unique World.
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

    def __iter__(self):
        raise NotImplementedError('Iteration not available in hogwild.')

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

    def getID(self):
        return self.inner_world.getID()

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
