#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from .agents import Teacher
from .data import TextData, HogwildTextData
from .thread_utils import SharedTable
from .metrics import Metrics
import time


class DialogTeacher(Teacher):
    """This class provides a set a basic functionality:
    - metrics tracking count of sent vs correctly answered queries
    - uses data class to store and query text data
    - generates action tables to send to the student agent from the data

    If you have opt.numthreads > 1, this also activates a shared memory
    array for the data and lock-protected shared-memory metrics.

    In order to subclass this class, you must implement setup_data() in your
    class (or subclass another class which does, like FbDialogTeacher), which
    reads your data file as an iterator. See the data module for a description
    of the requirements for setup_data().
    """

    def __init__(self, opt, shared=None):
        # Check for setup_data
        print("[DialogTeacher initializing.]")
        if not hasattr(self, 'setup_data'):
            raise RuntimeError('Must implement setup_data or subclass a class' +
                               ' which implements it (e.g. FbDialogTeacher)' +
                               ' in order to use this class.')

        self.datatype = opt['datatype']
        self.startTime = time.time()
        self.lastY = None
        self.lastR = None
        self.lastDone = False
        self.defaultPosReward = 1
        self.defaultNegReward = 0
        self.metrics = Metrics(opt)

        # Dynamically allocate which child class to use based on whether you
        # are using hogwild or not by overwriting share, act, and report methods
        # TODO(ahm): find a way to do this more clearly
        child = (_RegularDialogTeacher if opt.get('numthreads', 1) == 1 else
                 _HogwildDialogTeacher)
        child.__init__(self, opt, shared)
        self.share = lambda x: child.share(self, x)
        self.act = lambda x: child.act(self, x)
        self.report = lambda: child.report(self)

    def __len__(self):
        return len(self.data)

    def candidates(self):
        return None


class _RegularDialogTeacher(Teacher):

    def __init__(self, opt, shared=None):
        self.data = TextData(self.setup_data(opt['datafile']),
                             cands=self.candidates(),
                             random=self.datatype == 'train')

    # share datatype, data, metrics, and a lock on the metrics
    def share(self, opt):
        raise RuntimeError('no sharing: use HogwildFbDialogTeacher instead')

    # Check received text for correct answer then send new query.
    def act(self, observation):
        reward = None
        # First process observation for metrics and rewards.
        if self.lastY is not None:
            loss = self.metrics.update(observation.get('text', ''), self.lastY)
            if loss['correct']:
                # update reward
                if self.lastR is not None:
                    reward = self.lastR
                else:
                    reward = self.defaultPosReward
            else:
                reward = self.defaultNegReward
            self.lastY = None
            self.lastR = None
        done = self.lastDone
        self.lastDone = False

        # Then build reply.
        if not done:
            action = next(self.data)
            self.lastY = action.get('labels', None)
            self.lastR = action.pop('reward', None)
            self.lastDone = action.get('done', None)
            action['done'] = False
            if not self.datatype.startswith('train'):
                action.pop('labels', None)
        else:
            # Very last action gives final reward, and sends 'done' signal.
            action = {}
            action['done'] = True
        if reward is not None:
            action['reward'] = reward
        return action

    # Return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        return self.metrics.report()


class _HogwildDialogTeacher(Teacher):

    def __init__(self, opt, shared=None):
        # first initialize any shared objects
        if shared and shared.get('data'):
            self.data = shared['data']
        else:
            self.data = HogwildTextData(self.setup_data(opt['datafile']),
                                        cands=self.candidates(),
                                        random=self.datatype == 'train')

        if shared and shared.get('metrics'):
            self.metrics = shared['metrics']
        else:
            self.metrics = SharedTable({
                'cnt': 0,
                'correct': 0,
            })

    # share datatype, data, metrics, and a lock on the metrics
    def share(self, opt):
        if not hasattr(self, 'shared'):
            self.shared = {}
            self.shared['data'] = self.data
            self.shared['metrics'] = self.metrics
        opt['datatype'] = self.datatype
        return (opt, self.shared)

    # check received observation for correct answer then send new query
    def act(self, observation):
        # first process observation
        if (self.lastY is not None and observation.get('text')):
            if _check_answer(observation['text'], self.lastY):
                with self.metrics.get_lock():
                    self.metrics['correct'] += 1
            self.lastY = None

        with self.metrics.get_lock():
            self.metrics['cnt'] += 1

        # then send new reply
        action = next(self.data)
        self.lastY = action.get('labels', None)
        if not self.datatype.startswith('train'):
            if 'labels' in action:
                action.pop('labels', None)

        return action

    # return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        m = {}
        m['total'] = self.metrics['cnt']
        if self.metrics['correct'] > 0 or not self.datatype.startswith('train'):
            m['accuracy'] = self.metrics['correct'] / self.metrics['cnt']

        with self.metrics.get_lock():
            self.metrics['cnt'] = 0
            self.metrics['correct'] = 0
        return m
