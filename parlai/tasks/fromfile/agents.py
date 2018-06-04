# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from parlai.core.utils import str_to_msg

import copy

class FbformatTeacher(FbDialogTeacher):
    """This task simply loads the specified file: useful for quick tests without
    setting up a new task.
    """
    
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath', type=str,
                           help="Data file")

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath'):
            raise RuntimeError('fromfile_datapath not specified')
        opt['datafile'] = opt['fromfile_datapath']
        super().__init__(opt, shared)


class Fbformat2Teacher(FbDialogTeacher):
    """This task simply loads the specified file: useful for quick tests without
    setting up a new task. Used to set up a second task.
    """
    
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath2', type=str,
                           help="Data file")
        
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath2'):
            raise RuntimeError('fromfile_datapath2 not specified')
        opt['datafile'] = opt['fromfile_datapath2']
        super().__init__(opt, shared)
        

        
class ParlaiformatTeacher(FixedDialogTeacher):
    """This module provides access to data in the ParlAI Text Dialog format.


    Subclasses ``DialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "ParlAI text" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The way the data is set up is as follows:

    ::

        text:Sam went to the kitchen.
        text:Pat gave Sam the milk.
        text:Where is the milk?<TAB>labels:kitchen<TAB>reward:1<TAB>label_candidates:hallway|kitchen|bathroom
        text:Sam went to the hallway.
        text:Pat went to the bathroom.
        text:Where is the milk?<TAB>labels:hallway<TAB>reward:1<TAB>label_candidateshallway|kitchen|bathroom<TAB>episode_done:True

    Lines 1-6 represent a single episode, with two different examples: the
    first example is lines 1-3, and the second is lines 4-6.

    Lines 1,2,4, and 5 represent contextual information.

    Lines 3 and 6 contain a query, a label, a reward for getting the question
    correct, and three label candidates.

    Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second
    example and therefore the agent must remember the first example in order to
    do well.

    In general dialog in this format can be any speech, not just QA pairs:

    ::

        text:Hi how's it going?<TAB>labels:It's going great. What's new?
        text:Well I'm working on a new project at work.<TAB>labels:Oh me too!
        text:Oh cool!<TAB>labels:Tell me about yours.

    etc.

    Note that dialogs are interpreted as being one-way. For example, consider
    this dialog:

    ::

        1 X1    Y1
        2 X2    Y2
        3 X3    Y3

    A set of examples X1 => Y1, X2 => Y2, and X3 => Y3 will be generated.
    However, Y1 => X2 and Y2 => X3 are not created as separate examples by
    default. This makes sense for some data (we don't need to train on the idea
    that "kitchen" should be followed by "Sam went to the hallway..." above),
    but for other datasets it may be helpful to add additional examples in the
    reverse direction ("Oh cool!" is a response to "Oh me too!" above).
    """
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath', type=str,
                           help="Data file")

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath'):
            raise RuntimeError('fromfile_datapath not specified')
        datafile = opt['fromfile_datapath']
        if shared is None:
            self._setup_data(datafile)
        self.id = datafile
        self.reset()
        
    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        return self.examples[episode_idx]

    def _setup_data(self, path):
        print("[loading parlAI text data:" + path + "]")
        self.examples = []
        with open(path) as read:
            for line in read:
                msg = str_to_msg(line.rstrip('\n'))
                if msg:
                    self.examples.append(msg)

class Parlaiformat2Teacher(FixedDialogTeacher):
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath2', type=str,
                           help="Data file")

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath2'):
            raise RuntimeError('fromfile_datapath2 not specified')
        datafile = opt['fromfile_datapath2']
        if shared is None:
            self._setup_data(datafile)
        self.id = datafile
        self.reset()
        
    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        return self.examples[episode_idx]

    def _setup_data(self, path):
        print("[loading parlAI text data:" + path + "]")
        self.examples = []
        with open(path) as read:
            for line in read:
                msg = str_to_msg(line.rstrip('\n'))
                if msg:
                    self.examples.append(msg)
                    
class DefaultTeacher(FbformatTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
