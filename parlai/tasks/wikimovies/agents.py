# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build

import copy
import os


def _path(opt):
    build(opt)
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'train'
    elif dt == 'test':
        suffix = 'test'
    elif dt == 'valid':
        suffix = 'dev'
    return os.path.join(opt['datapath'], 'WikiMovies', 'movieqa', 'questions',
                        'wiki_entities',
                        'wiki-entities_qa_{suffix}.txt'.format(suffix=suffix))


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        task = opt.get('task')
        if not task:
            task = 'wikimovies:KB:kb'
        kb = task.split(':')
        if len(kb) == 3:
            kb = kb[2]
        elif len(kb) == 2:
            # default to 'kb' if 'kb', 'wiki', or 'ie' not specified
            kb = 'kb'
        self.kb = kb
        kbs = {}
        kbs['kb'] = os.path.join('wiki_entities', 'wiki_entities_kb.txt')
        kbs['wiki'] = 'wiki.txt'
        kbs['ie'] = 'wiki_ie.txt'
        opt['datafile'] = os.path.join(opt['datapath'], 'WikiMovies', 'movieqa',
                                       'knowledge_source', kbs[kb])

        super().__init__(opt, shared)

    def setup_data(self, path):
        """See FbDialogTeacher.setup_data for output format"""
        if self.kb == 'kb':
            return self.setup_kb_data(path)
        else:
            return super().setup_data(path)

    def setup_kb_data(self, path):
        print("[loading KBTeacher data:" + path + "]")
        with open(path) as read:
            for line in read:
                line = line.strip()
                if not line:
                    continue
                space_idx = line.find(' ')
                conv_id = line[:space_idx]
                # conversation index -- '1' means start of episode
                yield (line[space_idx+1:], None, None, None), conv_id == '1'


class DefaultTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        opt['cands_datafile'] = os.path.join(opt['datapath'], 'WikiMovies',
                                             'movieqa', 'knowledge_source',
                                             'entities.txt')
        super().__init__(opt, shared)
