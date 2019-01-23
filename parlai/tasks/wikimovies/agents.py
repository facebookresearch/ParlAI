#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDialogTeacher
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
        kbs = {}
        kbs['kb'] = os.path.join('wiki_entities', 'wiki_entities_kb.txt')
        kbs['wiki'] = 'wiki.txt'
        kbs['ie'] = 'wiki_ie.txt'
        opt['datafile'] = os.path.join(opt['datapath'], 'WikiMovies', 'movieqa',
                                       'knowledge_source', kbs[kb])
        super().__init__(opt, shared)


class DefaultTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        opt['cands_datafile'] = os.path.join(opt['datapath'], 'WikiMovies',
                                             'movieqa', 'knowledge_source',
                                             'entities.txt')
        super().__init__(opt, shared)
