# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import copy

from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build


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
    return (opt['datapath'] + '/WikiMovies/' +
            'movieqa/questions/wiki_entities/' +
            'wiki-entities_qa_{suffix}.txt'.format(
                suffix=suffix))


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        task = opt.get('task', 'wikimovies:KB:kb')
        kb = task.split(':')[2]
        kbs = {}
        kbs['kb'] = 'wiki_entities/wiki_entities_kb.txt'
        kbs['wiki'] = 'wiki.txt'
        kbs['ie'] = 'wiki_ie.txt'
        opt['datafile'] = (opt['datapath'] + '/WikiMovies/movieqa/' +
                           'knowledge_source/' + kbs[kb])
        super().__init__(opt, shared)


class DefaultTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        opt['cands_datafile'] = (opt['datapath'] +
                                 '/WikiMovies/movieqa/' +
                                 'knowledge_source/entities.txt')
        super().__init__(opt, shared)
