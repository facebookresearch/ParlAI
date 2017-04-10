# Copyright 2004-present Facebook. All Rights Reserved.

import copy

from parlai.core.fbdialog import FbDialogTeacher
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
    return (opt['datapath'] + 'WikiMovies/' +
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
        opt['datafile'] = (opt['datapath'] + 'WikiMovies/movieqa/' +
                           'knowledge_source/' + kbs[kb])
        super().__init__(opt, shared)


class DefaultTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        super().__init__(opt, shared)
