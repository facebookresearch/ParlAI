# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.core.fbdialog import FbDialogTeacher


class DefaultTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        # TODO(jase): add this datasets build -- not sure how to get
        # original data (web form) so maybe our converted version?
        dt = opt['datatype'].split(':')[0]
        opt['datafile'] = ('/mnt/vol/gfsai-east/ai-group/datasets/' +
                           'wikiqacorpus/memnn-multians/WikiQA-' +
                           dt + '.txt')
        super().__init__(opt, shared)
