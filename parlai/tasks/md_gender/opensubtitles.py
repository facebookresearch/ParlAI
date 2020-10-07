#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.md_gender.build import build  # noqa: F401


class OpensubtitlesTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared):
        raise RuntimeError('Data coming soon!')
