#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestDefaultTeacher(AutoTeacherTest):
    task = 'wizard_of_internet'


class TestApprenticeTeacher(AutoTeacherTest):
    task = 'wizard_of_internet:ApprenticeDialogTeacher'


class TestWizardGoldKnowledgeTeacher(AutoTeacherTest):
    task = 'wizard_of_internet:WizardDialogGoldKnowledgeTeacher'


class TestSearchQueryTeacher(AutoTeacherTest):
    task = 'wizard_of_internet:SearchQueryTeacher'


class TestGoldKnowledgeTeacher(AutoTeacherTest):
    task = 'wizard_of_internet:GoldKnowledgeTeacher'


class TestGoldDocsTeacher(AutoTeacherTest):
    task = 'wizard_of_internet:GoldDocsTeacher'


class TestGoldDocTitlesTeacher(AutoTeacherTest):
    task = 'wizard_of_internet:GoldDocTitlesTeacher'
