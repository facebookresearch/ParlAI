#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class DefaultTeacherTest(AutoTeacherTest):
    task = "wizard_of_wikipedia"


class TestWizardOfWikipediaTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:wizard_of_wikipedia"


class TestWizardDialogKnowledgeTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:wizard_dialog_knowledge"


class TestBasicdialogTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:basicdialog"


class TestBasicWizardDialogTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:basic_wizard_dialog"


class TestBasicApprenticeDialogTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:basic_apprentice_dialog"


class TestBasicBothDialogTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:basic_both_dialog"


class TestGeneratorTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:generator"


class TestWikiPageTitleTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:wiki_page_title"


class TestDocreaderTeacher(AutoTeacherTest):
    task = "wizard_of_wikipedia:docreader"
