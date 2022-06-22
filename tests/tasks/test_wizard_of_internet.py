#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from parlai.scripts.display_data import setup_args
from parlai.utils.testing import display_data
import parlai.tasks.wizard_of_internet.constants as CONST
from parlai.core.message import Message
from parlai.core.teachers import create_task_agent_from_taskname


class TestApprenticeDialogTeacher(unittest.TestCase):
    def test_display_data(self):
        parser = setup_args()
        opt = parser.parse_args(
            [
                '--task',
                'wizard_of_internet:ApprenticeDialogTeacher',
                '--num-examples',
                '100000',
            ]
        )
        display_data(opt)


class TestWizardDialogTeacher(unittest.TestCase):
    def test_display_data(self):
        parser = setup_args()
        opt = parser.parse_args(['--task', 'wizard_of_internet'])
        display_data(opt)

    def test_display_data_with_prepend_gold(self):
        parser = setup_args()
        opt = parser.parse_args(
            ['--task', 'wizard_of_internet:WizardDialogGoldKnowledgeTeacher']
        )
        for out_type in display_data(opt):
            started_knowledge_span = False
            for token in [w.strip() for w in out_type.split() if w.strip()]:
                if token == CONST.KNOWLEDGE_TOKEN:
                    self.assertFalse(started_knowledge_span)
                    started_knowledge_span = True
                elif token == CONST.END_KNOWLEDGE_TOKEN:
                    self.assertTrue(started_knowledge_span)
                    started_knowledge_span = False

            self.assertFalse(started_knowledge_span)


class TestWizardDialogTeacherCustomEval(unittest.TestCase):
    def test_custom_eval(self):
        opt = setup_args().parse_args(
            ['--task', 'wizard_of_internet', '--datatype', 'valid']
        )
        teacher = create_task_agent_from_taskname(opt)[0]
        teacher_action_message = Message(teacher.get(0))
        teacher_action_nonmessage = {k: v for k, v in teacher_action_message.items()}
        labels = teacher_action_message.get(
            'labels', teacher_action_message.get('eval_labels')
        )
        agent_nonmessage = {'text': labels[0], 'episode_done': False}
        agent_message = Message(agent_nonmessage)
        for t_act in [teacher_action_nonmessage, teacher_action_message]:
            for m_act in [agent_nonmessage, agent_message]:
                teacher.custom_evaluation(t_act, labels, m_act)


class TestSearchQueryTeacher(unittest.TestCase):
    def test_display_data(self):
        parser = setup_args()
        opt = parser.parse_args(
            [
                '--task',
                'wizard_of_internet:QueryTeacher',
                '--dialog-history',
                'onlylast',
                '--include-persona',
                'true',
                '--num-examples',
                '100000',
            ]
        )
        display_data(opt)


class TestKnowledgeTeachers(unittest.TestCase):
    def test_gold_knowledge_teacher(self):
        parser = setup_args()
        opt = parser.parse_args(
            [
                '--task',
                'wizard_of_internet:GoldKnowledgeTeacher',
                '--dialog-history',
                'onlylast',
                '--include-persona',
                'true',
                '--num-examples',
                '100000',
            ]
        )
        display_data(opt)

    def test_gold_docs_teacher(self):
        parser = setup_args()
        opt = parser.parse_args(
            [
                '--task',
                'wizard_of_internet:GoldDocsTeacher',
                '--dialog-history',
                'full',
                '--include-persona',
                'false',
                '--num-examples',
                '100000',
            ]
        )
        display_data(opt)

    def test_gold_doc_titles_teacher(self):
        parser = setup_args()
        opt = parser.parse_args(
            [
                '--task',
                'wizard_of_internet:GoldDocTitlesTeacher',
                '--dialog-history',
                'full',
                '--include-persona',
                'false',
                '--num-examples',
                '100000',
            ]
        )
        display_data(opt)


if __name__ == '__main__':
    unittest.main()
