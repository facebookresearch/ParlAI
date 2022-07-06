#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.scripts.display_data import display_data as display, setup_args
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.message import Message
from parlai.core.metrics import F1Metric, AverageMetric
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.core.worlds import create_task
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE

import unittest
import itertools
import parlai.utils.testing as testing_utils


def product_dict(dictionary):
    keys = dictionary.keys()
    vals = dictionary.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class TestWoW(unittest.TestCase):
    """
    Basic tests on the train_model.py example.
    """

    @unittest.skip
    def test_output(self):
        dts = ['train', 'valid', 'test']
        main_task = 'wizard_of_wikipedia'
        variants = [
            'WizardOfWikipediaTeacher',
            'WizardDialogKnowledgeTeacher',
            'BasicdialogTeacher',
            'DocreaderTeacher',
            'GeneratorTeacher',
        ]
        variant_args = {
            'WizardOfWikipediaTeacher': {},
            'WizardDialogKnowledgeTeacher': {
                'label_type': ['response', 'chosen_sent'],
                'include_knowledge': [False, True],
                'include_checked_sentence': [False, True],
            },
            'BasicdialogTeacher': {'wizard_dialog': [False, True]},
            'DocreaderTeacher': {
                'teacher_type': [
                    'docs',
                    'docs_sentence',
                    'more_docs',
                    'more_docs_sentence',
                    'span',
                ]
            },
            'GeneratorTeacher': {
                'only_checked_knowledge': [False, True],
                'ignorant_dropout': [0, 0.5, 1],
            },
        }
        splits = ['random_split', 'topic_split']

        for datatype in dts:
            for task_var in variants:
                for split in splits:
                    task_name = '{}:{}:{}'.format(main_task, task_var, split)
                    opt_defaults = {'task': task_name, 'datatype': datatype}
                    task_args = variant_args[task_var]
                    if len(task_args) == 0:
                        print('Testing {} with args {}'.format(task_name, opt_defaults))
                        self._run_display_test(opt_defaults)
                    else:
                        for combo in product_dict(task_args):
                            args = {**opt_defaults, **combo}
                            print('Testing {} with args {}'.format(task_name, args))
                            self._run_display_test(args)

    def _run_display_test(self, kwargs):
        with testing_utils.capture_output() as stdout:
            parser = setup_args()
            parser.set_defaults(**kwargs)
            opt = parser.parse_args([])
            agent = RepeatLabelAgent(opt)
            world = create_task(opt, agent)
            display(opt)

        str_output = stdout.getvalue()
        self.assertTrue(
            'loaded {} episodes with a total of {} examples'.format(
                world.num_episodes(), world.num_examples()
            )
            in str_output,
            'Wizard of Wikipedia failed with following args: {}'.format(opt)
            + str_output,
        )

    def test_custom_eval(self):
        """
        Test whether custom evaluation works.
        """
        parser = setup_args()
        opt = parser.parse_args(
            [
                '--task',
                'wizard_of_wikipedia',
                '--datatype',
                'valid',
                '--label-type',
                'chosen_sent',
            ]
        )
        teacher = create_task_agent_from_taskname(opt)[0]

        title = 'Gardening'
        cands = list('four')

        text = "Gardening\nI like Gardening, even when I've only been doing it for a short time."
        response = 'I live on a farm, we garden all year long, it is very relaxing.'
        checked_sent = (
            'Gardening is considered by many people to be a relaxing activity.'
        )
        checked_sent_label = f'{title}{TOKEN_KNOWLEDGE}{checked_sent}'

        retrieval_metric_keys = ['passage_r@1', 'passage_r@5', 'title_r@1', 'title_r@5']

        chosen_sent_teacher_action = Message(
            {
                'text': text,
                'labels': [checked_sent_label],
                'title': [title],
                'checked_sentence': [checked_sent],
            }
        )
        correct_chosen_sent_response = Message(
            {
                'text': checked_sent_label,
                'title_candidates': [title] + cands,
                'text_candidates': [checked_sent_label] + cands,
            }
        )
        top5_chosen_sent_response = Message(
            {
                'text': f'hello{TOKEN_KNOWLEDGE}goodbye',
                'title_candidates': cands + [title],
                'text_candidates': cands + [checked_sent_label],
            }
        )
        incorrect_chosen_sent_response = Message(
            {
                'text': f'hello{TOKEN_KNOWLEDGE}goodbye',
                'title_candidates': cands,
                'text_candidates': cands,
            }
        )

        response_teacher_action = Message(
            {'text': text, 'labels': [response], 'checked_sentence': checked_sent}
        )
        high_f1_response = Message({'text': checked_sent})
        low_f1_response = Message({'text': 'incorrect'})

        # 1) Test with correct top sentence
        teacher.reset_metrics()
        teacher.custom_evaluation(
            chosen_sent_teacher_action,
            [checked_sent_label],
            correct_chosen_sent_response,
        )
        report = teacher.report()
        for k in retrieval_metric_keys:
            assert k in report
            assert report[k] == AverageMetric(1)

        # 2) Test with top sentence in top 5
        teacher.reset_metrics()
        teacher.custom_evaluation(
            chosen_sent_teacher_action, [checked_sent_label], top5_chosen_sent_response
        )
        report = teacher.report()
        for k in retrieval_metric_keys:
            assert k in report
            assert report[k] == AverageMetric(1) if '5' in k else AverageMetric(0)

        # 3) Test with no top sentences
        teacher.reset_metrics()
        teacher.custom_evaluation(
            chosen_sent_teacher_action,
            [checked_sent_label],
            incorrect_chosen_sent_response,
        )
        report = teacher.report()
        for k in retrieval_metric_keys:
            assert k in report
            assert report[k] == AverageMetric(0)

        # 4) Test knowledge f1 with high f1
        teacher.label_type = 'response'
        teacher.reset_metrics()
        teacher.custom_evaluation(response_teacher_action, [response], high_f1_response)
        report = teacher.report()
        assert 'knowledge_f1' in report
        assert report['knowledge_f1'] == F1Metric(1)

        # 5) Test knowledge f1 with low f1
        teacher.reset_metrics()
        teacher.custom_evaluation(response_teacher_action, [response], low_f1_response)
        report = teacher.report()
        assert 'knowledge_f1' in report
        assert report['knowledge_f1'] == F1Metric(0)


if __name__ == '__main__':
    unittest.main()
