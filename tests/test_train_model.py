#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic tests that ensure train_model.py behaves in predictable ways.
"""
import os
import unittest
import json
import parlai.utils.logging as logging
import parlai.utils.testing as testing_utils
from parlai.utils.io import PathManager
from parlai.core.metrics import AverageMetric
from parlai.core.worlds import create_task
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.agents import register_agent, Agent
from parlai.scripts.eval_model import get_task_world_logs


class TestTrainModel(unittest.TestCase):
    def test_final_extra_eval_and_save_json(self):
        """
        Test "final_extra_valid_opt_filepath". Happens to test that saving reports as
        json works too.

        We copy train_model from testing_utils to directly access train loop.
        """
        import parlai.scripts.train_model as tms

        def get_tl(tmpdir):
            final_opt = Opt(
                {
                    'task': 'integration_tests',
                    'datatype': 'valid',
                    'validation_max_exs': 30,
                    'short_final_eval': True,
                }
            )
            final_opt.save(os.path.join(tmpdir, "final_opt.opt"))

            opt = Opt(
                {
                    'task': 'integration_tests',
                    'validation_max_exs': 10,
                    'model': 'repeat_label',
                    'model_file': os.path.join(tmpdir, 'model'),
                    'short_final_eval': True,
                    'num_epochs': 1.0,
                    'final_extra_opt': str(os.path.join(tmpdir, "final_opt.opt")),
                }
            )
            parser = tms.setup_args()
            parser.set_params(**opt)
            popt = parser.parse_args([])
            for k, v in opt.items():
                popt[k] = v
            return tms.TrainLoop(popt)

        with testing_utils.capture_output(), testing_utils.tempdir() as tmpdir:
            tl = get_tl(tmpdir)
            _, _ = tl.train()

            with open(os.path.join(tmpdir, 'model.trainstats')) as f:
                data = json.load(f)
                print(data)
                self.assertEqual(
                    data["final_valid_report"]["exs"],
                    10,
                    "Validation exs saved incorrectly",
                )

                self.assertEqual(
                    data["final_extra_valid_report"]["exs"],
                    30,
                    "Final validation exs saved incorrectly",
                )

    def test_fast_final_eval(self):
        valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests',
                'validation_max_exs': 10,
                'model': 'repeat_label',
                'short_final_eval': True,
                'num_epochs': 1.0,
            }
        )
        self.assertEqual(valid['exs'], 10, 'Validation exs is wrong')
        self.assertEqual(test['exs'], 10, 'Test exs is wrong')

    def test_multitasking_metrics_micro(self):
        valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': True,
            }
        )

        task1_acc = valid['integration_tests:candidate/accuracy']
        task2_acc = valid['integration_tests:multiturnCandidate/accuracy']
        total_acc = valid['accuracy']
        self.assertEqual(
            total_acc, task1_acc + task2_acc, 'Task accuracy is averaged incorrectly'
        )

        valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': True,
            }
        )
        task1_acc = valid['integration_tests:candidate/accuracy']
        task2_acc = valid['integration_tests:multiturnCandidate/accuracy']
        total_acc = valid['accuracy']
        # metrics should be averaged equally across tasks
        self.assertEqual(
            total_acc, task1_acc + task2_acc, 'Task accuracy is averaged incorrectly'
        )

    def test_multitasking_metrics_macro(self):
        valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': False,
            }
        )

        task1_acc = valid['integration_tests:candidate/accuracy']
        task2_acc = valid['integration_tests:multiturnCandidate/accuracy']
        total_acc = valid['accuracy']
        self.assertEqual(
            total_acc,
            0.5 * (task1_acc.value() + task2_acc.value()),
            'Task accuracy is averaged incorrectly',
        )

        valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': False,
            }
        )
        task1_acc = valid['integration_tests:candidate/accuracy']
        task2_acc = valid['integration_tests:multiturnCandidate/accuracy']
        total_acc = valid['accuracy']
        # metrics should be averaged equally across tasks
        self.assertEqual(
            total_acc,
            0.5 * (task1_acc.value() + task2_acc.value()),
            'Task accuracy is averaged incorrectly',
        )

    def test_multitasking_id_overlap(self):
        pp = ParlaiParser()
        opt = pp.parse_args(['--task', 'integration_tests,integration_tests'])
        with self.assertLogs(logging.logger) as cm:
            self.world = create_task(opt, None)
            self.assertIn(
                "have overlap in id 'integration_tests'", "\n".join(cm.output)
            )

    def _test_opt_step_opts(self, update_freq: int):
        """
        Test -tstep, -vstep, -lstep.

        :param update_freq:
            update frequency

        We copy train_model from testing_utils to directly access train loop.
        """
        import parlai.scripts.train_model as tms

        num_train_steps = 1001
        num_validations = 10
        num_logs = 100

        def get_tl(tmpdir):
            opt = {
                'task': 'integration_tests',
                'model': 'parlai.agents.test_agents.test_agents:MockTrainUpdatesAgent',
                'model_file': os.path.join(tmpdir, 'model'),
                'dict_file': os.path.join(tmpdir, 'model.dict'),
                # step opts
                'max_train_steps': num_train_steps,
                'validation_every_n_steps': int(num_train_steps / num_validations),
                'log_every_n_steps': int(num_train_steps / num_logs),
                'update_freq': update_freq,
            }
            parser = tms.setup_args()
            parser.set_params(**opt)
            popt = parser.parse_args([])
            for k, v in opt.items():
                popt[k] = v
            return tms.TrainLoop(popt)

        with testing_utils.capture_output(), testing_utils.tempdir() as tmpdir:
            tl = get_tl(tmpdir)
            valid, _ = tl.train()

        self.assertEqual(
            tl.valid_reports[-1]['total_train_updates'], num_train_steps - 1
        )
        self.assertEqual(len(tl.valid_reports), num_validations)
        self.assertEqual(len(tl.train_reports), num_logs)  # log every valid as well

    def test_opt_step(self):
        self._test_opt_step_opts(1)

    def test_opt_step_update_freq_2(self):
        self._test_opt_step_opts(2)

    def test_save_world_logs(self):
        """
        Test that we can save world logs from train model.
        """
        with testing_utils.tempdir() as tmpdir:
            log_report = os.path.join(tmpdir, 'world_logs.jsonl')
            valid, test = testing_utils.train_model(
                {
                    'task': 'integration_tests',
                    'validation_max_exs': 10,
                    'model': 'repeat_label',
                    'short_final_eval': True,
                    'num_epochs': 1.0,
                    'world_logs': log_report,
                }
            )
            with PathManager.open(log_report) as f:
                json_lines = f.readlines()
            assert len(json_lines) == 10

    def test_save_multiple_world_logs(self):
        """
        Test that we can save multiple world_logs from train model on multiple tasks.
        """
        with testing_utils.tempdir() as tmpdir:
            log_report = os.path.join(tmpdir, 'world_logs.jsonl')
            multitask = 'integration_tests,integration_tests:ReverseTeacher'
            valid, test = testing_utils.train_model(
                {
                    'task': multitask,
                    'validation_max_exs': 10,
                    'model': 'repeat_label',
                    'short_final_eval': True,
                    'num_epochs': 1.0,
                    'world_logs': log_report,
                }
            )

            for task in multitask.split(','):
                task_log_report = get_task_world_logs(
                    task, log_report, is_multitask=True
                )
                with PathManager.open(task_log_report) as f:
                    json_lines = f.readlines()
                assert len(json_lines) == 5

    def test_save_multiple_world_logs_evaltask(self):
        """
        Test that we can save multiple world_logs from train model on multiple tasks
        where there are more evaltasks than tasks.
        """
        with testing_utils.tempdir() as tmpdir:
            log_report = os.path.join(tmpdir, 'world_logs.jsonl')
            multitask = 'integration_tests,integration_tests:ReverseTeacher'
            evaltask = 'integration_tests,integration_tests:mutators=flatten,integration_tests:ReverseTeacher:mutator=reverse'
            valid, test = testing_utils.train_model(
                {
                    'task': multitask,
                    'evaltask': evaltask,
                    'validation_max_exs': 10,
                    'model': 'repeat_label',
                    'short_final_eval': True,
                    'num_epochs': 1.0,
                    'world_logs': log_report,
                }
            )

            for task in evaltask.split(','):
                task_log_report = get_task_world_logs(
                    task, log_report, is_multitask=True
                )
                with PathManager.open(task_log_report) as f:
                    json_lines = f.readlines()
                assert len(json_lines) == 4

    def test_save_multiple_world_logs_mutator(self):
        """
        Test that we can save multiple world_logs from train model on multiple tasks
        with mutators present.
        """
        with testing_utils.tempdir() as tmpdir:
            log_report = os.path.join(tmpdir, 'world_logs.jsonl')
            multitask = 'integration_tests:mutators=flatten,integration_tests:ReverseTeacher:mutator=reverse'
            valid, test = testing_utils.train_model(
                {
                    'task': multitask,
                    'validation_max_exs': 10,
                    'model': 'repeat_label',
                    'short_final_eval': True,
                    'num_epochs': 1.0,
                    'world_logs': log_report,
                }
            )

            for task in multitask.split(','):
                task_log_report = get_task_world_logs(
                    task, log_report, is_multitask=True
                )
                with PathManager.open(task_log_report) as f:
                    json_lines = f.readlines()
                assert len(json_lines) == 5


@register_agent("fake_report")
class FakeReportAgent(Agent):
    def __init__(self, opt, shared=None):
        self.count = 0
        super().__init__(opt, shared)

    def report(self):
        if self.count == 0:
            # initial score
            return {'loss': AverageMetric(3)}
        elif self.count == 1:
            # don't save the second validation
            return {'loss': AverageMetric(4)}
        else:
            # do save the third validation
            return {'loss': AverageMetric(2)}

    def receive_metrics(self, report):
        self.count += 1

    def act(self):
        return {}

    def save(self, fname):
        self.opt.save(fname + ".opt")
        with open(fname, "w") as f:
            f.write("Lol")

    def load(self, fname):
        pass


class TestValidationImpatience(unittest.TestCase):
    """
    Tests to check we handle impatience correctly upon preemption.
    """

    def test_impatience(self, **kwargs):
        from parlai.scripts.train_model import TrainModel, TrainLoop

        # shallow copy to prevent overwrites
        kwargs = kwargs.copy()
        with testing_utils.tempdir() as tmpdir:
            kwargs['model'] = 'fake_report'
            kwargs['task'] = 'integration_tests'
            kwargs['validation_metric'] = 'loss'
            kwargs['model_file'] = os.path.join(tmpdir, 'model')
            kwargs['dict_file'] = 'zoo:unittest/transformer_generator2/model.dict'
            kwargs['log_every_n_steps'] = 1
            kwargs['validation_every_n_steps'] = 10
            kwargs['max_train_steps'] = 100
            kwargs['save_after_valid'] = True
            opt = TrainModel.setup_args().parse_kwargs(**kwargs)

            logs_first = []
            main_loop = TrainLoop(opt)

            for i, train_step_log in enumerate(main_loop.train_steps()):
                if i % 10 == 1:
                    # simulate preemption
                    # load from preempted and check variables are the same
                    preempt_loop = TrainLoop(opt)
                    # assert main_loop.impatience == preempt_loop.impatience
                    # assert main_loop.last_valid_epoch == preempt_loop.last_valid_epoch
                    # assert main_loop.best_valid == preempt_loop.best_valid
                    print(i, preempt_loop.impatience, preempt_loop.best_valid)
                    if i == 1:
                        assert preempt_loop.impatience == 0
                        assert preempt_loop.best_valid is None
                    elif i == 11:
                        assert preempt_loop.impatience == 0
                        assert preempt_loop.best_valid == 3
                    elif i == 21:
                        assert preempt_loop.impatience == 1
                        assert preempt_loop.best_valid == 3
                    elif i == 31:
                        assert preempt_loop.impatience == 0
                        assert preempt_loop.best_valid == 2
                    else:
                        assert preempt_loop.impatience == (i - 31) // 10
                        assert preempt_loop.best_valid == 2
