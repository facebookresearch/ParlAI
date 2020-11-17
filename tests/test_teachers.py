#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test Teachers.

A module for testing various teacher types in ParlAI
"""

import os
import unittest
from parlai.utils import testing as testing_utils
from parlai.core.teachers import DialogTeacher
from parlai.core.metrics import SumMetric
import regex as re
from parlai.core.message import Message
from parlai.core.opt import Opt
import parlai.utils.logging as logging
from parlai.utils.io import PathManager
from parlai.core.loader import register_agent
from collections import defaultdict
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent


class TestAbstractImageTeacher(unittest.TestCase):
    """
    Test AbstractImageTeacher.
    """

    def _test_display_output(self, image_mode):
        """
        Test display data output with given image_mode.
        """
        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir
            PathManager.mkdirs(os.path.join(data_path, 'ImageTeacher'))

            opt = {
                'task': 'integration_tests:ImageTeacher',
                'datapath': data_path,
                'image_mode': image_mode,
                'verbose': True,
            }
            output = testing_utils.display_data(opt)
            train_labels = re.findall(r"\[labels\].*\n", output[0])
            valid_labels = re.findall(r"\[eval_labels\].*\n", output[1])
            test_labels = re.findall(r"\[eval_labels\].*\n", output[2])

            for i, lbls in enumerate([train_labels, valid_labels, test_labels]):
                self.assertGreater(len(lbls), 0, 'DisplayData failed')
                self.assertEqual(len(lbls), len(set(lbls)), output[i])

    def test_display_data_no_image(self):
        """
        Test that, with no images loaded, all examples are different.
        """
        self._test_display_output('no_image_model')

    @testing_utils.skipUnlessVision
    @testing_utils.skipUnlessGPU
    def test_display_data_resnet(self):
        """
        Test that, with pre-loaded image features, all examples are different.
        """
        self._test_display_output('resnet152')


class TestParlAIDialogTeacher(unittest.TestCase):
    def test_good_fileformat(self):
        """
        Checks that we fail to load a dataset where the use specified eval_labels.
        """
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "goodfile.txt")
            with PathManager.open(fp, "w") as f:
                f.write('id:test_file\ttext:input\tlabels:good label\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'verbose': True}
            testing_utils.display_data(opt)

    def test_bad_fileformat(self):
        """
        Checks that we fail to load a dataset where the use specified eval_labels.
        """
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.txt")
            with PathManager.open(fp, "w") as f:
                f.write('id:test_file\ttext:input\teval_labels:bad label\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'verbose': True}
            with self.assertRaises(ValueError):
                testing_utils.display_data(opt)

    def test_no_text(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.txt")
            with PathManager.open(fp, "w") as f:
                f.write('id:test_file\tlabels:bad label\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'verbose': True}
            with self.assertRaises(ValueError):
                testing_utils.display_data(opt)

    def test_no_labels(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.txt")
            with PathManager.open(fp, "w") as f:
                f.write('id:test_file\ttext:bad text\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'verbose': True}
            with self.assertRaises(ValueError):
                testing_utils.display_data(opt)

    def test_one_episode(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.txt")
            with PathManager.open(fp, "w") as f:
                for _ in range(1000):
                    f.write('id:test_file\ttext:placeholder\tlabels:placeholder\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'verbose': True}
            with self.assertLogs(logger=logging.logger, level='DEBUG') as cm:
                testing_utils.display_data(opt)
                print("\n".join(cm.output))
                assert any('long episode' in l for l in cm.output)

            # invert the logic of the assertion
            with self.assertRaises(self.failureException):
                fp = os.path.join(tmpdir, "goodfile.txt")
                with PathManager.open(fp, "w") as f:
                    for _ in range(1000):
                        f.write(
                            'id:test_file\ttext:placeholder\tlabels:placeholder\tepisode_done:True\n\n'
                        )
                opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'verbose': True}
                with self.assertLogs(logger=logging.logger, level='DEBUG') as cm:
                    testing_utils.display_data(opt)
                    assert any('long episode' in l for l in cm.output)


class TestConversationTeacher(unittest.TestCase):
    def test_good_fileformat(self):
        """
        Checks that we succeed in loading a well formatted jsonl file.
        """
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "goodfile.jsonl")
            with PathManager.open(fp, "w") as f:
                f.write(
                    '{"dialog": [[{"text": "Hi.", "id": "speaker1"}, {"text": "Hello.", "id": "speaker2"}]]}\n'
                )
            opt = {'task': 'jsonfile', 'jsonfile_datapath': fp, 'verbose': True}
            testing_utils.display_data(opt)

    def test_no_text(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.jsonl")
            with PathManager.open(fp, "w") as f:
                f.write(
                    '{"dialog": [[{"id": "speaker1"}, {"text": "Hello.", "id": "speaker2"}]]}\n'
                )
            opt = {'task': 'jsonfile', 'jsonfile_datapath': fp, 'verbose': True}
            with self.assertRaises(AttributeError):
                testing_utils.display_data(opt)

    def test_firstspeaker_label(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "goodfile.jsonl")
            with PathManager.open(fp, "w") as f:
                f.write(
                    '{"dialog": [[{"text": "Hi.", "id": "speaker1"}, {"text": "Hello.", "id": "speaker2"}]]}\n'
                )
            opt = {
                'task': 'jsonfile',
                'jsonfile_datapath': fp,
                'verbose': True,
                'label_turns': 'firstspeaker',
            }
            train_out, valid_out, test_out = testing_utils.display_data(opt)
            texts = [
                l.split(':', 1)[-1].strip()
                for l in train_out.split('\n')
                if l in train_out
                if 'text' in l
            ]
            labels = [
                l.split(':', 1)[-1].strip()
                for l in train_out.split('\n')
                if l in train_out
                if 'labels' in l
            ]
            self.assertEqual(texts[0], '__SILENCE__')
            self.assertEqual(labels[0], 'Hi.')

    def test_secondspeaker_label(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "goodfile.jsonl")
            with PathManager.open(fp, "w") as f:
                f.write(
                    '{"dialog": [[{"text": "Hi.", "id": "speaker1"}, {"text": "Hello.", "id": "speaker2"}]]}\n'
                )
            opt = {
                'task': 'jsonfile',
                'jsonfile_datapath': fp,
                'verbose': True,
                'label_turns': 'secondspeaker',
            }
            train_out, valid_out, test_out = testing_utils.display_data(opt)
            texts = [
                l.split(':', 1)[-1].strip()
                for l in train_out.split('\n')
                if l in train_out
                if 'text' in l
            ]
            labels = [
                l.split(':', 1)[-1].strip()
                for l in train_out.split('\n')
                if l in train_out
                if 'labels' in l
            ]
            self.assertEqual(texts[0], 'Hi.')
            self.assertEqual(labels[0], 'Hello.')

    def test_both_label(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "goodfile.jsonl")
            with PathManager.open(fp, "w") as f:
                f.write(
                    '{"dialog": [[{"text": "Hi.", "id": "speaker1"}, {"text": "Hello.", "id": "speaker2"}]]}\n'
                )
            opt = {
                'task': 'jsonfile',
                'jsonfile_datapath': fp,
                'verbose': True,
                'label_turns': 'both',
            }
            train_out, valid_out, test_out = testing_utils.display_data(opt)
            texts = [
                l.split(':', 1)[-1].strip()
                for l in train_out.split('\n')
                if l in train_out
                if 'text' in l
            ]
            labels = [
                l.split(':', 1)[-1].strip()
                for l in train_out.split('\n')
                if l in train_out
                if 'labels' in l
            ]
            num_episodes = train_out.count("END OF EPISODE")
            self.assertEqual(texts[0], '__SILENCE__')
            self.assertEqual(labels[0], 'Hi.')
            self.assertEqual(texts[1], 'Hi.')
            self.assertEqual(labels[1], 'Hello.')
            self.assertEqual(num_episodes, 2)


@register_agent("unique_examples")
class UniqueExamplesAgent(RepeatLabelAgent):
    """
    Simple agent which asserts that it has only seen unique examples.

    Useful for debugging. Inherits from RepeatLabelAgent.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.unique_examples = defaultdict(int)

    def reset(self):
        super().reset()
        self.unique_examples = defaultdict(int)

    def act(self):
        obs = self.observation
        text = obs.get('text')
        if text in self.unique_examples:
            raise RuntimeError(f'Already saw example: {text}')
        else:
            self.unique_examples[text] += 1

        return super().act()


class TestChunkTeacher(unittest.TestCase):
    """
    Test chunked teacher.
    """

    def test_no_batched(self):
        valid, test = testing_utils.eval_model(
            dict(task='integration_tests:chunky', model='repeat_label'),
            valid_datatype='valid:stream',
            test_datatype='test:stream',
        )
        assert valid['exs'] == 100
        assert test['exs'] == 100

    def test_batched(self):
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:chunky',
                model='parlai.agents.test_agents.test_agents:MockTorchAgent',
                batchsize=32,
            ),
            valid_datatype='valid:stream',
            test_datatype='test:stream',
        )
        assert valid['exs'] == 100
        assert test['exs'] == 100

    def test_dynamic_batched(self):
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:chunky',
                model='parlai.agents.test_agents.test_agents:MockTorchAgent',
                datatype='valid:stream',
                batchsize=32,
                truncate=16,
                dynamic_batching='full',
            ),
            valid_datatype='valid:stream',
            test_datatype='test:stream',
        )
        assert valid['exs'] == 100
        assert test['exs'] == 100

    def test_stream_only(self):
        with self.assertRaises(ValueError):
            valid, test = testing_utils.eval_model(
                dict(
                    task='integration_tests:chunky',
                    model='parlai.agents.test_agents.test_agents:MockTorchAgent',
                    batchsize=32,
                ),
                valid_datatype='valid',
            )

        with self.assertRaises(ValueError):
            valid, test = testing_utils.eval_model(
                dict(
                    task='integration_tests:chunky',
                    model='parlai.agents.test_agents.test_agents:MockTorchAgent',
                    batchsize=32,
                ),
                valid_datatype='valid:stream',
                test_datatype='test',
            )

    def test_slow_loading(self):
        """
        Test that a slow loading teacher sees the right examples during validation.
        """
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')
            valid, test = testing_utils.train_model(
                dict(
                    task='integration_tests:chunky_unique_slow',
                    model='unique_examples',
                    model_file=model_file,
                    datatype='train:stream',
                    num_epochs=0.5,
                    validation_every_n_epochs=0.1,
                    batchsize=1,
                    dynamic_batching='full',
                    dict_maxexs=0,
                )
            )


class CustomEvaluationTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = 'mock'
        super().__init__(opt, shared)

    def custom_evaluation(self, teacher_action, label, model_response):
        self.metrics.add('contains1', SumMetric(int('1' in model_response['text'])))

    def setup_data(self, fold):
        yield ('1 2', '1 2'), True
        yield ('3 4', '3 4'), True


class TestCustomEvaluation(unittest.TestCase):
    def test_custom_eval(self):
        opt = {'task': 'custom', 'datatype': 'valid'}
        teacher = CustomEvaluationTeacher(opt)
        teacher.act()
        teacher.observe({'text': 'a b'})
        teacher.act()
        teacher.observe({'text': '1 2'})
        report = teacher.report()
        assert 'contains1' in report
        assert report['contains1'] == 1
        assert report['exs'] == 2


class _MockTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = 'mock'
        super().__init__(opt)


class TupleTeacher(_MockTeacher):
    def setup_data(self, datafile):
        for _ in range(3):
            for j in range(1, 4):
                yield (str(j), str(j * 2)), j == 1


class DictTeacher(_MockTeacher):
    def setup_data(self, datafile):
        for _ in range(3):
            for j in range(1, 4):
                yield {'text': str(j), 'label': str(j * 2)}, j == 1


class MessageTeacher(_MockTeacher):
    def setup_data(self, datafile):
        for _ in range(3):
            for j in range(1, 4):
                yield Message({'text': str(j), 'label': str(j * 2)}), j == 1


class NoDatafileTeacher(DialogTeacher):
    def setup_data(self, datafile):
        yield Message({'text': datafile, 'label': datafile}), True


class ViolationTeacher(_MockTeacher):
    def setup_data(self, datafile):
        yield {'text': 'foo', 'episode_done': True}, True


class TestDialogTeacher(unittest.TestCase):
    def test_nodatafile(self):
        for dt in [
            'train:ordered',
            'train:stream:ordered',
            'valid',
            'test',
            'valid:stream',
            'test:stream',
        ]:
            opt = Opt({'datatype': dt, 'datapath': '/tmp', 'task': 'test'})
            with self.assertRaises(KeyError):
                NoDatafileTeacher(opt)

    def _verify_act(self, act, goal_text, goal_label, episode_done):
        assert 'eval_labels' in act or 'labels' in act
        labels = act.get('labels', act.get('eval_labels'))
        assert isinstance(labels, tuple)
        assert len(labels) == 1
        assert act['text'] == str(goal_text)
        assert labels[0] == str(goal_label)

    def _test_iterate(self, teacher_class):
        for dt in [
            'train:ordered',
            'train:stream:ordered',
            'valid',
            'test',
            'valid:stream',
            'test:stream',
        ]:
            opt = Opt({'datatype': dt, 'datapath': '/tmp', 'task': 'test'})
            teacher = teacher_class(opt)

            self._verify_act(teacher.act(), 1, 2, False)
            self._verify_act(teacher.act(), 2, 4, False)
            self._verify_act(teacher.act(), 3, 6, True)

            self._verify_act(teacher.act(), 1, 2, False)
            self._verify_act(teacher.act(), 2, 4, False)
            self._verify_act(teacher.act(), 3, 6, True)

            self._verify_act(teacher.act(), 1, 2, False)
            self._verify_act(teacher.act(), 2, 4, False)
            self._verify_act(teacher.act(), 3, 6, True)

            assert teacher.epoch_done()

    def test_tuple_teacher(self):
        self._test_iterate(TupleTeacher)

    def test_dict_teacher(self):
        self._test_iterate(DictTeacher)

    def test_message_teacher(self):
        self._test_iterate(MessageTeacher)

    def test_violation_teacher(self):
        with self.assertRaises(KeyError):
            self._test_iterate(ViolationTeacher)


if __name__ == '__main__':
    unittest.main()
