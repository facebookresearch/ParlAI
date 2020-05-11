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
import regex as re


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
            os.makedirs(os.path.join(data_path, 'ImageTeacher'))

            opt = {
                'task': 'integration_tests:ImageTeacher',
                'datapath': data_path,
                'image_mode': image_mode,
                'display_verbose': True,
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

    @testing_utils.skipUnlessTorch14
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
            with open(fp, "w") as f:
                f.write('id:test_file\ttext:input\tlabels:good label\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'display_verbose': True}
            testing_utils.display_data(opt)

    def test_bad_fileformat(self):
        """
        Checks that we fail to load a dataset where the use specified eval_labels.
        """
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.txt")
            with open(fp, "w") as f:
                f.write('id:test_file\ttext:input\teval_labels:bad label\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'display_verbose': True}
            with self.assertRaises(ValueError):
                testing_utils.display_data(opt)

    def test_no_text(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.txt")
            with open(fp, "w") as f:
                f.write('id:test_file\tlabels:bad label\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'display_verbose': True}
            with self.assertRaises(ValueError):
                testing_utils.display_data(opt)

    def test_no_labels(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.txt")
            with open(fp, "w") as f:
                f.write('id:test_file\ttext:bad text\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'display_verbose': True}
            with self.assertRaises(ValueError):
                testing_utils.display_data(opt)

    def test_one_episode(self):
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "badfile.txt")
            with open(fp, "w") as f:
                for _ in range(1000):
                    f.write('id:test_file\ttext:placeholder\tlabels:placeholder\n\n')
            opt = {'task': 'fromfile', 'fromfile_datapath': fp, 'display_verbose': True}
            with self.assertWarnsRegex(UserWarning, "long episode"):
                testing_utils.display_data(opt)

            # invert the logic of the assertion
            with self.assertRaises(self.failureException):
                fp = os.path.join(tmpdir, "goodfile.txt")
                with open(fp, "w") as f:
                    for _ in range(1000):
                        f.write(
                            'id:test_file\ttext:placeholder\tlabels:placeholder\tepisode_done:True\n\n'
                        )
                opt = {
                    'task': 'fromfile',
                    'fromfile_datapath': fp,
                    'display_verbose': True,
                }
                with self.assertWarnsRegex(UserWarning, "long episode"):
                    testing_utils.display_data(opt)


if __name__ == '__main__':
    unittest.main()
