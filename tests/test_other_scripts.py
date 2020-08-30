#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Catch all for a number of "other" scripts.
"""

import os
import unittest
import random
import parlai.utils.testing as testing_utils


class TestConvertToParlaiFormat(unittest.TestCase):
    def test_convert(self):
        from parlai.scripts.convert_data_to_parlai_format import (
            ConvertDataToParlaiFormat,
        )

        with testing_utils.tempdir() as tmpdir:
            fn = os.path.join(tmpdir, 'parlai.txt')
            ConvertDataToParlaiFormat.main(
                task='integration_tests:nocandidate', outfile=fn
            )
            with open(fn) as f:
                assert (
                    f.readline() == 'text:4 1 3 2\tlabels:4 1 3 2\tepisode_done:True\n'
                )
                assert f.readline() == '\n'
                assert (
                    f.readline() == 'text:3 0 4 1\tlabels:3 0 4 1\tepisode_done:True\n'
                )
                assert f.readline() == '\n'
                assert (
                    f.readline() == 'text:5 1 6 3\tlabels:5 1 6 3\tepisode_done:True\n'
                )
                assert f.readline() == '\n'
                assert (
                    f.readline() == 'text:4 5 6 2\tlabels:4 5 6 2\tepisode_done:True\n'
                )
                assert f.readline() == '\n'
                assert (
                    f.readline() == 'text:0 5 3 1\tlabels:0 5 3 1\tepisode_done:True\n'
                )
                assert f.readline() == '\n'


class TestVerifyData(unittest.TestCase):
    def test_verify_data(self):
        from parlai.scripts.verify_data import VerifyData

        report = VerifyData.main(task='integration_tests')
        assert report['did_not_return_message'] == 0
        assert report['empty_string_label_candidates'] == 0
        assert report['exs'] == 500
        assert report['label_candidates_with_missing_label'] == 0
        assert report['missing_label_candidates'] == 0
        assert report['missing_labels'] == 0
        assert report['missing_text'] == 0

    def test_verify_bad_data(self):
        from parlai.scripts.verify_data import VerifyData

        random.seed(42)

        report = VerifyData.main(task='integration_tests:bad_example')
        assert report['did_not_return_message'] == 0
        assert report['empty_string_label_candidates'] == 63
        assert report['exs'] == 437
        assert report['label_candidates_with_missing_label'] == 188
        assert report['missing_label_candidates'] == 62
        assert report['missing_labels'] == 63
        assert report['missing_text'] == 62
