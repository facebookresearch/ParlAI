#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import tempfile
import parlai.scripts.convert_data_to_parlai_format as cdpf
import parlai.scripts.data_stats as ds


class TestConvertToParlai(unittest.TestCase):
    def test_script(self):
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.close()
        cdpf.ConvertDataToParlaiFormat.main(task='integration_tests', outfile=tf.name)
        with open(tf.name) as f:
            lines = [l.strip() for l in f]
            lines = [l for l in lines if l]
            assert len(lines) == 500


class TestEvalDataStats(unittest.TestCase):
    def test_script(self):
        report = ds.DataStats.main(task='integration_tests')
        assert report['input/unique_tokens'] == 7
        assert report['labels/unique_tokens'] == 7
        assert report['both/unique_tokens'] == 7
        assert report['both/utterances'] == 1000
        assert report['input/utterances'] == 500
        assert report['labels/utterances'] == 500
        assert report['both/unique_utterances'] == 500
        assert report['input/unique_utterances'] == 500
        assert report['labels/unique_utterances'] == 500
        assert report['both/tokens'] == 4000
        assert report['input/tokens'] == 2000
        assert report['labels/tokens'] == 2000
        assert report['both/avg_utterance_length'] == 4
        assert report['input/avg_utterance_length'] == 4
        assert report['labels/avg_utterance_length'] == 4
