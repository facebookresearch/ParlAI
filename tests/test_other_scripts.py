#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Catch all for a number of "other" scripts.
"""

import os
import unittest
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


class TestVacuum(unittest.TestCase):
    def test_vacuum(self):
        with testing_utils.tempdir() as tmpdir:
            from parlai.scripts.vacuum import Vacuum

            model_file = os.path.join(tmpdir, 'model')
            valid, test = testing_utils.train_model(
                {
                    'task': 'integration_tests',
                    'optimizer': 'adam',
                    'learningrate': 0.01,
                    'model_file': model_file,
                    'num_epochs': 0.05,
                    'skip_generation': True,
                    'batchsize': 8,
                    # TODO: switch to test_agents/unigram
                    'model': 'transformer/generator',
                    'ffn_size': 32,
                    'embedding_size': 32,
                    'n_layers': 1,
                }
            )
            size_before = os.stat(model_file).st_size
            Vacuum.main(model_file=model_file)
            size_after = os.stat(model_file).st_size
            assert size_after < size_before
            assert os.path.exists(model_file + '.unvacuumed')
            valid2, test2 = testing_utils.eval_model(
                {'task': 'integration_tests', 'model_file': model_file, 'batchsize': 8}
            )
            for key in ['loss', 'exs', 'ppl', 'token_acc']:
                assert valid2[key] == valid[key], f"{key} score doesn't match"
                assert test2[key] == test[key], f"{key} score doesn't match"


class TestDetectOffensive(unittest.TestCase):
    def test_offensive(self):
        from parlai.scripts.detect_offensive_language import DetectOffensive

        report = DetectOffensive.main(
            task='babi:task1k:10', datatype='valid', safety='string_matcher'
        )
        assert report['string_offenses%'] == 0
        assert report['word_offenses'] == 0
        assert report['exs'] == 100


class TestParty(unittest.TestCase):
    def test_party(self):
        from parlai.scripts.party import Party

        Party.main(seconds=0.01)


class TestDataStats(unittest.TestCase):
    def test_simple(self):
        from parlai.scripts.data_stats import DataStats

        report = DataStats.main(task='integration_tests')
        assert report['both/avg_utterance_length'] == 4
        assert report['input/avg_utterance_length'] == 4
        assert report['labels/avg_utterance_length'] == 4
        assert report['both/tokens'] == 4000
        assert report['input/tokens'] == 2000
        assert report['labels/tokens'] == 2000
        assert report['both/unique_tokens'] == 7
        assert report['input/unique_tokens'] == 7
        assert report['labels/unique_tokens'] == 7
        assert report['both/unique_utterances'] == 500
        assert report['input/unique_utterances'] == 500
        assert report['labels/unique_utterances'] == 500
        assert report['both/utterances'] == 1000
        assert report['input/utterances'] == 500
        assert report['labels/utterances'] == 500


class TestProfileTrain(unittest.TestCase):
    """
    Test profile_train doesn't crash.
    """

    def test_cprofile(self):
        from parlai.scripts.profile_train import ProfileTrain

        with testing_utils.tempdir() as tmpdir:
            ProfileTrain.main(
                task='integration_tests:overfit',
                model='test_agents/unigram',
                model_file=os.path.join(tmpdir, 'model'),
                skip_generation=True,
            )

    def test_torch(self):
        from parlai.scripts.profile_train import ProfileTrain

        with testing_utils.tempdir() as tmpdir:
            ProfileTrain.main(
                task='integration_tests:overfit',
                model='test_agents/unigram',
                torch=True,
                model_file=os.path.join(tmpdir, 'model'),
                skip_generation=True,
            )

    @testing_utils.skipUnlessGPU
    def test_torch_cuda(self):
        from parlai.scripts.profile_train import ProfileTrain

        with testing_utils.tempdir() as tmpdir:
            ProfileTrain.main(
                task='integration_tests:overfit',
                model='test_agents/unigram',
                torch_cuda=True,
                model_file=os.path.join(tmpdir, 'model'),
                skip_generation=True,
            )


class TestTokenStats(unittest.TestCase):
    def test_token_stats(self):
        from parlai.scripts.token_stats import TokenStats
        from parlai.core.metrics import dict_report

        results = dict_report(TokenStats.main(task='integration_tests:multiturn'))
        assert results == {
            'exs': 2000,
            'max': 16,
            'mean': 7.5,
            'min': 1,
            'p01': 1,
            'p05': 1,
            'p10': 1,
            'p25': 4,
            'p50': 7.5,
            'p75': 11.5,
            'p90': 16,
            'p95': 16,
            'p99': 16,
            'p@128': 1,
        }
