#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import create_agent_from_model_file
import parlai.utils.testing as testing_utils

import os
import unittest

SKIP_TESTS = False
try:
    from parlai.agents.tfidf_retriever.tfidf_retriever import (  # noqa: F401
        TfidfRetrieverAgent,
    )
except ImportError:
    SKIP_TESTS = True


class TestTfidfRetriever(unittest.TestCase):
    """
    Basic tests on the display_data.py example.
    """

    @unittest.skipIf(SKIP_TESTS, "Missing  Tfidf dependencies.")
    def test_sparse_tfidf_multiworkers(self):
        with testing_utils.tempdir() as tmpdir:
            MODEL_FILE = os.path.join(tmpdir, 'tmp_test_babi')
            testing_utils.train_model(
                dict(
                    model='tfidf_retriever',
                    task='babi:task1k:1',
                    model_file=MODEL_FILE,
                    retriever_numworkers=4,
                    retriever_hashsize=2**8,
                    retriever_tokenizer='simple',
                    datatype='train:ordered',
                    batchsize=1,
                    num_epochs=1,
                )
            )

            agent = create_agent_from_model_file(MODEL_FILE)

            obs = {
                'text': (
                    'Mary moved to the bathroom. John went to the hallway. '
                    'Where is Mary?'
                ),
                'episode_done': True,
            }
            agent.observe(obs)
            reply = agent.act()
            assert reply['text'] == 'bathroom'

            ANS = 'The one true label.'
            new_example = {
                'text': 'A bunch of new words that are not in the other task, '
                'which the model should be able to use to identify '
                'this label.',
                'labels': [ANS],
                'episode_done': True,
            }
            agent.observe(new_example)
            reply = agent.act()
            assert 'text' in reply and reply['text'] == ANS

            new_example.pop('labels')
            agent.observe(new_example)
            reply = agent.act()
            assert reply['text'] == ANS

    @unittest.skipIf(SKIP_TESTS, "Missing  Tfidf dependencies.")
    def test_sparse_tfidf_retriever_singlethread(self):
        with testing_utils.tempdir() as tmpdir:
            MODEL_FILE = os.path.join(tmpdir, 'tmp_test_babi')
            testing_utils.train_model(
                dict(
                    model='tfidf_retriever',
                    task='babi:task1k:1',
                    model_file=MODEL_FILE,
                    retriever_numworkers=1,
                    retriever_hashsize=2**8,
                    retriever_tokenizer='simple',
                    datatype='train:ordered',
                    batchsize=1,
                    num_epochs=1,
                )
            )

            agent = create_agent_from_model_file(MODEL_FILE)

            obs = {
                'text': (
                    'Mary moved to the bathroom. John went to the hallway. '
                    'Where is Mary?'
                ),
                'episode_done': True,
            }
            agent.observe(obs)
            reply = agent.act()
            assert reply['text'] == 'bathroom'

            ANS = 'The one true label.'
            new_example = {
                'text': 'A bunch of new words that are not in the other task, '
                'which the model should be able to use to identify '
                'this label.',
                'labels': [ANS],
                'episode_done': True,
            }
            agent.observe(new_example)
            reply = agent.act()
            assert 'text' in reply and reply['text'] == ANS

            new_example.pop('labels')
            agent.observe(new_example)
            reply = agent.act()
            assert reply['text'] == ANS

    @unittest.skipIf(SKIP_TESTS, "Missing  Tfidf dependencies.")
    def test_sparse_tfidf_retriever_regexp(self):
        with testing_utils.tempdir() as tmpdir:
            MODEL_FILE = os.path.join(tmpdir, 'tmp_test_babi')
            testing_utils.train_model(
                dict(
                    model='tfidf_retriever',
                    task='babi:task1k:1',
                    model_file=MODEL_FILE,
                    retriever_tokenizer='regexp',
                    retriever_numworkers=4,
                    retriever_hashsize=2**8,
                    datatype='train:ordered',
                    batchsize=1,
                    num_epochs=1,
                )
            )

            agent = create_agent_from_model_file(MODEL_FILE)

            obs = {
                'text': (
                    'Mary moved to the bathroom. John went to the hallway. '
                    'Where is Mary?'
                ),
                'episode_done': True,
            }
            agent.observe(obs)
            reply = agent.act()
            assert reply['text'] == 'bathroom'

            ANS = 'The one true label.'
            new_example = {
                'text': 'A bunch of new words that are not in the other task, '
                'which the model should be able to use to identify '
                'this label.',
                'labels': [ANS],
                'episode_done': True,
            }
            agent.observe(new_example)
            reply = agent.act()
            assert 'text' in reply and reply['text'] == ANS

            new_example.pop('labels')
            agent.observe(new_example)
            reply = agent.act()
            assert reply['text'] == ANS


if __name__ == '__main__':
    unittest.main()
