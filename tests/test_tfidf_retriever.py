# Copyright 2004-present Facebook. All Rights Reserved.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

import os
import unittest


class TestTfidfRetriever(unittest.TestCase):
    """Basic tests on the display_data.py example."""

    def test_sparse_tfidf_retriever(self):
        try:
            from parlai.agents.tfidf_retriever.tfidf_retriever import TfidfRetrieverAgent
        except ImportError as e:
            if 'pip install' in e.msg or 'pytorch' in e.msg:
                print('Skipping TestTfidfRetriever, missing optional pip packages or pytorch.')
                return

        MODEL_FILE = '/tmp/tmp_test_babi'
        DB_PATH = '/tmp/tmp_test_babi.db'
        TFIDF_PATH = '/tmp/tmp_test_babi.tfidf'
        try:
            parser = ParlaiParser(True, True)
            parser.set_defaults(
                model='tfidf_retriever',
                task='babi:task1k:1',
                model_file=MODEL_FILE,
                retriever_numworkers=4,
                retriever_hashsize=2**8,
                datatype='train:ordered',
                num_epochs=1
            )
            opt = parser.parse_args(print_args=False)
            agent = create_agent(opt)
            train_world = create_task(opt, agent)
            # pass examples to dictionary
            while not train_world.epoch_done():
                train_world.parley()

            obs = {
                'text': 'Mary moved to the bathroom. John went to the hallway. Where is Mary?',
                'episode_done': True
            }
            agent.observe(obs)
            reply = agent.act()
            assert reply['text'] == 'bathroom'

            ANS = 'The one true label.'
            new_example = {
                'text': 'A bunch of new words that are not in the other task, which the model should be able to use to identify this label.',
                'labels': [ANS],
                'episode_done': True
            }
            agent.observe(new_example)
            reply = agent.act()
            assert 'text' in reply and reply['text'] == ANS

            new_example.pop('labels')
            agent.observe(new_example)
            reply = agent.act()
            assert reply['text'] == ANS
        finally:
            # clean up files
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            if os.path.exists(TFIDF_PATH + '.npz'):
                os.remove(TFIDF_PATH + '.npz')


if __name__ == '__main__':
    unittest.main()
