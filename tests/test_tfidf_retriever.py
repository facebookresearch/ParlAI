# Copyright 2004-present Facebook. All Rights Reserved.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

from parlai.agents.tfidf_retriever.tfidf_retriever import TfidfRetrieverAgent

import os
import unittest


class TestTfidfRetriever(unittest.TestCase):
    """Basic tests on the display_data.py example."""

    def test_sparse_tfidf_retriever(self):
        DB_PATH = '/tmp/tmp_test_babi.db'
        TFIDF_PATH = '/tmp/tmp_test_babi.tfidf'
        args = [
            '--model', 'tfidf_retriever',
            '--retriever-task', 'babi:task1k:1',
            '--retriever-dbpath', DB_PATH,
            '--retriever-tfidfpath', TFIDF_PATH,
            '--retriever-numworkers', '4',
            '--retriever-hashsize', str(2**8)
        ]
        try:
            parser = ParlaiParser(True, True)
            TfidfRetrieverAgent.add_cmdline_args(parser)
            opt = parser.parse_args(args, print_args=False)

            agent = create_agent(opt)

            obs = {
                'text': 'Mary moved to the bathroom. John went to the hallway. Where is Mary?',
            }
            agent.observe(obs)
            reply = agent.act()
            assert reply['text'] == 'bathroom'

            ANS = 'The one true label.'
            new_example = {
                'text': 'A bunch of new words that are not in the other task, which the model should be able to use to identify this label.',
                'labels': [ANS],
            }
            agent.observe(new_example)
            reply = agent.act()
            assert 'text' not in reply

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
