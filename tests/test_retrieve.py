# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import os
import threading
from tqdm import tqdm
import unittest

from parlai.agents.ir_baseline.ir_retrieve import StringMatchRetrieverAgent
from parlai.agents.ir_baseline.ir_util import (
    DEFAULT_LENGTH_PENALTY,
)
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from examples.build_dict import build_dict
from examples.build_retriever import build_retriever


class TestStringMatchRetriever(unittest.TestCase):
    """Basic tests on the built-in parlai StringMatchRetriever."""

    test_facts = [
        "Unconditional directed_by Brent McCorkle",
        "Unconditional written_by Brent McCorkle",
        "Unconditional release_year 2012",
        "Unconditional has_genre Drama",
        "Unconditional has_plot A woman's idyllic life is shattered when " +
        "her husband is killed in a senseless act of violence. As she " +
        "prepares to take matters into her own hands, two unexpected " +
        "encounters begin to change everything.",
        "Siam Sunset directed_by John Polson",
        "Siam Sunset starred_actors Linus Roache, Danielle Cormack",
        "Siam Sunset release_year 1999",
        "Siam Sunset has_genre Comedy",
        "Siam Sunset has_plot A British design executive, who seemingly " +
        "has everything going for him has his life totally changed when " +
        "a refrigerator falls from an aircraft and lands on his wife. " +
        "He decides to getaway ...",
    ]

    def test_retriever(self):
        # build a dict-file
        TMP_PATH = '/tmp/parlai_test_retrieve/'
        if not os.path.isdir(TMP_PATH):
            os.mkdir(TMP_PATH)
        DICT_FILE = TMP_PATH + 'dict.tsv'
        if os.path.isfile(DICT_FILE):
            os.remove(DICT_FILE)
        args = []
        argparser = ParlaiParser()
        DictionaryAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args(args)
        dict_agent = DictionaryAgent(opt)
        for fact in self.test_facts:
            dict_agent.observe({'text': fact})
            dict_agent.act()
        dict_agent.save(DICT_FILE)
        # test retriever
        RETRIEVER_FILE = TMP_PATH + 'retriever.tsv'
        if os.path.isfile(RETRIEVER_FILE):
            os.remove(RETRIEVER_FILE)
        args = [
            '--dict-file',
            DICT_FILE,
            '--retriever-file',
            RETRIEVER_FILE,
        ]
        argparser = ParlaiParser()
        DictionaryAgent.add_cmdline_args(argparser)
        StringMatchRetrieverAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args(args)
        my_retriever = StringMatchRetrieverAgent(opt)
        for fact in self.test_facts:
            my_retriever.observe({'text': fact})
            my_retriever.act()
        self._test_retriever_functionality(my_retriever)
        # test save and load
        my_retriever.save()
        my_retriever2 = StringMatchRetrieverAgent(opt)
        self._test_retriever_functionality(my_retriever2)
        # test multi-thread
        self._test_retriever_multithread(opt)

    def _test_retriever_functionality(self, my_retriever):
        # test that random facts are retrieved: i.e., not the same fact being returned everytime
        # If you are unlucky, there is a very slim chance (1/5^4 < 1%) that this test will fail.
        ans1 = my_retriever.retrieve("Unconditional", 1, ordered_randomly=True)
        ans2 = my_retriever.retrieve("Unconditional", 1, ordered_randomly=True)
        ans3 = my_retriever.retrieve("Unconditional", 1, ordered_randomly=True)
        ans4 = my_retriever.retrieve("Unconditional", 1, ordered_randomly=True)
        self.assertGreater(len(set(ans1 + ans2 + ans3 + ans4)), 1)
        ans5 = my_retriever.retrieve("Unconditional directed_by Brent", 2)
        self.assertEqual(list(ans5), [
            "Unconditional directed_by Brent McCorkle",
            "Unconditional written_by Brent McCorkle",
            ])
        # test input string that are not in facts
        ans6 = my_retriever.retrieve("not_in_facts", 1)
        self.assertEqual(ans6, [])
        # test input string that are partially not in facts
        ans7 = my_retriever.retrieve("directed_by Brent not_in_facts", 1)
        self.assertEqual(list(ans7), ["Unconditional directed_by Brent McCorkle"])
        # test input string that are stop words
        ans8 = my_retriever.retrieve("of", 1)
        self.assertEqual(ans8, [])


    def test_build_retriever(self, rebuild=True):
        # build dict
        TMP_PATH = '/tmp/parlai_test_build_retriever/'
        if not os.path.isdir(TMP_PATH):
            os.mkdir(TMP_PATH)
        DICT_FILE = TMP_PATH + 'dict.tsv'
        if os.path.isfile(DICT_FILE) and rebuild:
            os.remove(DICT_FILE)
        RETRIEVER_FILE = TMP_PATH + 'retrieve.tsv'
        if os.path.isfile(RETRIEVER_FILE) and rebuild:
            os.remove(RETRIEVER_FILE)
        DATABASE = 'wikimovies:KB:kb'
        args = [
            '--dict-file',
            DICT_FILE,
            '-t',
            DATABASE,
            '--retriever-file',
            RETRIEVER_FILE,
            '--retriever-maxexs',
            0,
        ]
        argparser = ParlaiParser()
        DictionaryAgent.add_cmdline_args(argparser)
        StringMatchRetrieverAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args(args)
        build_dict(opt)
        # build retriever
        build_retriever(opt)
        # test retriever
        my_retriever = StringMatchRetrieverAgent(opt)
        ans1 = my_retriever.retrieve("who directed Jurassic park", 10)
        self.assertTrue("Jurassic Park directed_by Steven Spielberg" in list(ans1))
        ans2 = my_retriever.retrieve("what movies did Steven Spielberg direct", 15)
        self.assertTrue("Jurassic Park directed_by Steven Spielberg" in list(ans2))


    def _test_retriever_multithread(self, opt):
        # prepare lock for using multi-thread
        StringMatchRetrieverAgent.prepare_multi_thread()

        def _trivial_insert(opt, fact, insert_times):
            retriever = StringMatchRetrieverAgent(opt)
            for _ in range(insert_times):
                retriever.observe({'text': fact})
                retriever.act()
            retriever.save()

        threads = []
        for ind in range(10):
            threads.append(threading.Thread(target=_trivial_insert,
                                            args=(opt, 'x' if ind % 2 == 1 else 'y', 1000)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # check token 'x' only appears in fact 'x',
        # and # of 'x' facts is half of total insertion
        retriever = StringMatchRetrieverAgent(opt)
        for row in retriever.cursor.execute(
            ("SELECT t2.fact, COUNT(*) FROM ((SELECT fact_id FROM %s WHERE token=?) AS t1 JOIN " +
            "%s AS t2 ON t1.fact_id = t2.fact_id) ")
            % (StringMatchRetrieverAgent.FREQ_TABLE_NAME, StringMatchRetrieverAgent.DOC_TABLE_NAME),
            'x',
        ):
            self.assertEqual(row, ('x', 1000 * 10 / 2))

if __name__ == '__main__':
    logging.basicConfig(format='[ *%(levelname)s* ] %(message)s', level=logging.INFO)
    unittest.main()
