# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import os
from multiprocessing import Process
import unittest

from examples.build_retriever import build_retriever
from parlai.agents.ir_baseline.ir_retrieve import StringMatchRetrieverAgent
from parlai.core.params import ParlaiParser


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

    def _init_opt(self, rebuild=True, numthreads=1):
        TMP_PATH = '/tmp/parlai_test_retrieve/'
        if not os.path.isdir(TMP_PATH):
            os.mkdir(TMP_PATH)
        RETRIEVER_FILE = TMP_PATH + 'retriever.npz'
        RETRIEVER_DB_FILE = TMP_PATH + 'retriever_database.db'
        RETRIEVER_TOKEN_FILE = TMP_PATH + 'retriever_token.npy'
        if os.path.isfile(RETRIEVER_FILE) and rebuild:
            os.remove(RETRIEVER_FILE)
        if os.path.isfile(RETRIEVER_DB_FILE) and rebuild:
            os.remove(RETRIEVER_DB_FILE)
        if os.path.isfile(RETRIEVER_TOKEN_FILE) and rebuild:
            os.remove(RETRIEVER_TOKEN_FILE)
        TASK = 'wikimovies:KB:kb'
        args = [
            '-t',
            TASK,
            '--numthreads',
            str(numthreads),
            '--retriever-file',
            RETRIEVER_FILE,
            '--retriever-database',
            RETRIEVER_DB_FILE,
            '--retriever-tokens',
            RETRIEVER_TOKEN_FILE,
            '--retriever-maxexs',
            '0',
        ]
        argparser = ParlaiParser()
        StringMatchRetrieverAgent.add_cmdline_args(argparser)
        return argparser.parse_args(args)

    def test_retriever(self):
        opt = self._init_opt()
        my_retriever = StringMatchRetrieverAgent(opt)
        for fact in self.test_facts:
            my_retriever.observe({'text': fact})
            my_retriever.act()
        my_retriever.save()
        self._test_retriever_functionality(my_retriever)
        # test dynamic update
        my_retriever.observe({'text': "test additional"})
        my_retriever.act()
        ans1 = list(my_retriever.retrieve('test', 1))
        self.assertEqual(ans1, ["test additional"])
        # test save/load
        my_retriever2 = StringMatchRetrieverAgent(opt)
        self._test_retriever_functionality(my_retriever2)

    def test_retriever_multithread(self):
        numthreads = 10
        num_iter = 1000
        opt = self._init_opt(numthreads=numthreads)
        my_retriever = StringMatchRetrieverAgent(opt)


        class LoadTestProcess(Process):

            def __init__(self, process_id, tests, num_iter, opt, retriever):
                self.process_id = process_id
                self.shared = retriever.share()
                self.opt = opt
                self.tests = tests
                self.num_iter = num_iter
                super().__init__()

            def run(self):
                my_retriever = StringMatchRetrieverAgent(self.opt, self.shared)
                for ind in range(self.num_iter):
                    for fact in self.tests:
                        my_retriever.observe(
                            {'text': ('process%d iter%d : ' + fact) % (self.process_id, ind)}
                        )
                        my_retriever.act()
                my_retriever.shutdown()

        for ind in range(numthreads):
            LoadTestProcess(ind, self.test_facts, num_iter, opt, my_retriever).start()
        my_retriever.shutdown()
        my_retriever2 = StringMatchRetrieverAgent(opt)
        ans1 = list(my_retriever.retrieve("iter1 Drama", numthreads))
        self.assertEqual(set(ans1), set([
            "process%d iter1 : Unconditional has_genre Drama" % ind
            for ind in range(numthreads)]))


    def _test_retriever_functionality(self, my_retriever):
        ans5 = list(my_retriever.retrieve("Unconditional Brent", 2))
        self.assertEqual(ans5, [
            "Unconditional directed_by Brent McCorkle",
            "Unconditional written_by Brent McCorkle",
            ])
        # test input string that are not in facts
        ans6 = list(my_retriever.retrieve("not_in_facts", 1))
        self.assertEqual(ans6, [])
        # test input string that are partially not in facts
        ans7 = list(my_retriever.retrieve("directed_by Brent not_in_facts", 1))
        self.assertEqual(ans7, ["Unconditional directed_by Brent McCorkle"])
        # test input string that are stop words
        ans8 = list(my_retriever.retrieve("of", 1))
        self.assertEqual(ans8, [])


    def _test_retriever_functionality_wikimovie(self, my_retriever):
        ans1 = list(my_retriever.retrieve("who directed Jurassic park", 50))
        self.assertTrue("Jurassic Park directed_by Steven Spielberg" in ans1)
        ans2 = list(my_retriever.retrieve("what movies did Steven Spielberg direct", 50))
        self.assertTrue("Jurassic Park directed_by Steven Spielberg" in ans2)

    def test_build_retriever(self):
        opt = self._init_opt()
        build_retriever(opt)
        my_retriever = StringMatchRetrieverAgent(opt)
        self._test_retriever_functionality_wikimovie(my_retriever)

    def test_build_retriever_multithread(self):
        opt = self._init_opt(numthreads=2)
        build_retriever(opt)
        opt['numthreads'] = 1
        my_retriever = StringMatchRetrieverAgent(opt)
        self._test_retriever_functionality_wikimovie(my_retriever)

if __name__ == '__main__':
    logging.basicConfig(format='[ *%(levelname)s* ] %(message)s', level=logging.INFO)
    unittest.main()
