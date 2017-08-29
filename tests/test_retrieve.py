# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import unittest

from parlai.core.dict import DictionaryAgent
from parlai.core.retrieve import StringMatchRetrieverAgent


class TestStringMatchRetriever(unittest.TestCase):
    """Basic tests on the built-in parlai StringMatchRetriever."""

    def test_retriever(self):
        opt = {
            'dict_max_ngram_size': 2,
            'dict_language': DictionaryAgent.default_lang,
            'dict_minfreq': DictionaryAgent.default_minfreq,
            'dict_nulltoken': DictionaryAgent.default_null,
            'dict_endtoken': DictionaryAgent.default_end,
            'dict_unktoken': DictionaryAgent.default_unk,
            'dict_starttoken': DictionaryAgent.default_start,
            'dict_maxexs': 100000,
        }
        my_retriever = StringMatchRetrieverAgent(opt)
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
        for fact in test_facts:
            my_retriever.observe({'text': fact})
            my_retriever.act()
        # test that random facts are retrieved: i.e., not the same fact being returned everytime
        # If you are unlucky, there is a very slim chance (1/5^4 < 1%) that this test will fail.
        ans1 = my_retriever.retrieve("Unconditional", 1)
        ans2 = my_retriever.retrieve("Unconditional", 1)
        ans3 = my_retriever.retrieve("Unconditional", 1)
        ans4 = my_retriever.retrieve("Unconditional", 1)
        assert(len(set(ans1 + ans2 + ans3 + ans4)) > 1)
        ans5 = my_retriever.retrieve("Unconditional directed_by Brent", 2, ordered_by_freq=True)
        assert(ans5 == [
            "Unconditional directed_by Brent McCorkle",
            "Unconditional written_by Brent McCorkle",
        ])
        # test input string that are not in facts
        ans6 = my_retriever.retrieve("not_in_facts", 1)
        assert(ans6 == [])
        # test input string that are partially not in facts
        ans7 = my_retriever.retrieve("directed_by Brent not_in_facts", 1, ordered_by_freq=True)
        assert(ans7 == ["Unconditional directed_by Brent McCorkle"])




if __name__ == '__main__':
    unittest.main()
