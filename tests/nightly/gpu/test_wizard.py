#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import create_agent
import unittest
import parlai.scripts.display_data as display_data
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
import parlai.utils.testing as testing_utils

from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
    KnowledgeRetrieverAgent,
)

END2END_OPTIONS = {
    'task': 'wizard_of_wikipedia:generator:random_split',
    'model_file': 'zoo:wizard_of_wikipedia/end2end_generator/model',
    'batchsize': 32,
    'log_every_n_secs': 30,
    'embedding_type': 'random',
}


RETRIEVAL_OPTIONS = {
    'task': 'wizard_of_wikipedia',
    'model': 'projects:wizard_of_wikipedia:wizard_transformer_ranker',
    'model_file': 'zoo:wizard_of_wikipedia/full_dialogue_retrieval_model/model',
    'datatype': 'test',
    'n_heads': 6,
    'ffn_size': 1200,
    'embeddings_scale': False,
    'delimiter': ' __SOC__ ',
    'n_positions': 1000,
    'legacy': True,
}


@testing_utils.skipUnlessGPU
class TestWizardModel(unittest.TestCase):
    """
    Checks that pre-trained Wizard models give the correct results.
    """

    @classmethod
    def setUpClass(cls):
        # go ahead and download things here
        with testing_utils.capture_output():
            parser = display_data.setup_args()
            parser.set_defaults(**END2END_OPTIONS)
            opt = parser.parse_args(print_args=False)
            opt['num_examples'] = 1
            display_data.display_data(opt)

    def test_end2end(self):
        stdout, valid, _ = testing_utils.eval_model(END2END_OPTIONS)
        self.assertEqual(
            valid['ppl'], 61.21, 'valid ppl = {}\nLOG:\n{}'.format(valid['ppl'], stdout)
        )
        self.assertEqual(
            valid['f1'], 0.1717, 'valid f1 = {}\nLOG:\n{}'.format(valid['f1'], stdout)
        )
        self.assertGreaterEqual(
            valid['know_acc'],
            0.2201,
            'valid know_acc = {}\nLOG:\n{}'.format(valid['know_acc'], stdout),
        )

    def test_retrieval(self):
        stdout, _, test = testing_utils.eval_model(RETRIEVAL_OPTIONS)
        self.assertGreaterEqual(
            test['accuracy'],
            0.86,
            'test acc = {}\nLOG:\n{}'.format(test['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            test['hits@5'],
            0.98,
            'test hits@5 = {}\nLOG:\n{}'.format(test['hits@5'], stdout),
        )
        self.assertGreaterEqual(
            test['hits@10'],
            0.99,
            'test hits@10 = {}\nLOG:\n{}'.format(test['hits@10'], stdout),
        )


class TestKnowledgeRetriever(unittest.TestCase):
    """
    Checks that the Knowledge Retriever module returns the correct results.
    """

    def test_knowledge_retriever(self):
        from parlai.core.params import ParlaiParser

        parser = ParlaiParser(False, False)
        KnowledgeRetrieverAgent.add_cmdline_args(parser)
        parser.set_params(
            model='projects:wizard_of_wikipedia:knowledge_retriever',
            add_token_knowledge=True,
        )
        knowledge_opt = parser.parse_args([], print_args=False)
        knowledge_agent = create_agent(knowledge_opt)

        knowledge_agent.observe(
            {
                'text': 'what do you think of mountain dew?',
                'chosen_topic': 'Mountain Dew',
                'episode_done': False,
            }
        )

        knowledge_act = knowledge_agent.act()

        title = knowledge_act['title']
        self.assertEqual(title, 'Mountain Dew', 'Did not save chosen topic correctly')

        knowledge = knowledge_act['text']
        self.assertIn(
            TOKEN_KNOWLEDGE, knowledge, 'Knowledge token was not inserted correctly'
        )

        checked_sentence = knowledge_act['checked_sentence']
        self.assertEqual(
            checked_sentence,
            'Mountain Dew (stylized as Mtn Dew) is a carbonated soft drink brand produced and owned by PepsiCo.',
            'Did not correctly choose the checked sentence',
        )


if __name__ == '__main__':
    unittest.main()
