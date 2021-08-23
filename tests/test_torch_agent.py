#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for TorchAgent.
"""

import os
import unittest
from parlai.core.agents import create_agent_from_shared
from parlai.utils.testing import tempdir
from parlai.utils.misc import Message

SKIP_TESTS = False
try:
    from parlai.core.torch_agent import Output
    from parlai.agents.test_agents.test_agents import MockTorchAgent, MockDict
    import torch
except ImportError:
    SKIP_TESTS = True


def get_agent(**kwargs):
    r"""
    Return opt-initialized agent.

    :param kwargs: any kwargs you want to set using parser.set_params(\*\*kwargs)
    """
    if 'no_cuda' not in kwargs:
        kwargs['no_cuda'] = True
    from parlai.core.params import ParlaiParser

    parser = ParlaiParser()
    MockTorchAgent.add_cmdline_args(parser, partial_opt=None)
    parser.set_params(**kwargs)
    opt = parser.parse_args([])
    return MockTorchAgent(opt)


@unittest.skipIf(SKIP_TESTS, "Torch not installed.")
class TestTorchAgent(unittest.TestCase):
    """
    Basic tests on the util functions in TorchAgent.
    """

    def test_mock(self):
        """
        Just make sure we can instantiate a mock agent.
        """
        agent = get_agent()
        self.assertTrue(isinstance(agent.dict, MockDict))

    def test_share(self):
        """
        Make sure share works and shares dictionary.
        """
        agent = get_agent()
        shared = agent.share()
        self.assertTrue('dict' in shared)

    def test__vectorize_text(self):
        """
        Test _vectorize_text and its different options.
        """
        agent = get_agent()
        text = "I'm sorry, Dave"

        # test add_start and add_end
        vec = agent._vectorize_text(text, add_start=False, add_end=False)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(text, add_start=True, add_end=False)
        self.assertEqual(len(vec), 4)
        self.assertEqual(vec.tolist(), [MockDict.BEG_IDX, 1, 2, 3])
        vec = agent._vectorize_text(text, add_start=False, add_end=True)
        self.assertEqual(len(vec), 4)
        self.assertEqual(vec.tolist(), [1, 2, 3, MockDict.END_IDX])
        vec = agent._vectorize_text(text, add_start=True, add_end=True)
        self.assertEqual(len(vec), 5)
        self.assertEqual(vec.tolist(), [MockDict.BEG_IDX, 1, 2, 3, MockDict.END_IDX])

        # now do it again with truncation=3
        vec = agent._vectorize_text(text, add_start=False, add_end=False, truncate=3)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(text, add_start=True, add_end=False, truncate=3)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(text, add_start=False, add_end=True, truncate=3)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [2, 3, MockDict.END_IDX])
        vec = agent._vectorize_text(text, add_start=True, add_end=True, truncate=3)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [2, 3, MockDict.END_IDX])

        # now do it again with truncation=2
        vec = agent._vectorize_text(text, add_start=False, add_end=False, truncate=2)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [2, 3])
        vec = agent._vectorize_text(text, add_start=True, add_end=False, truncate=2)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [2, 3])
        vec = agent._vectorize_text(text, add_start=False, add_end=True, truncate=2)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [3, MockDict.END_IDX])
        vec = agent._vectorize_text(text, add_start=True, add_end=True, truncate=2)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [3, MockDict.END_IDX])

        # now do it again with truncation=2, don't truncate_left
        vec = agent._vectorize_text(
            text, add_start=False, add_end=False, truncate=2, truncate_left=False
        )
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [1, 2])
        vec = agent._vectorize_text(
            text, add_start=True, add_end=False, truncate=2, truncate_left=False
        )
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [MockDict.BEG_IDX, 1])
        vec = agent._vectorize_text(
            text, add_start=False, add_end=True, truncate=2, truncate_left=False
        )
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [1, 2])
        vec = agent._vectorize_text(
            text, add_start=True, add_end=True, truncate=2, truncate_left=False
        )
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [MockDict.BEG_IDX, 1])

        # now do it again with truncation=3, don't truncate_left
        vec = agent._vectorize_text(
            text, add_start=False, add_end=False, truncate=3, truncate_left=False
        )
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(
            text, add_start=True, add_end=False, truncate=3, truncate_left=False
        )
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [MockDict.BEG_IDX, 1, 2])
        vec = agent._vectorize_text(
            text, add_start=False, add_end=True, truncate=3, truncate_left=False
        )
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(
            text, add_start=True, add_end=True, truncate=3, truncate_left=False
        )
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [MockDict.BEG_IDX, 1, 2])

    def test__check_truncate(self):
        """
        Make sure we are truncating when needed.
        """
        agent = get_agent()
        inp = torch.LongTensor([1, 2, 3])
        self.assertEqual(agent._check_truncate(inp, None).tolist(), [1, 2, 3])
        self.assertEqual(agent._check_truncate(inp, 4).tolist(), [1, 2, 3])
        self.assertEqual(agent._check_truncate(inp, 3).tolist(), [1, 2, 3])
        self.assertEqual(agent._check_truncate(inp, 2).tolist(), [1, 2])
        self.assertEqual(agent._check_truncate(inp, 1).tolist(), [1])
        self.assertEqual(agent._check_truncate(inp, 0).tolist(), [])

    def test_vectorize(self):
        """
        Test the vectorization of observations.

        Make sure they do not recompute results, and respect the different param
        options.
        """
        agent = get_agent()
        obs_labs = Message(
            {'text': 'No. Try not.', 'labels': ['Do.', 'Do not.'], 'episode_done': True}
        )
        obs_elabs = Message(
            {
                'text': 'No. Try not.',
                'eval_labels': ['Do.', 'Do not.'],
                'episode_done': True,
            }
        )

        for obs in (obs_labs, obs_elabs):
            lab_key = 'labels' if 'labels' in obs else 'eval_labels'
            lab_vec = lab_key + '_vec'
            lab_chc = lab_key + '_choice'

            inp = obs.copy()
            # test add_start=True, add_end=True
            agent.history.reset()
            agent.history.update_history(inp)
            out = agent.vectorize(inp, agent.history, add_start=True, add_end=True)
            self.assertEqual(out['text_vec'].tolist(), [1, 2, 3])
            # note that label could be either label above
            self.assertEqual(out[lab_vec][0].item(), MockDict.BEG_IDX)
            self.assertEqual(out[lab_vec][1].item(), 1)
            self.assertEqual(out[lab_vec][-1].item(), MockDict.END_IDX)
            self.assertEqual(out[lab_chc][:2], 'Do')

            # test add_start=True, add_end=False
            inp = obs.copy()
            out = agent.vectorize(inp, agent.history, add_start=True, add_end=False)
            self.assertEqual(out['text_vec'].tolist(), [1, 2, 3])
            # note that label could be either label above
            self.assertEqual(out[lab_vec][0].item(), MockDict.BEG_IDX)
            self.assertNotEqual(out[lab_vec][-1].item(), MockDict.END_IDX)
            self.assertEqual(out[lab_chc][:2], 'Do')

            # test add_start=False, add_end=True
            inp = obs.copy()
            out = agent.vectorize(inp, agent.history, add_start=False, add_end=True)
            self.assertEqual(out['text_vec'].tolist(), [1, 2, 3])
            # note that label could be either label above
            self.assertNotEqual(out[lab_vec][0].item(), MockDict.BEG_IDX)
            self.assertEqual(out[lab_vec][-1].item(), MockDict.END_IDX)
            self.assertEqual(out[lab_chc][:2], 'Do')

            # test add_start=False, add_end=False
            inp = obs.copy()
            out = agent.vectorize(inp, agent.history, add_start=False, add_end=False)
            self.assertEqual(out['text_vec'].tolist(), [1, 2, 3])
            # note that label could be either label above
            self.assertNotEqual(out[lab_vec][0].item(), MockDict.BEG_IDX)
            self.assertNotEqual(out[lab_vec][-1].item(), MockDict.END_IDX)
            self.assertEqual(out[lab_chc][:2], 'Do')

            # test caching of tensors
            out_again = agent.vectorize(out, agent.history)
            # should have cached result from before
            self.assertIs(out['text_vec'], out_again['text_vec'])
            self.assertEqual(out['text_vec'].tolist(), [1, 2, 3])
            # next: should truncate cached result
            prev_vec = out['text_vec']
            out_again = agent.vectorize(out, agent.history, text_truncate=1)
            self.assertIsNot(prev_vec, out_again['text_vec'])
            self.assertEqual(out['text_vec'].tolist(), [3])

        # test split_lines
        agent = get_agent(split_lines=True)
        obs = Message(
            {
                'text': 'Hello.\nMy name is Inogo Montoya.\n'
                'You killed my father.\nPrepare to die.',
                'episode_done': True,
            }
        )
        agent.history.update_history(obs)
        vecs = agent.history.get_history_vec_list()
        self.assertEqual(vecs, [[1], [1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3]])

        # check cache
        out_again = agent.vectorize(obs, agent.history)
        vecs = agent.history.get_history_vec_list()
        self.assertEqual(vecs, [[1], [1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3]])

    def test_batchify(self):
        """
        Make sure the batchify function sets up the right fields.
        """
        agent = get_agent(rank_candidates=True)
        obs_labs = [
            Message(
                {
                    'text': 'It\'s only a flesh wound.',
                    'labels': ['Yield!'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'The needs of the many outweigh...',
                    'labels': ['The needs of the few.'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'Hello there.',
                    'labels': ['General Kenobi.'],
                    'episode_done': True,
                }
            ),
        ]
        obs_elabs = [
            Message(
                {
                    'text': 'It\'s only a flesh wound.',
                    'eval_labels': ['Yield!'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'The needs of the many outweigh...',
                    'eval_labels': ['The needs of the few.'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'Hello there.',
                    'eval_labels': ['General Kenobi.'],
                    'episode_done': True,
                }
            ),
        ]
        for obs_batch in (obs_labs, obs_elabs):
            lab_key = 'labels' if 'labels' in obs_batch[0] else 'eval_labels'

            # nothing has been vectorized yet so should be empty
            batch = agent.batchify(obs_batch)
            self.assertIsNone(batch.text_vec)
            self.assertIsNone(batch.label_vec)
            self.assertIsNone(batch.labels)
            self.assertIsNone(batch.candidates)
            self.assertIsNone(batch.candidate_vecs)
            self.assertIsNone(batch.image)

            obs_vecs = []
            for o in obs_batch:
                agent.history.reset()
                agent.history.update_history(o)
                obs_vecs.append(
                    agent.vectorize(o, agent.history, add_start=False, add_end=False)
                )

            # is_valid should map to nothing
            def is_valid(obs):
                return False

            agent.is_valid = is_valid

            batch = agent.batchify(obs_batch)
            self.assertIsNone(batch.text_vec)
            self.assertIsNone(batch.label_vec)
            self.assertIsNone(batch.labels)
            self.assertIsNone(batch.candidates)
            self.assertIsNone(batch.candidate_vecs)
            self.assertIsNone(batch.image)

            # is_valid should check for text_vec
            def is_valid(obs):
                return 'text_vec' in obs

            agent.is_valid = is_valid

            batch = agent.batchify(obs_vecs)
            # which fields were filled vs should be empty?
            self.assertIsNotNone(batch.text_vec)
            self.assertIsNotNone(batch.label_vec)
            self.assertIsNotNone(batch.labels)
            self.assertIsNone(batch.candidates)
            self.assertIsNone(batch.candidate_vecs)
            self.assertIsNone(batch.image)

            # contents of certain fields:
            self.assertEqual(
                batch.text_vec.tolist(),
                [[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6], [1, 2, 0, 0, 0, 0]],
            )
            self.assertEqual(
                batch.label_vec.tolist(),
                [[1, 0, 0, 0, 0], [1, 2, 3, 4, 5], [1, 2, 0, 0, 0]],
            )
            self.assertEqual(batch.labels, [o[lab_key][0] for o in obs_batch])
            self.assertEqual(list(batch.valid_indices), [0, 1, 2])

            # now sort the batch, make sure fields are in sorted order
            batch = agent.batchify(obs_vecs, sort=True)
            self.assertEqual(
                batch.text_vec.tolist(),
                [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0], [1, 2, 0, 0, 0, 0]],
            )
            self.assertEqual(
                batch.label_vec.tolist(),
                [[1, 2, 3, 4, 5], [1, 0, 0, 0, 0], [1, 2, 0, 0, 0]],
            )
            labs = [o[lab_key][0] for o in obs_batch]
            self.assertEqual(batch.labels, [labs[i] for i in [1, 0, 2]])
            self.assertEqual(list(batch.valid_indices), [1, 0, 2])

            # now sort just on ys
            new_vecs = [vecs.copy() for vecs in obs_vecs]
            for vec in new_vecs:
                vec.pop('text')
                vec.pop('text_vec')

            def is_valid(obs):
                return 'labels_vec' in obs or 'eval_labels_vec' in obs

            agent.is_valid = is_valid

            batch = agent.batchify(new_vecs, sort=True)
            self.assertIsNone(batch.text_vec)
            self.assertIsNotNone(batch.label_vec)
            self.assertEqual(
                batch.label_vec.tolist(),
                [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0], [1, 0, 0, 0, 0]],
            )
            labs = [o[lab_key][0] for o in new_vecs]
            self.assertEqual(batch.labels, [labs[i] for i in [1, 2, 0]])
            self.assertEqual(list(batch.valid_indices), [1, 2, 0])

            # test is_valid
            def is_valid(obs):
                return 'text_vec' in obs and len(obs['text_vec']) < 3

            agent.is_valid = is_valid

            batch = agent.batchify(obs_vecs)
            self.assertEqual(batch.text_vec.tolist(), [[1, 2]])
            self.assertEqual(batch.label_vec.tolist(), [[1, 2]])
            self.assertEqual(batch.labels, obs_batch[2][lab_key])
            self.assertEqual(list(batch.valid_indices), [2])

        agent.history.reset()
        obs_cands = [
            agent.vectorize(
                Message({'label_candidates': ['A', 'B', 'C']}), agent.history
            ),
            agent.vectorize(
                Message({'label_candidates': ['1', '2', '5', '3', 'Sir']}),
                agent.history,
            ),
            agent.vectorize(
                Message({'label_candidates': ['Do', 'Re', 'Mi']}), agent.history
            ),
            agent.vectorize(
                Message({'label_candidates': ['Fa', 'So', 'La', 'Ti']}), agent.history
            ),
        ]

        # is_valid should check for label candidates vecs
        def is_valid(obs):
            return 'label_candidates_vecs' in obs

        agent.is_valid = is_valid

        batch = agent.batchify(obs_cands)
        self.assertTrue(agent.rank_candidates, 'Agent not set up to rank.')
        self.assertIsNone(batch.text_vec)
        self.assertIsNone(batch.label_vec)
        self.assertIsNone(batch.labels)
        self.assertIsNotNone(batch.valid_indices)
        self.assertIsNotNone(batch.candidates)
        self.assertIsNotNone(batch.candidate_vecs)
        self.assertEqual(list(batch.valid_indices), [0, 1, 2, 3])
        self.assertEqual(batch.candidates, [o['label_candidates'] for o in obs_cands])
        self.assertEqual(len(batch.candidate_vecs), len(obs_cands))
        for i, cs in enumerate(batch.candidate_vecs):
            self.assertEqual(len(cs), len(obs_cands[i]['label_candidates']))

    def test_match_batch(self):
        """
        Make sure predictions are correctly aligned when available.
        """
        agent = get_agent()

        # first try empty outputs
        reply = agent.match_batch([{}, {}, {}], [0, 1, 2], Output())
        self.assertEqual([{}, {}, {}], reply)
        reply = agent.match_batch([{}, {}, {}], [0, 1, 2], None)
        self.assertEqual([{}, {}, {}], reply)

        # try text in order
        reply = agent.match_batch(
            [{}, {}, {}], [0, 1, 2], Output(['E.T.', 'Phone', 'Home'])
        )
        self.assertEqual([{'text': 'E.T.'}, {'text': 'Phone'}, {'text': 'Home'}], reply)

        # try text out of order
        reply = agent.match_batch(
            [{}, {}, {}], [2, 0, 1], Output(['Home', 'E.T.', 'Phone'])
        )
        self.assertEqual([{'text': 'E.T.'}, {'text': 'Phone'}, {'text': 'Home'}], reply)

        # try text_candidates in order
        reply = agent.match_batch(
            [{}, {}],
            [0, 1],
            Output(
                None,
                [
                    ['More human than human.', 'Less human than human'],
                    ['Just walk into Mordor', 'Just QWOP into Mordor.'],
                ],
            ),
        )
        self.assertEqual(
            reply[0]['text_candidates'],
            ['More human than human.', 'Less human than human'],
        )
        self.assertEqual(
            reply[1]['text_candidates'],
            ['Just walk into Mordor', 'Just QWOP into Mordor.'],
        )
        # try text_candidates out of order
        reply = agent.match_batch(
            [{}, {}],
            [1, 0],
            Output(
                None,
                [
                    ['More human than human.', 'Less human than human'],
                    ['Just walk into Mordor', 'Just QWOP into Mordor.'],
                ],
            ),
        )
        self.assertEqual(
            reply[0]['text_candidates'],
            ['Just walk into Mordor', 'Just QWOP into Mordor.'],
        )
        self.assertEqual(
            reply[1]['text_candidates'],
            ['More human than human.', 'Less human than human'],
        )

        # try both text and text_candidates in order
        reply = agent.match_batch(
            [{}, {}],
            [0, 1],
            Output(
                ['You shall be avenged...', 'Man creates dinosaurs...'],
                [
                    ['By Grabthar’s hammer.', 'By the suns of Worvan.'],
                    ['Dinosaurs eat man.', 'Woman inherits the earth.'],
                ],
            ),
        )
        self.assertEqual(reply[0]['text'], 'You shall be avenged...')
        self.assertEqual(
            reply[0]['text_candidates'],
            ['By Grabthar’s hammer.', 'By the suns of Worvan.'],
        )
        self.assertEqual(reply[1]['text'], 'Man creates dinosaurs...')
        self.assertEqual(
            reply[1]['text_candidates'],
            ['Dinosaurs eat man.', 'Woman inherits the earth.'],
        )

        # try both text and text_candidates out of order
        reply = agent.match_batch(
            [{}, {}],
            [1, 0],
            Output(
                ['You shall be avenged...', 'Man creates dinosaurs...'],
                [
                    ['By Grabthar’s hammer.', 'By the suns of Worvan.'],
                    ['Dinosaurs eat man.', 'Woman inherits the earth.'],
                ],
            ),
        )
        self.assertEqual(reply[0]['text'], 'Man creates dinosaurs...')
        self.assertEqual(
            reply[0]['text_candidates'],
            ['Dinosaurs eat man.', 'Woman inherits the earth.'],
        )
        self.assertEqual(reply[1]['text'], 'You shall be avenged...')
        self.assertEqual(
            reply[1]['text_candidates'],
            ['By Grabthar’s hammer.', 'By the suns of Worvan.'],
        )

    def test__add_person_tokens(self):
        """
        Make sure person tokens are added to the write place in text.
        """
        agent = get_agent()
        text = (
            "I've seen things you people wouldn't believe.\n"
            "Attack ships on fire off the shoulder of Orion.\n"
            "I watched C-beams glitter in the dark near the Tannhauser gate.\n"
            "All those moments will be lost in time, like tears in rain."
        )
        prefix = 'PRE'
        out = agent.history._add_person_tokens(text, prefix, add_after_newln=False)
        self.assertEqual(out, prefix + ' ' + text)
        out = agent.history._add_person_tokens(text, prefix, add_after_newln=True)
        idx = text.rfind('\n') + 1
        self.assertEqual(out, text[:idx] + prefix + ' ' + text[idx:])

    def test_history(self):
        """
        Test different dialog history settings.
        """
        # try with unlimited history
        agent = get_agent(history_size=-1)
        obs = {'text': 'I am Groot.', 'labels': ['I am Groot?'], 'episode_done': False}

        # first exchange
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')

        # second exchange, no reply
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.\nI am Groot.')

        # include reply
        agent.history.add_reply('I am Groot?')
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.\nI am Groot.\nI am Groot?\nI am Groot.')

        # on reset should be same as first exchange
        agent.history.reset()
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')

        # now try with history size = 1
        agent = get_agent(history_size=1)
        # first exchange
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')
        # second exchange should change nothing
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')
        # third exchange with reply should change nothing
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')
        # now if we add the reply, we should only have the reply left
        agent.history.add_reply(obs['labels'][0])
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot?')

        # now try with history size = 2
        agent = get_agent(history_size=2)

        # first exchange
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')

        # second exchange with reply should contain reply
        agent.history.add_reply('I am Groot?')
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot?\nI am Groot.')

        # now try with history size = 3
        agent = get_agent(history_size=3)
        # first exchange
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')
        # second exchange with reply should contain reply and input
        agent.history.add_reply('I am Groot?')
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.\nI am Groot?\nI am Groot.')

        # now test add_person_tokens
        agent = get_agent(history_size=3, person_tokens=True)
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, '{} I am Groot.'.format(agent.P1_TOKEN))
        # second exchange, history should still contain the tokens
        agent.history.add_reply('I am Groot?')
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(
            text,
            '{} I am Groot.\n{} I am Groot?\n{} I am Groot.'.format(
                agent.P1_TOKEN, agent.P2_TOKEN, agent.P1_TOKEN
            ),
        )

        # now add add_p1_after_newln
        agent = get_agent(history_size=3, person_tokens=True, add_p1_after_newln=True)
        ctx_obs = obs.copy()  # context then utterance in this text field
        ctx_obs['text'] = 'Groot is Groot.\nI am Groot.'
        agent.history.update_history(ctx_obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'Groot is Groot.\n{} I am Groot.'.format(agent.P1_TOKEN))
        # second exchange, history should still contain context text
        agent.history.add_reply('I am Groot?')
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(
            text,
            'Groot is Groot.\n{} I am Groot.\n{} I am Groot?\n{} I am Groot.'.format(
                agent.P1_TOKEN, agent.P2_TOKEN, agent.P1_TOKEN
            ),
        )

        # test history vecs
        agent.history.reset()
        agent.history.update_history(obs)
        vec = agent.history.get_history_vec()
        self.assertEqual(vec, [2001, 1, 2, 3])

        # test history vec list
        agent.history.update_history(obs)
        vecs = agent.history.get_history_vec_list()
        self.assertEqual(vecs, [[2001, 1, 2, 3], [2001, 1, 2, 3]])

        # test clearing history
        agent.history.reset()
        text = agent.history.get_history_str()
        self.assertIsNone(text)
        vecs = agent.history.get_history_vec_list()
        self.assertEqual(vecs, [])

        # test delimiter
        agent = get_agent(history_size=-1, delimiter=' Groot! ')
        agent.history.update_history(obs)
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot. Groot! I am Groot.')

        # test global_end_token, this will append a selected token to the end
        # of history block
        agent = get_agent(history_add_global_end_token='end')
        agent.history.reset()
        agent.history.update_history(obs)
        vec = agent.history.get_history_vec()
        self.assertEqual(vec, [1, 2, 3, MockDict.END_IDX])

        # test temp history
        agent = get_agent(
            history_size=-1, include_temp_history=True, delimiter='__delim__'
        )
        agent.history.reset()
        agent.history.update_history(obs, temp_history=' temp history')
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot. temp history')
        vec = agent.history.get_history_vec()
        self.assertEqual(vec, [1, 2, 3, 1, 2])

        agent.history.update_history(obs, temp_history=' temp history')
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.__delim__I am Groot. temp history')
        vecs = agent.history.get_history_vec_list()
        self.assertEqual(vecs, [[1, 2, 3], [1, 2, 3]])
        vec = agent.history.get_history_vec()
        self.assertEqual(vec, [1, 2, 3, 1, 1, 2, 3, 1, 2])

    def test_reversed_history(self):
        agent = get_agent(history_reversed=True)
        agent.history.reset()
        agent.history.update_history({'text': 'hello i am stephen'})
        agent.history.update_history({'text': 'i am bob'})
        assert agent.history.get_history_str() == 'hello i am stephen\ni am bob'
        agent.history.reset()
        agent.history.update_history(
            {'text': 'your persona: filler\nhello i am stephen'}
        )
        agent.history.update_history({'text': 'i am bob'})
        assert (
            agent.history.get_history_str()
            == 'your persona: filler\nhello i am stephen\ni am bob'
        )

    def test_temp_history_observe(self):
        """
        Test temp_history when provided via a field in the observation.
        """
        agent = get_agent(dict_file='zoo:unittest/transformer_generator2/model.dict')
        obs = agent.observe(
            {'text': '1 2 3', 'temp_history': '4 5 6', 'episode_done': False}
        )
        assert len(obs['text_vec']) == 6
        assert obs['full_text'] == '1 2 34 5 6'
        obs = agent.observe(
            {'text': '1 2 3', 'temp_history': '6', 'episode_done': False}
        )
        assert len(obs['text_vec']) == 7
        assert obs['full_text'] == '1 2 3\n1 2 36'
        obs = agent.observe({'text': '1 2 3', 'episode_done': False})
        assert len(obs['text_vec']) == 9
        assert obs['full_text'] == '1 2 3\n1 2 3\n1 2 3'

        # make sure temp history is forgotten after a reset
        obs = agent.observe(
            {'text': '1 2 3', 'temp_history': '4', 'episode_done': True}
        )
        assert len(obs['text_vec']) == 13
        assert obs['full_text'] == '1 2 3\n1 2 3\n1 2 3\n1 2 34'
        agent.act()  # get that self-observe in

        obs = agent.observe({'text': '1 2 3', 'episode_done': True})
        assert len(obs['text_vec']) == 3
        assert obs['full_text'] == '1 2 3'
        agent.act()  # get that self-observe in

    def test_observe(self):
        """
        Make sure agent stores and returns observation.
        """
        agent = get_agent()
        # text could be none
        obs = {'text': None, 'episode_done': True}
        out = agent.observe(obs.copy())
        self.assertIsNotNone(out)
        # make sure we throw an exception for having an episode done without a reset
        obs = {'text': "I'll be back.", 'labels': ["I'm back."], 'episode_done': True}
        with self.assertRaises(RuntimeError):
            agent.observe(obs.copy())
        # okay, let's do it properly now
        agent.reset()
        obs = {'text': "I'll be back.", 'labels': ["I'm back."], 'episode_done': True}
        out = agent.observe(obs.copy())
        self.assertIsNotNone(out)
        self.assertIsNotNone(agent.observation)
        self.assertEqual(out['text'], "I'll be back.")
        # now try with episode not done
        agent = get_agent()
        obs['episode_done'] = False
        out = agent.observe(obs.copy())
        self.assertIsNotNone(out)
        self.assertIsNotNone(agent.observation)
        self.assertEqual(out['text'], "I'll be back.")
        # should remember history
        agent.act()
        out = agent.observe(obs.copy())
        self.assertEqual(out['full_text'], "I'll be back.\nI'm back.\nI'll be back.")

    def test_batch_act(self):
        """
        Make sure batch act calls the right step.
        """
        agent = get_agent()

        obs_labs = [
            Message(
                {
                    'text': "It's only a flesh wound.",
                    'labels': ['Yield!'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'The needs of the many outweigh...',
                    'labels': ['The needs of the few.'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'Hello there.',
                    'labels': ['General Kenobi.'],
                    'episode_done': True,
                }
            ),
        ]
        obs_labs_vecs = []
        for o in obs_labs:
            agent.history.reset()
            agent.history.update_history(o)
            obs_labs_vecs.append(agent.vectorize(o, agent.history))
        reply = agent.batch_act(obs_labs_vecs)
        for i in range(len(obs_labs_vecs)):
            self.assertEqual(reply[i]['text'], 'Training {}!'.format(i))

        obs_elabs = [
            Message(
                {
                    'text': "It's only a flesh wound.",
                    'eval_labels': ['Yield!'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'The needs of the many outweigh...',
                    'eval_labels': ['The needs of the few.'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'Hello there.',
                    'eval_labels': ['General Kenobi.'],
                    'episode_done': True,
                }
            ),
        ]
        obs_elabs_vecs = []
        for o in obs_elabs:
            agent.history.reset()
            agent.history.update_history(o)
            obs_elabs_vecs.append(agent.vectorize(o, agent.history))
        reply = agent.batch_act(obs_elabs_vecs)
        for i in range(len(obs_elabs_vecs)):
            self.assertIn('Evaluating {}'.format(i), reply[i]['text'])

    def test_respond(self):
        """
        Tests respond() in the base Agent class, where the agent provides
        a string response to a single message.
        """
        agent = get_agent()
        message = Message(
            {
                'text': "It's only a flesh wound.",
                'labels': ['Yield!'],
                'episode_done': True,
            }
        )
        response = agent.respond(message)
        self.assertEqual(response, 'Training 0!')
        message = Message(
            {
                'text': "It's only a flesh wound.",
                'eval_labels': ['Yield!'],
                'episode_done': True,
            }
        )
        response = agent.respond(message)
        self.assertIn('Evaluating 0', response)

    def test_batch_respond(self):
        """
        Tests batch_respond() in the base Agent class, where the agent provides
        a batch response to a batch of messages.
        """
        agent = get_agent()

        obs_labs = [
            Message(
                {
                    'text': "It's only a flesh wound.",
                    'labels': ['Yield!'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'The needs of the many outweigh...',
                    'labels': ['The needs of the few.'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'Hello there.',
                    'labels': ['General Kenobi.'],
                    'episode_done': True,
                }
            ),
        ]
        response = agent.batch_respond(obs_labs)
        for i, resp in enumerate(response):
            self.assertEqual(resp, 'Training {}!'.format(i))

        obs_elabs = [
            Message(
                {
                    'text': "It's only a flesh wound.",
                    'eval_labels': ['Yield!'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'The needs of the many outweigh...',
                    'eval_labels': ['The needs of the few.'],
                    'episode_done': True,
                }
            ),
            Message(
                {
                    'text': 'Hello there.',
                    'eval_labels': ['General Kenobi.'],
                    'episode_done': True,
                }
            ),
        ]
        response = agent.batch_respond(obs_elabs)
        for i, resp in enumerate(response):
            self.assertIn('Evaluating {}'.format(i), resp)

    def test_interactive_mode(self):
        """
        Test if conversation history is destroyed in MTurk mode.
        """
        # both manually setting bs to 1 and interactive mode true
        agent = get_agent(batchsize=1, interactive_mode=True)
        agent.observe(Message({'text': 'foo', 'episode_done': True}))
        response = agent.act()
        self.assertIn(
            'Evaluating 0', response['text'], 'Incorrect output in single act()'
        )
        shared = create_agent_from_shared(agent.share())
        shared.observe(Message({'text': 'bar', 'episode_done': True}))
        response = shared.act()
        self.assertIn(
            'Evaluating 0', response['text'], 'Incorrect output in single act()'
        )

        # now just bs 1
        agent = get_agent(batchsize=1, interactive_mode=False)
        agent.observe(Message({'text': 'foo', 'episode_done': True}))
        response = agent.act()
        self.assertIn(
            'Evaluating 0', response['text'], 'Incorrect output in single act()'
        )
        shared = create_agent_from_shared(agent.share())
        shared.observe(Message({'text': 'bar', 'episode_done': True}))
        response = shared.act()
        self.assertIn(
            'Evaluating 0', response['text'], 'Incorrect output in single act()'
        )

        # now just interactive
        shared = create_agent_from_shared(agent.share())
        agent.observe(Message({'text': 'foo', 'episode_done': True}))
        response = agent.act()
        self.assertIn(
            'Evaluating 0', response['text'], 'Incorrect output in single act()'
        )
        shared = create_agent_from_shared(agent.share())
        shared.observe(Message({'text': 'bar', 'episode_done': True}))
        response = shared.act()
        self.assertIn(
            'Evaluating 0', response['text'], 'Incorrect output in single act()'
        )

        # finally, actively attempt to sabotage
        agent = get_agent(batchsize=16, interactive_mode=False)
        agent.observe(Message({'text': 'foo', 'episode_done': True}))
        response = agent.act()
        self.assertIn(
            'Evaluating 0', response['text'], 'Incorrect output in single act()'
        )
        shared = create_agent_from_shared(agent.share())
        shared.observe(Message({'text': 'bar', 'episode_done': True}))
        response = shared.act()
        self.assertIn(
            'Evaluating 0', response['text'], 'Incorrect output in single act()'
        )

    def test_use_reply(self):
        """
        Check that self-observe is correctly acting on labels.
        """
        # default is hybrid label-model, which uses the label if it's available, and
        # otherwise the label
        # first check if there is a label available
        agent = get_agent()
        obs = Message({'text': 'Call', 'labels': ['Response'], 'episode_done': False})
        agent.observe(obs)
        _ = agent.act()
        self.assertEqual(agent.history.get_history_str(), 'Call\nResponse')
        # check if there is no label
        agent.reset()
        obs = Message({'text': 'Call', 'episode_done': False})
        agent.observe(obs)
        _ = agent.act()
        self.assertEqual(
            agent.history.get_history_str(), 'Call\nEvaluating 0 (responding to [[1]])!'
        )
        # now some of the other possible values of --use-reply
        # --use-reply model. even if there is a label, we should see the model's out
        agent = get_agent(use_reply='model')
        obs = Message({'text': 'Call', 'labels': ['Response'], 'episode_done': False})
        agent.observe(obs)
        _ = agent.act()
        self.assertEqual(agent.history.get_history_str(), 'Call\nTraining 0!')
        # --use-reply none doesn't hear itself
        agent = get_agent(use_reply='none')
        obs = Message({'text': 'Call', 'labels': ['Response'], 'episode_done': False})
        agent.observe(obs)
        agent.act()
        self.assertEqual(agent.history.get_history_str(), 'Call')

    def test_mturk_racehistory(self):
        """
        Emulate a setting where batch_act misappropriately handles mturk.
        """
        agent = get_agent(batchsize=16, interactive_mode=True, echo=True)
        share1 = create_agent_from_shared(agent.share())

        share1.observe(Message({'text': 'thread1-msg1', 'episode_done': False}))
        share2 = create_agent_from_shared(agent.share())
        share2.observe(Message({'text': 'thread2-msg1', 'episode_done': False}))
        share1.act()
        share2.act()

        share1.observe(Message({'text': 'thread1-msg2', 'episode_done': False}))
        share2.observe(Message({'text': 'thread2-msg2', 'episode_done': False}))
        share2.act()
        share1.act()

        share2.observe(Message({'text': 'thread2-msg3', 'episode_done': False}))
        share1.observe(Message({'text': 'thread1-msg3', 'episode_done': False}))

        self.assertNotIn('thread1-msg1', share2.history.get_history_str())
        self.assertNotIn('thread2-msg1', share1.history.get_history_str())
        self.assertNotIn('thread1-msg2', share2.history.get_history_str())
        self.assertNotIn('thread2-msg2', share1.history.get_history_str())

    def test_resume_checkpoint(self):
        """
        Make sure when resuming training that model uses appropriate mf.

        Copy train_model from testing_utils to directly access agent.
        """
        import parlai.scripts.train_model as tms

        def get_popt_and_tl(opt):
            parser = tms.setup_args()
            parser.set_params(**opt)
            popt = parser.parse_args([])
            for k, v in opt.items():
                popt[k] = v
            return popt, tms.TrainLoop(popt)

        def get_opt(init_mf, mf):
            return {
                'task': 'integration_tests',
                'init_model': init_mf,
                'model': 'parlai.agents.test_agents.test_agents:MockTorchAgent',
                'model_file': mf,
                'num_epochs': 3,
                'validation_every_n_epochs': 1,
                'save_after_valid': True,
                'log_every_n_secs': 10,
            }

        with tempdir() as tmpdir:
            # First train model with init_model path set
            mf = os.path.join(tmpdir, 'model')
            init_mf = os.path.join(tmpdir, 'init_model')
            with open(init_mf, 'w') as f:
                f.write(' ')
            opt = get_opt(init_mf, mf)
            popt, tl = get_popt_and_tl(opt)
            agent = tl.agent
            # init model file should be set appropriately
            init_model_file, is_finetune = agent._get_init_model(popt, None)
            self.assertEqual(init_model_file, init_mf)
            self.assertTrue(is_finetune)
            valid, test = tl.train()
            # now, train the model for another epoch
            opt = get_opt('{}.checkpoint'.format(mf), mf)
            opt['load_from_checkpoint'] = True
            popt, tl = get_popt_and_tl(opt)
            agent = tl.agent
            init_model_file, is_finetune = agent._get_init_model(popt, None)
            self.assertEqual(init_model_file, '{}.checkpoint'.format(mf))
            self.assertFalse(is_finetune)

    def test_truncate_metrics(self):
        agent = get_agent(model='test_agents/unigram', truncate=5)
        obs = {
            'text': "I'll be back. I'll be back. I'll be back.",
            'labels': ["I'll be back. I'll be back. I'll be back."],
            'episode_done': True,
        }
        obs = agent.observe(obs)
        agent.act()
        self.assertEqual(agent._local_metrics['ctrunc'][0].value(), 1.0)
        self.assertEqual(agent._local_metrics['ltrunc'][0].value(), 1.0)
        self.assertEqual(agent._local_metrics['clen'][0].value(), 9)
        self.assertEqual(agent._local_metrics['llen'][0].value(), 11)
        self.assertEqual(agent._local_metrics['ctrunclen'][0].value(), 4)
        self.assertEqual(agent._local_metrics['ltrunclen'][0].value(), 6)
