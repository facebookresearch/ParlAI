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
from parlai.utils.testing import capture_output, tempdir
from parlai.utils.misc import Message
import parlai.utils.testing as testing_utils

from collections import deque

SKIP_TESTS = False
try:
    from parlai.core.torch_agent import Output
    from parlai.agents.test_agents.dummy_torch_agent import MockTorchAgent, MockDict
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
    MockTorchAgent.add_cmdline_args(parser)
    parser.set_params(**kwargs)
    opt = parser.parse_args(print_args=False)
    with testing_utils.capture_output():
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
            self.assertIsNone(batch.text_lengths)
            self.assertIsNone(batch.label_vec)
            self.assertIsNone(batch.label_lengths)
            self.assertIsNone(batch.labels)
            self.assertIsNone(batch.valid_indices)
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
            self.assertIsNone(batch.text_lengths)
            self.assertIsNone(batch.label_vec)
            self.assertIsNone(batch.label_lengths)
            self.assertIsNone(batch.labels)
            self.assertIsNone(batch.valid_indices)
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
            self.assertIsNotNone(batch.text_lengths)
            self.assertIsNotNone(batch.label_vec)
            self.assertIsNotNone(batch.label_lengths)
            self.assertIsNotNone(batch.labels)
            self.assertIsNotNone(batch.valid_indices)
            self.assertIsNone(batch.candidates)
            self.assertIsNone(batch.candidate_vecs)
            self.assertIsNone(batch.image)

            # contents of certain fields:
            self.assertEqual(
                batch.text_vec.tolist(),
                [[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6], [1, 2, 0, 0, 0, 0]],
            )
            self.assertEqual(batch.text_lengths, [5, 6, 2])
            self.assertEqual(
                batch.label_vec.tolist(),
                [[1, 0, 0, 0, 0], [1, 2, 3, 4, 5], [1, 2, 0, 0, 0]],
            )
            self.assertEqual(batch.label_lengths, [1, 5, 2])
            self.assertEqual(batch.labels, [o[lab_key][0] for o in obs_batch])
            self.assertEqual(list(batch.valid_indices), [0, 1, 2])

            # now sort the batch, make sure fields are in sorted order
            batch = agent.batchify(obs_vecs, sort=True)
            self.assertEqual(
                batch.text_vec.tolist(),
                [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0], [1, 2, 0, 0, 0, 0]],
            )
            self.assertEqual(batch.text_lengths, [6, 5, 2])
            self.assertEqual(
                batch.label_vec.tolist(),
                [[1, 2, 3, 4, 5], [1, 0, 0, 0, 0], [1, 2, 0, 0, 0]],
            )
            self.assertEqual(batch.label_lengths, [5, 1, 2])
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
            self.assertIsNone(batch.text_lengths)
            self.assertIsNotNone(batch.label_vec)
            self.assertIsNotNone(batch.label_lengths)
            self.assertEqual(
                batch.label_vec.tolist(),
                [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0], [1, 0, 0, 0, 0]],
            )
            self.assertEqual(batch.label_lengths, [5, 2, 1])
            labs = [o[lab_key][0] for o in new_vecs]
            self.assertEqual(batch.labels, [labs[i] for i in [1, 2, 0]])
            self.assertEqual(list(batch.valid_indices), [1, 2, 0])

            # test is_valid
            def is_valid(obs):
                return 'text_vec' in obs and len(obs['text_vec']) < 3

            agent.is_valid = is_valid

            batch = agent.batchify(obs_vecs)
            self.assertEqual(batch.text_vec.tolist(), [[1, 2]])
            self.assertEqual(batch.text_lengths, [2])
            self.assertEqual(batch.label_vec.tolist(), [[1, 2]])
            self.assertEqual(batch.label_lengths, [2])
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
        self.assertIsNone(batch.text_lengths)
        self.assertIsNone(batch.label_vec)
        self.assertIsNone(batch.label_lengths)
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

        # include reply and set episode_done to clear history after this one
        end_obs = obs.copy()
        end_obs['episode_done'] = True
        agent.history.update_history(end_obs, add_next='I am Groot?')
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.\nI am Groot.\nI am Groot?\nI am Groot.')

        # because of episode_done, should be same as first exchange
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

        # now try with history size = 2
        agent = get_agent(history_size=2)

        # first exchange
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')

        # second exchange with reply should contain reply
        agent.history.update_history(obs, add_next='I am Groot?')
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot?\nI am Groot.')

        # third exchange without reply should have two inputs
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.\nI am Groot.')

        # now try with history size = 3
        agent = get_agent(history_size=3)

        # first exchange
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.')

        # second exchange with reply should contain reply and input
        agent.history.update_history(obs, add_next='I am Groot?')
        text = agent.history.get_history_str()
        self.assertEqual(text, 'I am Groot.\nI am Groot?\nI am Groot.')

        # now test add_person_tokens
        agent = get_agent(history_size=3, person_tokens=True)
        agent.history.update_history(obs)
        text = agent.history.get_history_str()
        self.assertEqual(text, '{} I am Groot.'.format(agent.P1_TOKEN))

        # second exchange, history should still contain the tokens
        agent.history.update_history(obs, add_next='I am Groot?')
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
        agent.history.update_history(obs, add_next='I am Groot?')
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
        self.assertEqual(vec, deque([2001, 1, 2, 3]))

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

    def test_last_reply(self):
        """
        Make sure last reply returns expected values.
        """
        agent = get_agent()
        # nothing to retrieve
        self.assertIsNone(agent.last_reply())
        # set agent's generated replies
        agent.replies = {
            'batch_reply': [{'text': 'It\'s okay! I\'m a leaf on the wind.'}]
        }
        # If the observation was previously an episode end, we shouldn't have any
        # older reply
        self.assertEqual(agent.last_reply(), None)
        # now agent should remember what it said
        agent.observation = {'episode_done': False}
        self.assertEqual(agent.last_reply(), 'It\'s okay! I\'m a leaf on the wind.')
        # now set true observation
        agent.observation = {
            'text': 'Will that work?',
            'labels': ['I\'m a leaf on the wind. Watch how I soar.'],
            'episode_done': False,
        }
        # now agent should remember true label
        self.assertEqual(
            agent.last_reply(), 'I\'m a leaf on the wind. Watch how I soar.'
        )
        # but not if we tell to use the model reply
        self.assertEqual(
            agent.last_reply(use_reply='model'), 'It\'s okay! I\'m a leaf on the wind.'
        )
        # if we don't want to use the last reply at all, it should be None
        self.assertIsNone(agent.last_reply(use_reply='none'))

    def test_observe(self):
        """
        Make sure agent stores and returns observation.
        """
        agent = get_agent()
        # text could be none
        obs = {'text': None, 'episode_done': True}
        out = agent.observe(obs.copy())
        self.assertIsNotNone(out)
        obs = {'text': "I'll be back.", 'labels': ["I'm back."], 'episode_done': True}
        out = agent.observe(obs.copy())
        self.assertIsNotNone(out)
        self.assertIsNotNone(agent.observation)
        self.assertEqual(out['text'], "I'll be back.")
        # episode was done so shouldn't remember history
        out = agent.observe(obs.copy())
        self.assertEqual(out['text'], "I'll be back.")
        self.assertTrue('text_vec' in out, 'Text should be vectorized.')

        # now try with episode not done
        obs['episode_done'] = False
        out = agent.observe(obs.copy())
        self.assertIsNotNone(out)
        self.assertIsNotNone(agent.observation)
        self.assertEqual(out['text'], "I'll be back.")
        # should remember history
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

        # finally, the expected failure
        with self.assertRaises(RuntimeError):
            agent = get_agent(batchsize=16, interactive_mode=False)
            agent.observe(Message({'text': 'foo', 'episode_done': True}))
            response = agent.act()
            shared = create_agent_from_shared(agent.share())
            shared.observe(Message({'text': 'bar', 'episode_done': True}))
            response = shared.act()

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
            popt = parser.parse_args(print_args=False)
            for k, v in opt.items():
                popt[k] = v
            return popt, tms.TrainLoop(popt)

        def get_opt(init_mf, mf):
            return {
                'task': 'integration_tests',
                'init_model': init_mf,
                'model': 'parlai.agents.test_agents.dummy_torch_agent:MockTorchAgent',
                'model_file': mf,
                'num_epochs': 3,
                'validation_every_n_epochs': 1,
                'save_after_valid': True,
                'log_every_n_secs': 10,
            }

        with capture_output():
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


class TestLegacyVersioning(unittest.TestCase):
    def test_legacy_version(self):
        # simply tries to load and run some models with versioning attached
        with capture_output():
            with self.assertRaises(RuntimeError):
                testing_utils.display_model(
                    {
                        'model_file': 'models:convai2/seq2seq/convai2_self_seq2seq_model',
                        'task': 'convai2',
                        'no_cuda': True,
                    }
                )

            testing_utils.display_model(
                {
                    'model': 'legacy:seq2seq:0',
                    'model_file': 'models:convai2/seq2seq/convai2_self_seq2seq_model',
                    'task': 'convai2',
                    'no_cuda': True,
                }
            )


if __name__ == '__main__':
    unittest.main()
