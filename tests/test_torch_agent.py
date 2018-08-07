# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import unittest
from functools import lru_cache
from parlai.core.agents import Agent
from parlai.core.torch_agent import TorchAgent, Batch, Output


class MockDict(Agent):
    """Mock Dictionary Agent which just implements indexing and txt2vec."""

    null_token = '__null__'
    NULL_IDX = 0
    start_token = '__start__'
    START_IDX = 1001
    end_token = '__end__'
    END_IDX = 1002
    p1_token = '__p1__'
    P1_IDX = 2001
    p2_token = '__p2__'
    P2_IDX = 2002

    def __init__(self, opt, shared=None):
        """Initialize idx for incremental indexing."""
        self.idx = 0

    def __getitem__(self, key):
        """Return index of special token or return the token."""
        if key == self.null_token:
            return self.NULL_IDX
        elif key == self.start_token:
            return self.START_IDX
        elif key == self.end_token:
            return self.END_IDX
        elif key == self.p1_token:
            return self.P1_IDX
        elif key == self.p2_token:
            return self.P2_IDX
        else:
            self.idx += 1
            return self.idx

    def txt2vec(self, txt):
        """Return index of special tokens or range from 1 for each token."""
        self.idx = 0
        return [self[tok] for tok in txt.split()]


class TorchAgent(TorchAgent):
    """Use MockDict instead of regular DictionaryAgent."""

    @staticmethod
    def dictionary_class():
        """Replace normal dictionary class with mock one."""
        return MockDict


@lru_cache(maxsize=32)
def get_agent(**kwargs):
    """Return opt-initialized agent.

    :param kwargs: any kwargs you want to set using parser.set_params(**kwargs)
    """
    if 'no_cuda' not in kwargs:
        kwargs['no_cuda'] = True
    from parlai.core.params import ParlaiParser
    parser = ParlaiParser()
    TorchAgent.add_cmdline_args(parser)
    parser.set_params(**kwargs)
    opt = parser.parse_args(print_args=False)
    return TorchAgent(opt)

class TestTorchAgent(unittest.TestCase):
    """Basic tests on the util functions in TorchAgent."""

    def test_mock(self):
        """Just make sure we can instantiate a mock agent."""
        agent = get_agent()
        self.assertTrue(isinstance(agent.dict, MockDict))

    def test_share(self):
        """Make sure share works and shares dictionary."""
        agent = get_agent()
        shared = agent.share()
        self.assertTrue('dict' in shared)

    def test__vectorize_text(self):
        """Test _vectorize_text and its different options."""
        agent = get_agent()
        text = 'hello there john'

        # test add_start and add_end
        vec = agent._vectorize_text(text, add_start=False, add_end=False)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(text, add_start=True, add_end=False)
        self.assertEqual(len(vec), 4)
        self.assertEqual(vec.tolist(), [MockDict.START_IDX, 1, 2, 3])
        vec = agent._vectorize_text(text, add_start=False, add_end=True)
        self.assertEqual(len(vec), 4)
        self.assertEqual(vec.tolist(), [1, 2, 3, MockDict.END_IDX])
        vec = agent._vectorize_text(text, add_start=True, add_end=True)
        self.assertEqual(len(vec), 5)
        self.assertEqual(vec.tolist(), [MockDict.START_IDX, 1, 2, 3,
                                        MockDict.END_IDX])

        # now do it again with truncation=3
        vec = agent._vectorize_text(text, add_start=False, add_end=False,
                                    truncate=3)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(text, add_start=True, add_end=False,
                                    truncate=3)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(text, add_start=False, add_end=True,
                                    truncate=3)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [2, 3, MockDict.END_IDX])
        vec = agent._vectorize_text(text, add_start=True, add_end=True,
                                    truncate=3)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [2, 3, MockDict.END_IDX])

        # now do it again with truncation=2
        vec = agent._vectorize_text(text, add_start=False, add_end=False,
                                    truncate=2)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [2, 3])
        vec = agent._vectorize_text(text, add_start=True, add_end=False,
                                    truncate=2)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [2, 3])
        vec = agent._vectorize_text(text, add_start=False, add_end=True,
                                    truncate=2)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [3, MockDict.END_IDX])
        vec = agent._vectorize_text(text, add_start=True, add_end=True,
                                    truncate=2)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [3, MockDict.END_IDX])

        # now do it again with truncation=2, don't truncate_left
        vec = agent._vectorize_text(text, add_start=False, add_end=False,
                                    truncate=2, truncate_left=False)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [1, 2])
        vec = agent._vectorize_text(text, add_start=True, add_end=False,
                                    truncate=2, truncate_left=False)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [MockDict.START_IDX, 1])
        vec = agent._vectorize_text(text, add_start=False, add_end=True,
                                    truncate=2, truncate_left=False)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [1, 2])
        vec = agent._vectorize_text(text, add_start=True, add_end=True,
                                    truncate=2, truncate_left=False)
        self.assertEqual(len(vec), 2)
        self.assertEqual(vec.tolist(), [MockDict.START_IDX, 1])

        # now do it again with truncation=3, don't truncate_left
        vec = agent._vectorize_text(text, add_start=False, add_end=False,
                                    truncate=3, truncate_left=False)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(text, add_start=True, add_end=False,
                                    truncate=3, truncate_left=False)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [MockDict.START_IDX, 1, 2])
        vec = agent._vectorize_text(text, add_start=False, add_end=True,
                                    truncate=3, truncate_left=False)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [1, 2, 3])
        vec = agent._vectorize_text(text, add_start=True, add_end=True,
                                    truncate=3, truncate_left=False)
        self.assertEqual(len(vec), 3)
        self.assertEqual(vec.tolist(), [MockDict.START_IDX, 1, 2])

    def test__check_truncate(self):
        """Make sure we are truncating when needed."""
        agent = get_agent()
        inp = torch.LongTensor([1, 2, 3])
        self.assertEqual(agent._check_truncate(inp, None).tolist(), [1, 2, 3])
        self.assertEqual(agent._check_truncate(inp, 3).tolist(), [1, 2, 3])
        self.assertEqual(agent._check_truncate(inp, 2).tolist(), [1, 2])
        self.assertEqual(agent._check_truncate(inp, 1).tolist(), [1])
        self.assertEqual(agent._check_truncate(inp, 0).tolist(), [])

    def test_vectorize(self):
        """
        """
        return

        from parlai.core.params import ParlaiParser
        parser = ParlaiParser()
        TorchAgent.add_cmdline_args(parser)
        parser.set_params(no_cuda=True)
        opt = parser.parse_args(print_args=False)
        mdict = MockDict()

        shared = {'opt': opt, 'dict': mdict}
        agent = TorchAgent(opt, shared)
        observation = {}
        observation["text"] = "What does the dog do?"
        observation["labels"] = ["The dog jumps over the cat."]

        # add start and end
        obs_vec = agent.vectorize(observation, add_start=True, add_end=True)
        self.assertTrue('text_vec' in obs_vec,
                        "Field 'text_vec' missing from vectorized observation")
        self.assertTrue(obs_vec['text_vec'].numpy().tolist() == [7, 8, 9],
                        "Vectorized text is incorrect.")
        self.assertTrue('labels_vec' in obs_vec,
                        "Field 'labels_vec' missing from vectorized observation")
        self.assertTrue(obs_vec['labels_vec'].numpy().tolist() ==
                        [mdict.START_IDX, 7, 8, 9, mdict.END_IDX],
                        "Vectorized label is incorrect.")
        # no start, add end
        obs_vec = agent.vectorize(observation, add_start=False, add_end=True)
        self.assertTrue(obs_vec['labels_vec'].numpy().tolist() ==
                        [7, 8, 9, mdict.END_IDX],
                        "Vectorized label is incorrect.")
        # add start, no end
        obs_vec = agent.vectorize(observation, add_start=True, add_end=False)
        self.assertTrue(obs_vec['labels_vec'].numpy().tolist() ==
                        [mdict.START_IDX, 7, 8, 9],
                        "Vectorized label is incorrect.")
        # no start, no end
        obs_vec = agent.vectorize(observation, add_start=False, add_end=False)
        self.assertTrue(obs_vec['labels_vec'].numpy().tolist() == [7, 8, 9],
                        "Vectorized label is incorrect.")

        observation = {}
        observation["text"] = "What does the dog do?"
        observation["eval_labels"] = ["The dog jumps over the cat."]

        # eval_labels
        obs_vec = agent.vectorize(observation)
        self.assertTrue('eval_labels_vec' in obs_vec,
                        "Field \'eval_labels_vec\' missing from vectorized observation")
        self.assertTrue(obs_vec['eval_labels_vec'].numpy().tolist() ==
                        [mdict.START_IDX, 7, 8, 9, mdict.END_IDX],
                        "Vectorized label is incorrect.")
        # truncate
        obs_vec = agent.vectorize(observation, truncate=2)
        self.assertTrue('eval_labels_vec' in obs_vec,
                        "Field \'eval_labels_vec\' missing from vectorized observation")
        self.assertTrue(obs_vec['eval_labels_vec'].numpy().tolist() ==
                        [mdict.START_IDX, 7],
                        "Vectorized label is incorrect: " + str(obs_vec['eval_labels_vec']))

        # truncate
        obs_vec = agent.vectorize(observation, truncate=10)
        self.assertTrue('eval_labels_vec' in obs_vec,
                        "Field \'eval_labels_vec\' missing from vectorized observation")
        self.assertTrue(obs_vec['eval_labels_vec'].numpy().tolist() ==
                        [mdict.START_IDX, 7, 8, 9, mdict.END_IDX],
                        "Vectorized label is incorrect.")

    def test_map_unmap(self):
        return
        try:
            from parlai.core.torch_agent import TorchAgent, Output
        except ImportError as e:
            if 'pytorch' in e.msg:
                print('Skipping TestTorchAgent.test_map_unmap, no pytorch.')
                return

        observations = []
        observations.append({"text": "What is a painting?",
                             "labels": ["Paint on a canvas."]})
        observations.append({})
        observations.append({})
        observations.append({"text": "What is a painting?",
                             "labels": ["Paint on a canvas."]})
        observations.append({})
        observations.append({})

        from parlai.core.params import ParlaiParser
        parser = ParlaiParser()
        TorchAgent.add_cmdline_args(parser)
        parser.set_params(no_cuda=True)
        opt = parser.parse_args(print_args=False)
        mdict = MockDict()

        shared = {'opt': opt, 'dict': mdict}
        agent = TorchAgent(opt, shared)

        vec_observations = [agent.vectorize(obs) for obs in observations]

        batch = agent.batchify(vec_observations)

        self.assertTrue(batch.text_vec is not None, "Missing 'text_vecs' field.")
        self.assertTrue(batch.text_vec.numpy().tolist() == [[7, 8, 9], [7, 8, 9]],
                        "Incorrectly vectorized text field of obs_batch.")
        self.assertTrue(batch.label_vec is not None, "Missing 'label_vec' field.")
        self.assertTrue(batch.label_vec.numpy().tolist() ==
                        [[mdict.START_IDX, 7, 8, 9, mdict.END_IDX],
                         [mdict.START_IDX, 7, 8, 9, mdict.END_IDX]],
                        "Incorrectly vectorized text field of obs_batch.")
        self.assertTrue(batch.labels == ["Paint on a canvas.", "Paint on a canvas."],
                        "Doesn't return correct labels: " + str(batch.labels))
        true_i = [0, 3]
        self.assertTrue(all(batch.valid_indices[i] == true_i[i] for i in range(2)),
                        "Returns incorrect indices of valid observations.")

        observations = []
        observations.append({"text": "What is a painting?",
                             "eval_labels": ["Paint on a canvas."]})
        observations.append({})
        observations.append({})
        observations.append({"text": "What is a painting?",
                             "eval_labels": ["Paint on a canvas."]})
        observations.append({})
        observations.append({})

        vec_observations = [agent.vectorize(obs) for obs in observations]

        batch = agent.batchify(vec_observations)

        self.assertTrue(batch.label_vec is not None, "Missing \'eval_label_vec\' field.")
        self.assertTrue(batch.label_vec.numpy().tolist() ==
                        [[mdict.START_IDX, 7, 8, 9, mdict.END_IDX],
                         [mdict.START_IDX, 7, 8, 9, mdict.END_IDX]],
                        "Incorrectly vectorized text field of obs_batch.")

        batch_reply = [{} for i in range(6)]
        predictions = ["Oil on a canvas.", "Oil on a canvas."]
        output = Output(predictions, None)
        expected_unmapped = batch_reply.copy()
        expected_unmapped[0]["text"] = "Oil on a canvas."
        expected_unmapped[3]["text"] = "Oil on a canvas."
        self.assertTrue(agent.match_batch(batch_reply, batch.valid_indices, output) == expected_unmapped,
                        "Unmapped predictions do not match expected results.")

    def test_maintain_dialog_history(self):
        return
        try:
            from parlai.core.torch_agent import TorchAgent
        except ImportError as e:
            if 'pytorch' in e.msg:
                print('Skipping TestTorchAgent.test_maintain_dialog_history, no pytorch.')
                return

        from parlai.core.params import ParlaiParser
        parser = ParlaiParser()
        TorchAgent.add_cmdline_args(parser)
        parser.set_params(no_cuda=True, truncate=5)
        opt = parser.parse_args(print_args=False)
        mdict = MockDict()

        shared = {'opt': opt, 'dict': mdict}
        agent = TorchAgent(opt, shared)

        observation = {"text": "What is a painting?",
                       "labels": ["Paint on a canvas."],
                       "episode_done": False}

        agent.maintain_dialog_history(observation)

        self.assertTrue('dialog' in agent.history, "Failed initializing self.history.")
        self.assertTrue('episode_done' in agent.history, "Failed initializing self.history.")
        self.assertTrue('labels' in agent.history, "Failed initializing self.history.")
        self.assertTrue(list(agent.history['dialog']) == [7, 8, 9],
                        "Failed adding vectorized text to dialog.")
        self.assertTrue(not agent.history['episode_done'],
                        "Failed to properly store episode_done field.")
        self.assertTrue(agent.history['labels'] == observation['labels'],
                        "Failed saving labels.")

        observation['text_vec'] = agent.maintain_dialog_history(observation)
        print(agent.history['dialog'])
        self.assertTrue(list(agent.history['dialog']) == [8, 9, 7, 8, 9],
                        "Failed adding vectorized text to dialog.")


if __name__ == '__main__':
    try:
        import torch
        unittest.main()
    except ImportError as e:
        print('Skipping TestTorchAgent, no pytorch.')
