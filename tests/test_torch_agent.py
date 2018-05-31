# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_agent import TorchAgent


import os
import unittest


class MockDict(object):
    null_token = 0
    start_token = 1
    end_token = 2

    def __getitem__(self, key):
        return key

    def txt2vec(self, txt):
        return [1, 3, 5]

class TestTorchAgent(unittest.TestCase):
    """Basic tests on the util functions in TorchAgent."""

    def test_vectorize(self):
        opt = {}
        opt['no_cuda'] = True
        opt['history_tokens'] = 10000
        opt['history_dialog'] = 10
        opt['history_replies'] = 'label_else_model'
        dict = MockDict()

        shared = {'opt': opt, 'dict': dict}
        agent = TorchAgent(opt, shared)
        observation = {}
        observation["text"] = "What does the dog do?"
        observation["labels"] = ["The dog jumps over the cat."]

        obs_vec = agent.vectorize(observation)

        self.assertTrue('text_vec' in obs_vec,
                        "Field \'text_vec\' missing from vectorized observation")
        self.assertTrue(obs_vec['text_vec'].numpy().tolist() == [1, 3, 5],
                        "Vectorized text is incorrect.")
        self.assertTrue('labels_vec' in obs_vec,
                        "Field \'labels_vec\' missing from vectorized observation")
        self.assertTrue(obs_vec['labels_vec'][0].numpy().tolist() == [1, 3, 5, dict.end_token],
                        "Vectorized label is incorrect.")

        observation = {}
        observation["text"] = "What does the dog do?"
        observation["eval_labels"] = ["The dog jumps over the cat."]

        obs_vec = agent.vectorize(observation)

        self.assertTrue('eval_labels_vec' in obs_vec,
                        "Field \'eval_labels_vec\' missing from vectorized observation")
        self.assertTrue(obs_vec['eval_labels_vec'][0].numpy().tolist() == [1, 3, 5, dict.end_token],
                        "Vectorized label is incorrect.")

    def test_map_unmap(self):
        observations = []
        observations.append({"text": "What is a painting?", "labels": ["Paint on a canvas."]})
        observations.append({})
        observations.append({})
        observations.append({"text": "What is a painting?", "labels": ["Paint on a canvas."]})
        observations.append({})
        observations.append({})

        opt = {}
        opt['no_cuda'] = True
        opt['history_tokens'] = 10000
        opt['history_dialog'] = 10
        opt['history_replies'] = 'label_else_model'
        dict = MockDict()

        shared = {'opt': opt, 'dict': dict}
        agent = TorchAgent(opt, shared)

        vec_observations = [agent.vectorize(obs) for obs in observations]

        mapped_valid = agent.map_valid(vec_observations)

        text_vecs, label_vecs, labels, valid_inds = mapped_valid

        self.assertTrue(text_vecs is not None, "Missing \'text_vecs\' field.")
        self.assertTrue(text_vecs.numpy().tolist() == [[1, 3, 5], [1, 3, 5]],
                        "Incorrectly vectorized text field of obs_batch.")
        self.assertTrue(label_vecs is not None, "Missing \'label_vec\' field.")
        self.assertTrue(label_vecs.numpy().tolist() == [[1, 3, 5, 2], [1, 3, 5, 2]],
                        "Incorrectly vectorized text field of obs_batch.")
        self.assertTrue(labels == ["Paint on a canvas.", "Paint on a canvas."],
                        "Doesn't return correct labels.")
        self.assertTrue(valid_inds == [0,3],
                        "Returns incorrect indices of valid observations.")

        observations = []
        observations.append({"text": "What is a painting?", "eval_labels": ["Paint on a canvas."]})
        observations.append({})
        observations.append({})
        observations.append({"text": "What is a painting?", "eval_labels": ["Paint on a canvas."]})
        observations.append({})
        observations.append({})

        vec_observations = [agent.vectorize(obs) for obs in observations]

        mapped_valid = agent.map_valid(vec_observations)

        text_vecs, label_vecs, labels, valid_inds = mapped_valid

        self.assertTrue(label_vecs is not None, "Missing \'label_vec\' field.")
        self.assertTrue(label_vecs.numpy().tolist() == [[1, 3, 5, 2], [1, 3, 5, 2]],
                        "Incorrectly vectorized text field of obs_batch.")

        predictions = ["Oil on a canvas.", "Oil on a canvas."]
        expected_unmapped = ["Oil on a canvas.", None, None, "Oil on a canvas.", None, None]
        self.assertTrue(agent.unmap_valid(predictions, valid_inds, 6) == expected_unmapped,
                        "Unmapped predictions do not match expected results.")

    def test_maintain_dialog_history(self):
        MAX_DEQUE_LEN = 8
        opt = {}
        opt['no_cuda'] = True
        opt['history_tokens'] = MAX_DEQUE_LEN
        opt['history_dialog'] = 10
        opt['history_replies'] = 'label_else_model'
        dict = MockDict()

        shared = {'opt': opt, 'dict': dict}
        agent = TorchAgent(opt, shared)

        observation = {"text": "What is a painting?",
                       "labels": ["Paint on a canvas."],
                       "episode_done": False}

        agent.maintain_dialog_history(observation)

        self.assertTrue('dialog' in agent.history, "Failed initializing self.history.")
        self.assertTrue('episode_done' in agent.history, "Failed initializing self.history.")
        self.assertTrue('labels' in agent.history, "Failed initializing self.history.")
        self.assertTrue(list(agent.history['dialog']) == [1, 1, 3, 5, 2],
                        "Failed adding vectorized text to dialog.")
        self.assertTrue(not agent.history['episode_done'],
                        "Failed to properly store episode_done field.")
        self.assertTrue(agent.history['labels'] == observation['labels'],
                        "Failed saving labels.")

        observation['text_vec'] = agent.maintain_dialog_history(observation)
        self.assertTrue(list(agent.history['dialog']) == [3, 5, 2, 1, 1, 3, 5, 2],
                        "Failed adding vectorized text to dialog.")



if __name__ == '__main__':
    unittest.main()
