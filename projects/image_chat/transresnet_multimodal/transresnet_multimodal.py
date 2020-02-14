#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transresnet Multimodal Model (https://arxiv.org/abs/1811.00945).
"""

from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from parlai.core.message import Message
from .modules import TransresnetMultimodalModel
from projects.personality_captions.transresnet.transresnet import TransresnetAgent

import torch
from torch import optim
import random
import os
import numpy as np
import tqdm
from collections import deque


class TransresnetMultimodalAgent(TransresnetAgent):
    """
    Model from "Engaging Image Chat: Modeling Personality in Grounded Dialogue".

    See paper for more details: (https://arxiv.org/abs/1811.00945)

    An extension of the model from https://arxiv.org/abs/1810.10665; given
    an image, personality, and dialogue history, predicts the next utterance
    in a dialogue.
    """

    ######################################
    # Initialization and argument parsers
    ######################################

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Override to add personality-override option.
        """
        TransresnetMultimodalModel.add_cmdline_args(argparser)
        TransresnetAgent.add_cmdline_args(argparser)
        arg_group = argparser.add_argument_group("TransresnetMultimodal Arguments")
        argparser.add_argument(
            "--personality-override",
            type=str,
            default=None,
            help="for use in other tasks where no personality "
            "is given. This will give the model a personality "
            "(whichever is specifed).",
        )
        argparser.add_argument(
            "--personalities-path",
            type=str,
            default=None,
            help="Path to personalities list",
        )
        DictionaryAgent.add_cmdline_args(argparser)
        return arg_group

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.metrics = {
            k: {"hits@1/100": 0.0, "loss": 0.0, "num_samples": 0, "med_rank": []}
            for k in ["first_round", "second_round", "third_round+"]
        }
        if shared is None:
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.opt["learningrate"],
            )
        else:
            self.optimizer = shared["optimizer"]

        self.history = deque(maxlen=None)
        self.personality_override = opt.get("personality_override")

    def _build_model(self, path=None):
        init_model_path = None
        if self.opt.get("init_model") and os.path.isfile(self.opt["init_model"]):
            init_model_path = self.opt["init_model"]
        elif self.opt.get("model_file") and os.path.isfile(self.opt["model_file"]):
            init_model_path = self.opt["model_file"]
        elif path is not None:
            init_model_path = path
        print("Creating or loading model")
        self.model = TransresnetMultimodalModel(
            self.opt, self.personalities_list, self.dict
        )
        if init_model_path is not None:
            self.load(init_model_path)
        if self.use_cuda:
            self.model.cuda()

    def _setup_cands(self):
        """
        Override for different call to model.
        """
        self.fixed_cands = None
        self.fixed_cands_enc = None
        if self.fcp is not None:
            with open(self.fcp) as f:
                self.fixed_cands = [c.replace("\n", "") for c in f.readlines()]
            cands_enc_file = "{}.cands_enc".format(self.fcp)
            print("loading saved cand encodings")
            if os.path.isfile(cands_enc_file):
                self.fixed_cands_enc = torch.load(
                    cands_enc_file, map_location=lambda cpu, _: cpu
                )
            else:
                print("Extracting cand encodings")
                self.model.eval()
                pbar = tqdm.tqdm(
                    total=len(self.fixed_cands),
                    unit="cand",
                    unit_scale=True,
                    desc="Extracting candidate encodings",
                )
                fixed_cands_enc = []
                for _, batch in enumerate(
                    [
                        self.fixed_cands[i : i + 50]
                        for i in range(0, len(self.fixed_cands) - 50, 50)
                    ]
                ):
                    embedding = self.model.forward_text_encoder(batch).detach()
                    fixed_cands_enc.append(embedding)
                    pbar.update(50)
                self.fixed_cands_enc = torch.cat(fixed_cands_enc, 0)
                torch.save(self.fixed_cands_enc, cands_enc_file)

    def share(self):
        """
        Override to share optimizer.
        """
        shared = super().share()
        shared["optimizer"] = self.optimizer
        return shared

    def observe(self, observation):
        """
        Observe an observation.

        Additionally retrieves the dialogue history for the observation.

        :param observation:
            observation

        :return:
            the observation, with dialogue history included.
        """
        observation = Message(observation)  # TODO: eventually this will not be
        # necessary as we migrate all teachers to return Message objects
        self.observation = self.get_dialogue_history(observation)
        return self.observation

    def train_step(self, valid_obs, image_feats, personalities, dialogue_histories):
        """
        Model train step.

        :param valid_obs:
            list of valid observations

        :param image_feats:
            list of image features, one per example

        :param personalities:
            list of personalities, one per example

        :param dialogue_histories:
            list of dialogue histories, one per example

        :return:
            the total loss and the number of correct examples
        """
        self.model.train()
        labels = [random.choice(v["labels"]) for v in valid_obs]
        loss, num_correct, _ = self.model(
            image_feats,
            personalities,
            dialogue_histories,
            labels,
            batchsize=len(valid_obs),
        )
        return loss, num_correct

    def eval_step(self, valid_obs, image_feats, personalities, dialogue_histories):
        """
        Model eval step.

        :param valid_obs:
            list of valid observations

        :param image_feats:
            list of image features, one per example

        :param personalities:
            list of personalities, one per example

        :param dialogue_histories:
            list of dialogue histories, one per example

        :return:
            the total loss, number of correct examples,
            the ranked position of each correct caption,
            and the ranked lists of candidates (one per example)
        """
        self.model.eval()
        med_rank = None
        chosen_responses = None
        candidates_encoded = None

        if self.fixed_cands is not None:
            candidates_encoded = self.fixed_cands_enc
            candidates = self.fixed_cands
        else:
            candidates = [v["label_candidates"] for v in valid_obs]
        chosen_responses = self.model.choose_best_response(
            image_feats,
            personalities,
            dialogue_histories,
            candidates,
            candidates_encoded=candidates_encoded,
            k=-1 if self.fixed_cands is None else 100,
            batchsize=len(valid_obs),
        )
        loss = -1
        if self.fixed_cands is not None:
            num_correct = 0
        else:
            labels = [v.get("eval_labels") for v in valid_obs]
            equality_list = [
                1 if chosen_responses[i][0] in labels[i] else 0
                for i in range(len(labels))
            ]
            # calculate med ranks
            med_rank = []
            for i, e_list in enumerate(chosen_responses):
                lowest_rank = len(e_list) + 1
                for c in labels[i]:
                    lowest_rank = min(lowest_rank, e_list.index(c) + 1)
                med_rank.append(lowest_rank)
            num_correct = sum(equality_list)

        return loss, num_correct, med_rank, chosen_responses

    def batch_act(self, observations):
        """
        Act on a batch of observations.

        :param observations:
            list of observations

        :return:
            A list of acts, one for each observation
        """
        is_training = any(["labels" in obs for obs in observations])
        valid_obs, valid_indexes = self.filter_valid_obs(observations, is_training)
        image_feats = self.extract_image_feats(valid_obs)
        personalities, dialogue_histories, dialogue_round = self.extract_texts(
            valid_obs
        )
        chosen_responses = None
        med_rank = None
        if is_training:
            self.optimizer.zero_grad()
            loss, num_correct = self.train_step(
                valid_obs, image_feats, personalities, dialogue_histories
            )
            loss.backward()
            self.optimizer.step()
        else:
            loss, num_correct, med_rank, chosen_responses = self.eval_step(
                valid_obs, image_feats, personalities, dialogue_histories
            )
        self.update_metrics(loss, num_correct, len(valid_obs), dialogue_round, med_rank)
        result = [
            {"text": "No Response During Traiing", "id": self.getID()}
            for _ in range(len(observations))
        ]
        if chosen_responses is not None:
            for i, index_obs in enumerate(valid_indexes):
                result[index_obs]["text"] = chosen_responses[i][0]
                result[index_obs]["text_candidates"] = chosen_responses[i]
        return result

    def extract_texts(self, obs):
        """
        Extract the personalities and dialogue histories from observations.

        Additionally determine which dialogue round we are in.

        Note that this function assumes that the personality is the
        last line of the `text` field in the observation.

        :param obs:
            list of observations

        :return:
            a list of personalities, a list of dialogue histories, and the
            current dialogue round (either first, second, or third+)
        """
        splits = [v.get("text").split("\n") for v in obs]
        if self.personality_override:
            splits = [s + [self.personality_override] for s in splits]
        personalities = [t[-1] for t in splits]
        dialogue_histories = None
        dialogue_round = "first_round"
        if len(splits[0]) >= 2:
            dialogue_round = "second_round" if len(splits[0]) == 2 else "third_round+"
            dialogue_histories = ["\n".join(t[:-1]) for t in splits]

        return personalities, dialogue_histories, dialogue_round

    def get_dialogue_history(self, obs):
        """
        Get dialogue history for an observation.

        :param obs:
            observation

        :return:
            the observation with the dialogue history in the `text` field
        """
        if len(self.history) > 0:
            obs.force_set("text", "\n".join(self.history) + "\n" + obs["text"])
        if "labels" in obs:
            self.history.append(random.choice(obs["labels"]))
        elif "eval_labels" in obs:
            self.history.append(random.choice(obs["eval_labels"]))
        if obs.get("episode_done", True):
            # end of this episode, clear the history
            self.history.clear()

        return obs

    def update_metrics(
        self, loss, num_correct, num_samples, dialogue_round, med_rank=None
    ):
        """
        Update Metrics.

        Overriden to include dialogue round

        :param loss:
            float loss
        :param num_correct:
            number of examples for which chosen caption is correct
        :param num_samples:
            total number of examples
        :param med_rank:
            rank of correct caption for each example
        """
        self.metrics[dialogue_round]["hits@1/100"] += num_correct
        self.metrics[dialogue_round]["loss"] += loss
        self.metrics[dialogue_round]["num_samples"] += num_samples
        if med_rank:
            self.metrics[dialogue_round]["med_rank"] += med_rank

    def receive_metrics(self, metrics_dict):
        """
        Receive the metrics from validation.

        Unfreeze text encoder weights after a certain number of rounds without improvement.

        Override to account for different dialogue rounds.

        :param metrics_dict:
            the metrics dictionary
        """
        if "tasks" in metrics_dict:
            metrics_dict = metrics_dict["tasks"]["internal:comment_battle:imageDialog"]
        if self.freeze_patience != -1 and self.is_frozen:
            m_key = "hits@1/100"
            ms = [
                metrics_dict[r].get(m_key, -1)
                for r in ["first_round", "second_round", "third_round+"]
            ]
            m = sum(ms) / len([m for m in ms if m >= 0])
            if m > self.freeze_best_metric:
                self.freeze_impatience = 0
                self.freeze_best_metric = m
                print("performance not good enough to unfreeze the model.")
            else:
                self.freeze_impatience += 1
                print("Growing impatience for unfreezing")
                if self.freeze_impatience >= self.freeze_patience:
                    self.is_frozen = False
                    print(
                        "Reached impatience for fine tuning. "
                        "Reloading the best model so far."
                    )
                    self._build_model(self.model_file)
                    if self.use_cuda:
                        self.model = self.model.cuda()
                    print("Unfreezing.")
                    self.model.unfreeze_text_encoder()
                    print("Done")

    def reset(self):
        """
        Override to reset dialogue history.
        """
        super().reset()
        self.history.clear()

    def reset_metrics(self):
        """
        Reset per-dialogue round metrics.
        """
        for v in self.metrics.values():
            v["hits@1/100"] = 0.0
            v["loss"] = 0.0
            v["num_samples"] = 0.0
            if "med_rank" in v:
                v["med_rank"] = []

    def report(self):
        """
        Report per-dialogue round metrics.
        """
        m = {}
        for k, v in self.metrics.items():
            if "num_samples" not in v:
                print(self.metrics)
                print(k)
                __import__("ipdb").set_trace()  # FIXME
            if v["num_samples"] > 0:
                m[f"{k}/hits@1/100"] = round_sigfigs(
                    v["hits@1/100"] / v["num_samples"], 4
                )
                m[f"{k}/loss"] = round_sigfigs(v["loss"] / v["num_samples"], 4)
                if "med_rank" in v:
                    m[f"{k}/med_rank"] = np.median(v["med_rank"])
        return m
