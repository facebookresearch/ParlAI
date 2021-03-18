#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from parlai.utils.data import DatatypeHelper
from .build import build
import os
import json


class _Abstract(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = DatatypeHelper.fold(opt['datatype'])
        self.dpath = os.path.join(opt["datapath"], "CCPE")
        if shared is None:
            build(opt)
        super().__init__(opt, shared)

    def _get_turns_of_speaker(self, utts, idx, speaker):
        result = {"speaker": speaker, "text": "", "segments": []}
        while idx < len(utts) and utts[idx]["speaker"] == speaker:
            if len(result["text"]) == 0:
                result["text"] = utts[idx]["text"]
            else:
                result["text"] += "\n" + utts[idx]["text"]
            result["segments"].extend(utts[idx].get("segments", []))
            idx += 1
        return result, idx

    def _load_data(self, fold):
        fpath = os.path.join(self.opt['datapath'], 'CCPE', 'ccpe.json')
        with PathManager.open(fpath, 'r') as infile:
            json_data = json.load(infile)

        # do a 80:10:10 train/valid/test split
        full_episode_count = len(json_data) // 10
        if fold == 'train':
            json_data = json_data[: full_episode_count * 8]
        elif fold == 'valid':
            json_data = json_data[full_episode_count * 8 : full_episode_count * 9]
        elif fold == 'test':
            json_data = json_data[full_episode_count * 9 :]

        episodes = []
        for conversation in json_data:
            episode = []  # Assume all turns are alternating with ASSISTANT first
            idx = 0
            utts = conversation["utterances"]
            if utts[0]["speaker"] != "ASSISTANT":
                # Make a dummy assistant turn and add a user turn
                episode.append({"speaker": "ASSISTANT", "text": "", "segments": []})
                turn, idx = self._get_turns_of_speaker(utts, idx, "USER")
                episode.append(turn)
            while idx < len(utts):
                turn, idx = self._get_turns_of_speaker(utts, idx, "ASSISTANT")
                episode.append(turn)
                turn, idx = self._get_turns_of_speaker(utts, idx, "USER")
                episode.append(turn)
            if len(episode) % 2 != 0:
                # Add a dummy user turn so we don't need to worry about labels
                episode.append({"speaker": "USER", "text": "", "segments": []})
            episodes.append(episode)
        return episodes


class CcpeAssistantTeacher(_Abstract):
    def setup_data(self, fold):
        episodes = self._load_data(fold)
        for episode in episodes:
            first = True
            for i in range(len(episode) // 2):
                assistant_turn = episode[2 * i]
                user_turn = episode[2 * i + 1]
                if len(assistant_turn["text"]) == 0:
                    continue  # trains fail otherwise
                yield {
                    "text": assistant_turn["text"],
                    "textSegments": assistant_turn["segments"],
                    "label": user_turn["text"],
                    "labelSegments": user_turn["segments"],
                }, first
                first = False


class CcpeUserTeacher(_Abstract):
    def setup_data(self, fold):
        episodes = self._load_data(fold)
        for episode in episodes:
            first = True
            for i in range((len(episode) // 2) - 1):
                user_turn = episode[2 * i + 1]
                assistant_turn = episode[2 * i + 2]
                if len(user_turn["text"]) == 0:
                    continue  # trains fail otherwise
                yield {
                    "text": user_turn["text"],
                    "textSegments": user_turn["segments"],
                    "label": assistant_turn["text"],
                    "labelSegments": assistant_turn["segments"],
                }, first
                first = False


class DefaultTeacher(CcpeAssistantTeacher):
    pass
