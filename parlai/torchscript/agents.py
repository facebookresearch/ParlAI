#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.utils.io import PathManager


class TorchScriptAgent(Agent):
    """
    ParlAI agent exported to TorchScript with JIT compilation and then loaded from disk.

    Metrics and batch act are currently unsupported, and CUDA is unsupported because
    TorchScripting is currently CPU-only.
    """

    def __init__(self, opt: Opt, shared=None):

        super().__init__(opt=opt, shared=shared)
        with PathManager.open(self.opt['model_file'], "rb") as f:
            self.module = torch.jit.load(f)

        # Track incoming history strings
        self.history: List[str] = []

    def share(self):
        """
        Share the scripted module object.
        """
        shared = super().share()
        shared['module'] = self.module
        return shared

    def observe(self, observation: Message) -> Message:
        # TODO: support self._validate_observe_invariants() method of TorchAgent

        self.history.append(observation['text'])

        return super().observe(observation)

    def self_observe(self, self_message: Message) -> None:
        # TODO: support self._validate_self_observe_invariants() method of TorchAgent

        assert self.observation is not None
        if self.observation['episode_done']:
            # oh this was the last example in the episode. reset the history
            self.history = []
            # additionally mark the last observation as invalid
            self.observation = None
        else:
            self.history.append(self_message['text'])

    def reset(self):
        super().reset()
        self.history = []

    def act(self) -> Message:
        response_text = self.module('\n'.join(self.history))
        response = Message({'text': response_text, 'episode_done': False})
        # self.observation will determine if we're going onto a new episode
        self.self_observe(response)
        return response
