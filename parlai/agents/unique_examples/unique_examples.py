#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Simple agent which asserts that it has only seen unique examples.

Useful for debugging. Inherits from RepeatLabelAgent.
"""

from collections import defaultdict

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent


class UniqueExamplesAgent(RepeatLabelAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.unique_examples = defaultdict(int)

    def reset(self):
        super().reset()
        self.unique_examples = defaultdict(int)

    def act(self):
        obs = self.observation
        text = obs.get('text')
        if text in self.unique_examples:
            raise RuntimeError(f'Already saw example: {text}')
        else:
            self.unique_examples[text] += 1

        return super().act()
