#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from parlai.core.message import Message
from projects.style_gen.modules import STYLE_SEP_TOKEN
from projects.style_gen.style_gen import StyleGenAgent


class NoBiasStyleGenAgent(StyleGenAgent):
    """
    Subclass of style gen agent that appends "no_bias" to every example's context.
    """

    def get_temp_history(self, observation: Message) -> Optional[str]:
        return STYLE_SEP_TOKEN + 'no_bias'
