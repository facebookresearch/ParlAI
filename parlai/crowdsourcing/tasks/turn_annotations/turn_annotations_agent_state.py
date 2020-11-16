#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    ParlAIChatAgentState,
)


class TurnAnnotationsAgentState(ParlAIChatAgentState):
    """
    ParlAI-style chat agent that saves turn-annotations info at the end of the chat.
    """

    def load_data(self) -> None:
        super().load_data()
        # TODO: add docstring
        # {{{TODO: write this. Do we even want to call super().load_data()? What are the reqs for this?}}}

    def save_data(self) -> None:
        super().save_data()
        # TODO: add docstring
        # {{{TODO: write this. do we even want to call super().save_data() or no?}}}
