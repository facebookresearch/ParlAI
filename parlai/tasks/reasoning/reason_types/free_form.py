#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from typing import Dict, Optional, Tuple

from parlai.tasks.reasoning.base import (
    t_REASON_PREFIX_TOKEN,
    t_REASON,
    AbstractReason,
)


class FreeFormReason(AbstractReason):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("FreeFormReason args")
        group.add_argument(
            "--reason-token",
            default="REASON: ",
            type=str,
        )
        return parser

    def get_full_reason_text(
        self, example_dict
    ) -> Tuple[t_REASON_PREFIX_TOKEN, t_REASON, Dict]:
        return (
            self.opt["reason_token"],
            example_dict["reason"],
            {"reason", example_dict["reason"]},
        )
