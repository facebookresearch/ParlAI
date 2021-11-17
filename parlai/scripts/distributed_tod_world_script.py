#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Distributed script for running TOD model-model chats.

Not to be called directly; should be called from SLURM
"""

from parlai.scripts.tod_world_script import TodWorldScript
from parlai.core.script import ParlaiScript
import parlai.utils.distributed as distributed_utils


class DistributedTodWorldScript(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = TodWorldScript.setup_args()
        parser.add_distributed_training_args()
        parser.add_argument("--port", type=int, default=61337, help="TCP port number")
        return parser

    def run(self):
        with distributed_utils.slurm_distributed_context(self.opt) as opt:
            return TodWorldScript(opt).run()


if __name__ == "__main__":
    DistributedTodWorldScript.main()
