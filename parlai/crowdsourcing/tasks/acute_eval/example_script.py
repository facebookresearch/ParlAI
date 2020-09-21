#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from dataclasses import dataclass, field

import hydra
from mephisto.core.operator import Operator
from mephisto.utils.scripts import load_db_and_process_config
from omegaconf import DictConfig

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import BLUEPRINT_TYPE

"""
Example script for running ACUTE-Evals.
The only argument that *must* be set for this to be run is:
``pairings_filepath``:  Path to pairings file in the format specified in the README.md

The following args are useful to tweak to fit your specific needs;
- ``annotations_per_pair``: A useful arg if you'd like to evaluate a given conversation pair more than once.
- ``num_matchup_pairs``: Essentially, how many pairs of conversations you would like to evaluate
- ``subtasks_per_unit``: How many comparisons you'd like a turker to complete in one HIT

"""

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    "conf/base",
    {"conf": "example"},
]

extra_args = {
    "pairings_filepath": args['pairings_filepath'],
    "block_on_onboarding_fail": True,
    "block_qualification": f"acute_eval_{int(time.time())}_block",
    # num times to use the same conversation pair
    "annotations_per_pair": args["annotations_per_pair"],
    "random_seed": 42,  # random seed
    "subtasks_per_unit": args[
        "subtasks_per_unit"
    ],  # num comparisons to show within one unit
    "num_matchup_pairs": args[
        "num_matchup_pairs"
    ],  # num pairs of conversations to be compared
    # question phrasing
    "s1_choice": "I would prefer to talk to <Speaker 1>",
    "s2_choice": "I would prefer to talk to <Speaker 2>",
    "eval_question": "Who would you prefer to talk to for a long conversation?",
    "assignment_duration_in_seconds": 600,
}

operator = Operator(db)
operator.parse_and_launch_run_wrapper(shlex.split(ARG_STRING), extra_args=extra_args)
operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)
