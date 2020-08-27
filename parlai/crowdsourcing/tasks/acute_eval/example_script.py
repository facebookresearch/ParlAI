#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
import shlex
from mephisto.core.operator import Operator
from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import BLUEPRINT_TYPE
from mephisto.utils.scripts import MephistoRunScriptParser

"""
Example script for running ACUTE-EVAL.
The only argument that *must* be modified for this to be run is:
``pairings_filepath``:  Path to pairings file in the format specified in the README.md

The following args are useful to tweak to fit your specific needs;
    - ``annotations_per_pair``: A useful arg if you'd like to evaluate a given conversation pair
                                more than once.
    - ``num_matchup_pairs``:    Essentially, how many pairs of conversations you would like to evaluate
    - ``subtasks_per_unit``:     How many comparisons you'd like a turker to complete in one HIT

"""

TASK_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

parser = MephistoRunScriptParser()
parser.add_argument(
    "-pfp",
    "--pairings-filepath",
    default=f"{TASK_DIRECTORY}/pairings.jsonl",
    help="Path to pairings file",
)
parser.add_argument(
    "-app",
    "--annotations-per-pair",
    default=1,
    help="Annotations per pairing, to ensure worker agreement, default 1",
    type=int,
)
parser.add_argument(
    "-nmp",
    "--num-matchup-pairs",
    default=2,
    help="Number of pairs per model matchup, default 2",
    type=int,
)
parser.add_argument(
    "-spu",
    "--subtasks-per-unit",
    default=5,
    help="Number of conversations to evaluate per task, default 5",
    type=int,
)

architect_type, requester_name, db, args = parser.parse_launch_arguments()

USE_LOCAL = True

task_title = "Which Conversational Partner is Better?"
task_description = "Evaluate quality of conversations through comparison."
hit_keywords = "chat,evaluation,comparison,conversation"

ARG_STRING = (
    f"--blueprint-type {BLUEPRINT_TYPE} "
    f"--architect-type {architect_type} "
    f"--requester-name {requester_name} "
    f'--task-title "\\"{task_title}\\"" '
    f'--task-description "\\"{task_description}\\"" '
    "--task-reward 0.5 "
    f"--task-tags {hit_keywords} "
    f"--maximum-units-per-worker 0 "  # Num of units a worker is allowed to do, 0 is infinite
    f"--allowed-concurrent 1 "  # Workers can only do one task at a time, or onboarding may break
)

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
