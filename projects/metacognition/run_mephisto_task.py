#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import shlex
import shutil
import subprocess
from mephisto.core.operator import Operator
from mephisto.utils.scripts import MephistoRunScriptParser, str2bool

from wsgi import QASample

parser = MephistoRunScriptParser()
parser.add_argument(
    "-uo",
    "--use-onboarding",
    default=True,
    help="Launch task with an onboarding world",
    type=str2bool,
)
architect_type, requester_name, db, args = parser.parse_launch_arguments()

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FRONTEND_SOURCE_DIR = os.path.join(TASK_DIRECTORY, "webapp")
FRONTEND_BUILD_DIR = os.path.join(FRONTEND_SOURCE_DIR, "build")
STATIC_FILES_DIR = os.path.join(FRONTEND_SOURCE_DIR, "src", "static")
USE_ONBOARDING = args["use_onboarding"]
QUESTIONS_PER_BATCH = 9

task_title = "Can our chatbot answer questions correctly 💯 and confidently 🙋 or does just spin us some yarn 🧶?"
task_description = f"You will be shown {QUESTIONS_PER_BATCH} questions (and the correct answers to them) per HIT. Judge correctness and certainty of the answer a chatbot gave to each of the questions!"

ARG_STRING = (
    "--blueprint-type static_react_task "
    f"--architect-type {architect_type} "
    f"--requester-name {requester_name} "
    f"--provider-type {args['provider_type']} "
    f'--task-title "\\"{task_title}\\"" '
    f'--task-description "\\"{task_description}\\"" '
    "--task-tags simple,correct,classification,language "
    f'--task-source "{TASK_DIRECTORY}/webapp/build/bundle.js" '
    f"--units-per-assignment 3 "
    f"--task-name metacognition_test "
    f'--extra-source-dir "{STATIC_FILES_DIR}" '
    f"--use-hobby 1 "
    f"--allow-concurrent 1 "
)

if USE_ONBOARDING:
    ARG_STRING += f"--onboarding-qualification metacognition150-qualification"


def prep(sample: QASample):
    clean_golds = []
    for gold in sample.gold:
        gold = gold.replace(" (disambiguation)", "")
        if gold not in clean_golds and QASample._tok(gold) not in sample.tok_question:
            clean_golds.append(gold)
    return {
        "question": sample.question,
        "prediction": sample.prediction,
        "golds": sorted(clean_golds),
    }


already_tabooed_questions = [
    "In the Harry Potter books there are 4 houses at Hogwarts School. Three of them are Gryffendor, Ravenclaw and Slytherin. Name the fourth.",
    "Who was the illustrator for most of Roald Dahl\u2019s stories for children?",
    "Produced until 2001, what was the name of the 128-bit game console produced by Sega that has developed quite a cult following?",
]

with open(
    os.path.join(STATIC_FILES_DIR, "blender3B_forced_IDK_test__need_2_more.jsonl")
) as f:
    all_samples = [
        s
        for s in [json.loads(line) for line in f if line]
        # if s["question"] not in already_tabooed_questions  # don't mess with it lol
    ]
    random.seed(0)
    random.shuffle(all_samples)
    batches = [
        all_samples[i : i + QUESTIONS_PER_BATCH]
        for i in range(0, len(all_samples), QUESTIONS_PER_BATCH)
    ]
    assert all([len(b) == QUESTIONS_PER_BATCH for b in batches[:-1]])
    if len(batches[-1]) != QUESTIONS_PER_BATCH:
        print("Last batch only has", len(batches[-1]), "items!")

    extra_args = {
        "static_task_data": [{"samples": b, "annotations": []} for b in batches]
    }


# build the task
def build_task():
    return_dir = os.getcwd()
    os.chdir(FRONTEND_SOURCE_DIR)
    if os.path.exists(FRONTEND_BUILD_DIR):
        shutil.rmtree(FRONTEND_BUILD_DIR)
    packages_installed = subprocess.call(["npm", "install"])
    if packages_installed != 0:
        raise Exception(
            "please make sure npm is installed, otherwise view "
            "the above error for more info."
        )

    webpack_complete = subprocess.call(["npm", "run", "dev"])
    if webpack_complete != 0:
        raise Exception(
            "Webpack appears to have failed to build your "
            "frontend. See the above error for more information."
        )
    os.chdir(return_dir)


build_task()

operator = Operator(db)
operator.parse_and_launch_run_wrapper(shlex.split(ARG_STRING), extra_args=extra_args)
operator.wait_for_runs_then_shutdown()
