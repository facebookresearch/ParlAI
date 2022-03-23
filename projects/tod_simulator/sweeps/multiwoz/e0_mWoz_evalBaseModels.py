#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_internal.projects.param_sweep_utils.param_sweep import run_grid
import time
import os
import json

SCRIPT_NAME = os.path.basename(__file__).replace(".py", "")
TODAY = format(time.asctime().replace(":", "-").replace(" ", "_")[:-14])

SWEEP_NAME = f"{SCRIPT_NAME}{TODAY}"

here_path = os.path.realpath(__file__).replace(".py", "")
projects = here_path[here_path.find("/projects") :]
SAVEROOT = "/checkpoint/mpchen" + projects + TODAY

HOURS = 3
GPUS = 8
NODES = 1



MODELS = [
    "/checkpoint/mpchen/projects/tod_simulator/sweeps/pretrain_no_mwozTue_Feb_22/8bc/model --api-schemas True", "/checkpoint/mpchen/projects/tod_simulator/sweeps/pretrain_no_mwozTue_Feb_22/7f9/model --api-schemas False"]
print(MODELS)

TEACHER_MWOZ = [
    "multiwoz_v22",
    "multiwoz_v22:UserSimulatorTeacher", 
]

# Define param grid
grid = {
    # dataset params
    "-t": TEACHER_MWOZ,
    # other params
    "-mf": MODELS,
    "--model": [
        "parlai_fb.agents.bart.r3f:R3fFirstTurnHistoryRepeatAgent",
    ],
    "--batchsize": [128],
    "--skip-generation": [False],
    "-dt": ["valid", "test"], 
}


if __name__ == "__main__":
    run_grid(
        grid,
        {},
        SWEEP_NAME,
        saveroot=SAVEROOT,
        prefix="python -m parlai.scripts.eval_model",
        create_model_file=False,
        include_job_id=False,
        gpus=GPUS,
        data_parallel=True,
        copy_env=True,
        nodes=1,
        cpus=10,
        volta=True,
        volta32=True,
        partition="learnlab",
        jobtime="{}:00:00".format(HOURS),
        hashname=True,
        requeue=False,
        mem_gb=400,
    )
