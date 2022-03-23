#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_internal.projects.param_sweep_utils.param_sweep import run_grid
import parlai_fb.projects.taskoriented.user_generator.uber_simulator.bart_2.model_consts as m
import time
import os

SCRIPT_NAME = os.path.basename(__file__).replace(".py", "")
TODAY = format(time.asctime().replace(":", "-").replace(" ", "_")[:-14])
SWEEP_NAME = f"{SCRIPT_NAME}{TODAY}"

here_path = os.path.realpath(__file__).replace(".py", "")
projects = here_path[here_path.find("/projects") :]
SAVEROOT = "/checkpoint/mpchen" + projects + TODAY + "/"

HOURS = 4
GPUS = 8

path = "/checkpoint/mpchen/projects/tod_simulator/sweeps/pretrain_no_mwozTue_Feb_22/8bc/model"

# Define param grid
grid = {
    "--goal-grounding-model": [
        f"parlai.tasks.multiwoz_v22.agents:SingleGoalAgent",
    ],
    "--api-resp-model": ["parlai_fb.agents.tod.agents:TodStandaloneApiAgent"],
    "--standalone-api-file": [
        "/checkpoint/mpchen/projects/user_simulator/standalone_api_data/multiwoz_simpler/output", 
    ],
    "--exact-api-call": [True],
    "--display-examples": [False],
    "--skip-generation": [False],
    "-bs": [128],
    # Varying args
    "--num-episodes": [999999999],  # just something high lol
    "--api-schema-grounding-model": [
        "parlai.tasks.multiwoz_v22.agents:SingleApiSchemaAgent"
    ],
    "--inference": ["nucleus"],
    "--topp": [0.9],
    "--episodes-randomization-seed": [
        str(x)
        + " "
        + f" --system-model-file {path} --api-schemas True --user-model-file {path} --api-schemas True --datatype train --report-filename {SAVEROOT}/wozFT_trainData-{x} --world-logs {SAVEROOT}/wozFT_trainData-{x}"        
        for x in range(0, 20)
    ],
}


if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=SWEEP_NAME,
        saveroot=SAVEROOT,
        prefix="python -u -m parlai.scripts.distributed_tod_world_script",
        partition="learnlab",
        jobtime=f"{HOURS}:00:00",
        gpus=GPUS,
        nodes=1,
        create_model_file=False,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        email_updates=True,
    )
