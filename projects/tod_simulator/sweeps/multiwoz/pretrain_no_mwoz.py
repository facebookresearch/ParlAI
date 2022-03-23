#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_internal.projects.param_sweep_utils.param_sweep import run_grid
import time
import os

SCRIPT_NAME = os.path.basename(__file__).replace(".py", "")
TODAY = format(time.asctime().replace(":", "-").replace(" ", "_")[:-14])
SWEEP_NAME = f"{SCRIPT_NAME}{TODAY}"

here_path = os.path.realpath(__file__).replace(".py", "")
projects = here_path[here_path.find("/projects") :]
SAVEROOT = "/checkpoint/mpchen" + projects + TODAY

HOURS = 23
GPUS = 8

# This version was able to benefit from the open source versions of the teachers

TEACHERS_NO_GSGD_GOOD = [
    "taskmaster:SystemTeacher",
    "taskmaster2:SystemTeacher",
    "taskmaster3:SystemTeacher",
    "msr_e2e:SystemTeacher",
    "taskmaster2:UserSimulatorTeacher",
    "taskmaster3:UserSimulatorTeacher",
    "msr_e2e:UserSimulatorTeacher",
    "multidogo:UserSimulatorTeacher",
]

TEACHERS_NO_GSGD_FUNKY = [
    "metalwoz:SystemTeacher",  # also without the STANDARD_ whatevers, so could be interesting.
    "multidogo:SystemTeacher",  # API responses make no sense
    "metalwoz:UserSimulatorTeacher",  # also without the STANDARD_ whatevers, so could be interesting.
    "taskmaster:UserSimulatorTeacher",  # no goals
]

TEACHER_GSGD = [
    "google_sgd_simulation_splits:InDomainUserSimulatorTeacher",
    "google_sgd_simulation_splits:InDomainSystemTeacher",
]

ALL_TEACHERS = TEACHER_GSGD + TEACHERS_NO_GSGD_GOOD + TEACHERS_NO_GSGD_FUNKY
ALL_TEACHERS_NO_GSGD = TEACHERS_NO_GSGD_GOOD + TEACHERS_NO_GSGD_FUNKY
ALL_GOOD_TEACHERS = TEACHER_GSGD + TEACHERS_NO_GSGD_GOOD

TEACHER_OPTIONS = [
    ",".join(ALL_TEACHERS),
    #    ",".join(ALL_TEACHERS_NO_GSGD),
 #   ",".join(ALL_GOOD_TEACHERS),
    #    ",".join(TEACHERS_NO_GSGD_GOOD),
 #   ",".join(TEACHER_GSGD),
]

print(TEACHER_OPTIONS[0])

# Define param grid
grid = {
    # dataset params
    "-t": TEACHER_OPTIONS,
    "--api-schemas": [True, False],
    # other params
    "--model": [
        "parlai_fb.agents.bart.r3f:R3fFirstTurnHistoryRepeatAgent",
    ],
    "--fp16": [True],
    "--label-truncate": [512],
    "--log-every-n-secs": [30],
    "--lr-scheduler": ["invsqrt"],
    "--max-lr-steps": [-1],
    "--max-train-steps": [-1],
    "--optimizer": ["adam"],
    "--save-after-valid": [True],
    "--text-truncate": [512],
    "--warmup-updates": [1000],
    "--fp16-impl": ["mem_efficient"],
    "--gradient-clip": [0.1],
    "--skip-generation": [True],
    "-vp": [8],
    "--max-train-time": [HOURS * 60 * 60 - 30 * 60],
    "--load-from-checkpoint": ["true"],
    "-vmt": ["token_em -vmm max"],
    "--multitask-weights": ["stochastic"],
    # Sweeping params
    "--batchsize": [4],
    "--update-freq": [8],
    "-lr": [1e-4],
    "-vstep": [1000],
}


if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=SWEEP_NAME,
        saveroot=SAVEROOT,
        prefix="python -u -m parlai.scripts.distributed_train",
        partition="learnlab",
        jobtime=f"{HOURS}:00:00",
        gpus=8,
        nodes=1,
        create_model_file=True,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        email_updates=True,
        wandb=True,
    )
