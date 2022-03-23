#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_internal.projects.param_sweep_utils.param_sweep import run_grid
import projects.tod_simulator.sweeps.helpers as h
import projects.tod_simulator.sweeps.multiwoz.model_consts as al_variant
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
NODES = 1

MODELS = [
    "/checkpoint/mpchen/projects/tod_simulator/sweeps/pretrain_no_mwozTue_Feb_22/8bc/model --api-schemas True", "/checkpoint/mpchen/projects/tod_simulator/sweeps/pretrain_no_mwozTue_Feb_22/7f9/model --api-schemas False"]

# ALL_BASE = "/checkpoint/mpchen/projects/taskoriented/user_generator/uber_simulator/setup_exploration/"
# MODELS = [ ALL_BASE + model + "/model" for model in MODELS]
# MODELS = [model + " " + m.get_api_schemas_flag(model) for model in MODELS ]

print(MODELS)

# Usually won't have to change between variant + setup
SETUP = al_variant.LOTS_NUCLEUS09_TOKENEM
NUCLEUS_MM_TOPP = al_variant.NUCLEUS_VALS[SETUP]

rl_level = h.get_level_from_sweep_dir(SAVEROOT)
use_idx = rl_level - 1
CONVO_LADDER = al_variant.LADDER_LOOKUP[SETUP][:rl_level]
CONVO = "" #  CONVO_LADDER[use_idx]
AL_SAMPLES_LADDER = []
CUSTOM_NAME_BASE = al_variant.THIS_VARIANT + "__" + SETUP

if use_idx != -1:
    raise RuntimeError("THIS SCRIPT SHOULD ONLY BE USED FOR L0")


TEACHER_MWOZ = [
    ",".join(
        [
            f"parlai.tasks.multiwoz_v22.agents:SystemTeacher",
            f"parlai.tasks.multiwoz_v22.agents:UserSimulatorTeacher",
        ]
    )
]

EXISTING_CONVOS = " ".join(CONVO_LADDER)
EXISTING_AL_SAMPLES = " ".join(AL_SAMPLES_LADDER)

for model in MODELS:
    if (
        not (al_variant.SWEEP_SHORT_NAME[SETUP] in SCRIPT_NAME or rl_level == 1)
        or not SETUP in projects
    ):
        raise RuntimeError(
            f"The models listed are '{SETUP}' models, but this is *not* an '{SETUP}' script (at least by name)"
        )


# Things that don't really need to change ever
FIND_RANDOM_AL_SAMPLES = "random" in projects
RL_LEVEL = h.get_level_from_sweep_dir(SAVEROOT)
if al_variant.THIS_VARIANT not in projects:
    raise RuntimeError("Wrong variant being used!")


# Define param grid
grid = {
    ########### Args for TOD script
    "--nucleus-mm-topp": [NUCLEUS_MM_TOPP],
    # model-model script name
    "--custom-model-model-name": [CUSTOM_NAME_BASE + str(rl_level)],
    # Params for filtering things to set up next convo + active learning
#    "--find-random-al-samples": [FIND_RANDOM_AL_SAMPLES],
#    "--existing-al-files": [EXISTING_AL_SAMPLES],
    "--existing-train-files": [EXISTING_CONVOS],
    "--rl-level": [RL_LEVEL],
 #   "--skip-al-generation": [True],
    # dataset params
    "-t": TEACHER_MWOZ,
    #  Handelled in model args  "--api-schemas": [True, False],
    # other params
    "-im": MODELS,
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
    "-vp": [8],
    "--max-train-time": [HOURS * 60 * 60 - 30 * 60],
    "--load-from-checkpoint": ["true"],
    "--multitask-weights": ["1"],
    "--batchsize": [4],
    "--update-freq": [8],
    "-lr": [1e-4, 5e-5, 1e-5, 5e-6],
    "-vstep": [100],
    # Sweeping params
    "-vmt": [
        "token_em -vmm max --skip-generation True",
    ],
}


if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=SWEEP_NAME,
        saveroot=SAVEROOT,
        prefix="python -u -m projects.tod_simulator.scripts.tod_distributed_uber_multiwoz_script",
        partition="learnlab",
        jobtime=f"{HOURS}:00:00",
        gpus=GPUS,
        nodes=NODES,
        create_model_file=True,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        email_updates=True,
        wandb=True,
    )
