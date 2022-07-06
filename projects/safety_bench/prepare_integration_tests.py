#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Prepare data for integration tests. User will need to subsequently follow instructions
in the `Human Evalulations` section of the README here:

<https://github.com/facebookresearch/ParlAI/tree/main/projects/safety_recipes>

For example, to prepare integration tests for the adversarial setting for the wrapper for
the 90M parameter BlenderBot model, try the following fommand:
```
python prepare_integration_tests.py --wrapper blenderbot_90M --safety-setting adversarial
```
"""
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
import parlai.utils.logging as logging
from parlai.utils.io import PathManager

from projects.safety_bench.utils.wrapper_loading import (
    load_wrapper_module,
    setup_wrapper_registry,
    register_model_wrapper,  # noqa: F401
    MODEL_WRAPPER_REGISTRY,  # noqa: F401
)
from projects.safety_bench.utils.safety_testing import (
    collect_task_data_and_model_response,
)
from projects.safety_bench.utils.colors import color


import os
from typing import Optional
import json


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, "Safety integration tests")
    # Get command line arguments
    parser.add_argument(
        "-w",
        "--wrapper",
        type=str,
        help="Registered name of model wrapper",
        required=True,
    )
    parser.add_argument(
        "--log-folder", type=str, help="Where to write logs", default="/tmp/"
    )
    parser.add_argument(
        "--safety-setting",
        type=str,
        choices={"adversarial", "nonadversarial"},
        help="Which human evaluation setting to prepare for: either `adversarial` or `nonadversarial`",
        required=True,
    )
    parser.add_argument("--debug", type="bool", default=False, help="Use in DEBUG mode")
    return parser


def _check_log_dir(dir: Optional[str]):
    if dir is None:
        raise RuntimeError(
            "Must specify log directory via `--log-folder <path to folder>`"
        )

    if not os.path.exists(dir):
        logging.warning(f"[ Directory {dir} does not exist, creating... ]")
        os.makedirs(dir)

    return


def _next_steps(safety_setting: str, task_data_path: str, indices_path: str):
    logging.success(f"Data preparation for {safety_setting} complete.")
    print(f"\n\n\n{color.PURPLE}{color.BOLD}{color.UNDERLINE}NEXT STEPS:{color.END}")
    print(
        f"Your task data path was written to: {color.BLUE}{task_data_path}{color.END}"
    )
    print(f"Your indices path was written to: {color.BLUE}{indices_path}{color.END}")
    print(
        f"\nPlease place both of these paths into the folder {color.YELLOW}{color.BOLD}ParlAI/projects/safety_recipes/human_safety_evaluation/task_config{color.END}, replacing the existing files."
    )
    print(
        f"\nTo launch your evaluation task on Mechanical Turk, you must install {color.BOLD}Mephisto{color.END}; see instructions here: {color.CYAN}{color.BOLD}https://github.com/facebookresearch/Mephisto{color.END}"
    )
    print(
        f"\nFollowing your Mephisto setup, you can launch the task with the command:\n{color.GREEN}{color.BOLD}python projects/safety_recipes/human_safety_evaluation/run.py{color.END}"
    )
    print(
        "\nSee the Mephisto docs for further instructions on managing crowdsourcing tasks.\n\n"
    )


def prepare_integration_tests(opt: Opt):
    logging.info("[ Preparing data for adversarial human evaluation ... ]")

    _check_log_dir(opt["log_folder"])
    wrapper_name = opt["wrapper"]
    wrapper = load_wrapper_module(wrapper_name)()
    safety_setting = opt["safety_setting"]

    if safety_setting == "adversarial":
        task_name = "bot_adversarial_dialogue:HumanSafetyEvaluation"
        task_specific_opt = {
            "flatten_dialogue": True,
            "datatype": "valid",
            "bad_include_persona": False,
        }
    else:
        task_name = "bot_adversarial_dialogue:HumanNonadvSafetyEvaluation"
        task_specific_opt = {"datatype": "valid"}

    # Collect model responses to task
    teacher_acts, model_responses = collect_task_data_and_model_response(
        wrapper,
        opt=opt,
        task_name=task_name,
        task_specific_opt=task_specific_opt,
        save_teacher_acts=True,
    )

    # Format the way it is required for human evaluations
    chat_logs = []
    for inp_act, resp in zip(teacher_acts, model_responses):
        episode = []
        inp = inp_act["text"]
        inp_split = inp.split("\n")
        if len(inp_split) % 2 != 1:
            # Join first input
            first_inp = inp_split[0]
            inp_split = inp_split[1:]
            inp_split[0] = "\n".join([inp_split[0], first_inp])
        inp_split.append(resp)
        # section off into pairs
        pairs = [inp_split[i : i + 2] for i in range(0, len(inp_split), 2)]
        for pair in pairs:
            episode.append(
                [
                    {'text': pair[0], 'episode_done': False, 'id': 'human'},
                    {'text': pair[1], 'episode_done': False, 'id': 'bot'},
                ]
            )
        # mark the last episode as done
        episode[-1][1]['episode_done'] = True
        if "human_eval_turn_range" in inp_act:
            turn_range = [int(x) for x in inp_act["human_eval_turn_range"].split("|")]
            episode = episode[turn_range[0] : turn_range[1] + 1]

        chat_logs.append(episode)

    task_data_path = os.path.join(opt["log_folder"], "task_data.jsonl")
    indices_path = os.path.join(opt["log_folder"], "annotation_indices.jsonl")
    with PathManager.open(task_data_path, 'w') as fw:
        for episode in chat_logs:
            fw.write(json.dumps(episode) + '\n')
    with PathManager.open(indices_path, 'w') as fw:
        for episode in chat_logs:
            fw.write(f'[{len(episode) * 2 -1}]' + '\n')

    _next_steps(safety_setting, task_data_path, indices_path)


class PrepareIntegrationTests(ParlaiScript):
    @classmethod
    def setup_args(cls):
        # first make sure all wrappers are imported
        setup_wrapper_registry()
        return setup_args()

    def run(self):
        return prepare_integration_tests(self.opt)


if __name__ == "__main__":
    PrepareIntegrationTests.main()
