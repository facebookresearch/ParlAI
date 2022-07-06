#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Run all safety unit tests for a model.

For example, to run all unit tests for the wrapper for the 90M parameter
BlenderBot model, try the following command:

```
python run_unit_tests.py --wrapper blenderbot_90M
```
"""
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
import parlai.utils.logging as logging

from projects.safety_bench.utils.wrapper_loading import (  # noqa: F401
    load_wrapper_module,
    setup_wrapper_registry,
    register_model_wrapper,
    MODEL_WRAPPER_REGISTRY,
)
from projects.safety_bench.unit_tests import (
    generate_offensive_language_test,
    response_to_offensive_language_test,
)
from projects.safety_bench.utils.colors import color
from projects.safety_bench.utils.perspective_api import get_perspective_api_key

import os
from typing import Optional


PAPER_LINK = "<https://arxiv.org/abs/2107.03451>"
PERSONA_BIAS_PAPER_LINK = "Sheng et. al (2021): <https://arxiv.org/abs/2104.08728>"


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, "Safety unit tests")
    # Get command line arguments
    parser.add_argument(
        "-w",
        "--wrapper",
        type=str,
        help="Registered name of model wrapper",
        required=True,
    )
    parser.add_argument(
        "--log-folder",
        type=str,
        default="/tmp/",
        help="Where to write logs of model outputs",
    )
    parser.add_argument(
        "--tests-to-run",
        type=str,
        default="all",
        choices={"all", "generate", "response"},
        help=(
            "Which tests to run; by default, run all. "
            "If `generate`, run tests for generating offensive language. "
            "If `response`, run tests for checking responses to offensive language. "
        ),
    )
    parser.add_argument("--debug", type="bool", default=False, help="Use in DEBUG mode")
    return parser


def _pretty_report(key_item, report):
    """
    Print a pretty report.
    """
    print(f"\n{color.PURPLE}{color.BOLD}{key_item}:{color.END}")
    metrics = {key: val for key, val in report.items() if key.startswith(key_item)}
    for metric, val in metrics.items():
        print(f"{color.BLUE}{metric.split(':')[-1]}:{color.END}")
        for key in sorted(list(val.keys())):
            v = val[key]
            assert v >= 0.0 and v <= 1.0
            if v <= 1 / 3:
                v_col = color.GREEN
            elif v <= 2 / 3:
                v_col = color.YELLOW
            else:
                v_col = color.RED
            print(f"\t{key}: {v_col}{v * 100 :.2f}{color.END}")


def _check_log_dir(dir: Optional[str]):
    if dir is None:
        raise RuntimeError(
            "Must specify log directory via `--log-folder <path to folder>`"
        )

    if not os.path.exists(dir):
        logging.warning(f"[ Directory {dir} does not exist, creating... ]")
        os.makedirs(dir)

    return


def _interpret_results():
    print(
        f"\n\n{color.YELLOW}{color.BOLD}How should I interpret my results?{color.END}"
    )
    print(
        f"\nFor the {color.BOLD}Unsafe Generation{color.END} test, we examine how the model responds "
        "to various dialogue inputs, representing 4 different settings. We report the percentage of the "
        "model's responses that are flagged as unsafe by each of the provided tools,  including the percentage "
        "that are flagged as unsafe by all tools or by at least one of the tools. "
        "Please check the provided logs to see examples where the tools agree and where they disagree. "
        f"\n\nFor the {color.BOLD}Response to Offensive Language{color.END} test, we examine how the model "
        f"responds to a previously constructed set of hateful inputs by {PERSONA_BIAS_PAPER_LINK}. "
        f"We attempt to ascertain whether the model's response affirms the hateful input by measuring the percentage "
        "of responses that (1) do not contain negations (2) are flagged as offensive by a safety classifier that uses "
        "context, and (3) has positive sentiment. Each of these is simply a proxy for affirmation, and we recommend "
        "reading the provided logs to understand how the model behaves. "
        f"\n\nFurther details are provided in Section 5 of {PAPER_LINK}."
    )


def _disclaimer():
    print(f"\n\n{color.RED}{color.BOLD}DISCLAIMER:{color.END}")
    disclaimer_text = (
        "These unit tests are limited in many ways. "
        "The tools are imperfect and known to be biased. "
        "The provided settings are not comprehensive. "
        "These tests cannot guarantee the safety of your model, and are meant simply as a first pass at understanding its safety limitations. "
        f"Please see further discussion in Section 5 of {PAPER_LINK} about the limitations and next steps. "
        "We recommend using human evaluation and adversarial probing to further understand the model's "
        "ability to generate or respond inappropriately to unsafe content."
    )
    print(f"{color.RED}{disclaimer_text}{color.END}")


def run_safety_unit_tests(opt):
    get_perspective_api_key(opt)  # try loading Perspective API key if it exists
    _check_log_dir(opt["log_folder"])
    wrapper_name = opt["wrapper"]
    wrapper = load_wrapper_module(wrapper_name)()

    # Tests that check for generating offensive language
    if opt["tests_to_run"] in ["all", "generate"]:
        gen_report = generate_offensive_language_test.main(wrapper, opt)
    else:
        gen_report = None

    # Tests that check the appropriateness of response to offensive language
    if opt["tests_to_run"] in ["all", "response"]:
        resp_report = response_to_offensive_language_test.main(wrapper, opt)
    else:
        resp_report = None

    if opt.get("debug"):
        logging.error("WARNING: ran in debug mode. Numbers are incorrect")

    # generate the rainbow stars
    stars = (
        "".join([col + "*" * (60 // len(color.RAINBOW)) for col in color.RAINBOW])
        + color.END
    )
    print("\n\n" + stars)
    print(
        f"{color.CYAN}{color.BOLD}{color.UNDERLINE}Safety Unit Tests: Report for {wrapper_name} {color.END}"
    )
    if gen_report is not None:
        # generation metrics
        _pretty_report("Unsafe Generation", gen_report)
    if resp_report is not None:
        # response metrics
        _pretty_report("Response to Offensive Language", resp_report)

    print(stars + "\n\n")
    print(
        f"All model logs -- including safety scores -- can be found in {color.YELLOW}{opt['log_folder']}{color.END}."
    )
    _interpret_results()
    _disclaimer()


class SafetyUnitTests(ParlaiScript):
    @classmethod
    def setup_args(cls):
        # first make sure all wrappers are imported
        setup_wrapper_registry()
        return setup_args()

    def run(self):
        return run_safety_unit_tests(self.opt)


if __name__ == "__main__":
    SafetyUnitTests.main()
