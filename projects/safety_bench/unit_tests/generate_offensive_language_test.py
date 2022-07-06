#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the model's propensity to generate offensive language in 4 different settings:

- Safe (using task `blended_skill_talk`)
- Real-world noise (using task `twitter`)
- Unsafe (using task `dialogue_safety:standard`)
- Adversarial Unsafe (using task `bot_adversarial_dialogue:HumanSafetyEvaluation`)

We collect the model's responses to contexts from each of these settings and report the
percentage of examples that are flagged as offensive by each of the provided tools:
- A list of unsafe words/phrases
- A safety classifier
- (Optionally - if the user has access to it for their project) the Perspective API
"""
from parlai.core.opt import Opt
from parlai.utils import logging

from projects.safety_bench.utils.safety_testing import (
    collect_task_data_and_model_response,
    check_texts_with_safety_tooling,
    write_log,
    write_report,
)

from typing import Dict, Optional


def test_generation_safe(model_wrapper, opt: Opt) -> Dict:
    """
    Test generation on the BlendedSkillTalk valid test (largely safe content)
    """
    logging.info("[ Evaluating model responses in a safe setting ]")
    report = {}
    # get task data and model responses
    model_input, model_responses = collect_task_data_and_model_response(
        model_wrapper,
        opt=opt,
        task_name="blended_skill_talk",
        task_specific_opt={"datatype": "valid", "mutators": "flatten"},
        num_examples=180,
    )
    # evaluate all model responses
    score_tracker, stringified_scores = check_texts_with_safety_tooling(
        text_list=model_responses, opt=opt
    )

    report["Unsafe Generation:Safe Setting"] = score_tracker

    # write to log
    write_log(
        opt["log_folder"],
        "unsafe_generation=safe_setting",
        model_input,
        model_responses,
        stringified_scores,
    )

    logging.success("Finished evaluating in a safe setting")

    return report


def test_generation_realworldnoise(model_wrapper, opt: Opt) -> Dict:
    """
    Test generation on the Twitter valid test.

    This mimics a "real world noise" setting.
    """
    logging.info("[ Evaluating model responses in a real world noise setting ]")
    report = {}
    # get task data and model responses
    model_input, model_responses = collect_task_data_and_model_response(
        model_wrapper,
        opt=opt,
        task_name="twitter",
        task_specific_opt={"datatype": "valid"},
        num_examples=180,
    )
    # evaluate all model responses
    score_tracker, stringified_scores = check_texts_with_safety_tooling(
        text_list=model_responses, opt=opt
    )

    report["Unsafe Generation:Real World Noise Setting"] = score_tracker
    write_log(
        opt["log_folder"],
        "unsafe_generation=real_world_noise_setting",
        model_input,
        model_responses,
        stringified_scores,
    )

    logging.success("Finished evaluating in real world noise setting")

    return report


def test_generation_nonadversarial_unsafe(model_wrapper, opt: Opt) -> Dict:
    """
    Test generation on the standard unsafe set.
    """
    logging.info("[ Evaluating model responses in a non-adversarial unsafe setting ]")
    report = {}
    # get task data and model responses

    def filter_not_ok(act: Dict) -> Optional[str]:
        if act["labels"] == ["__notok__"]:
            return act["text"]

        return None

    model_input, model_responses = collect_task_data_and_model_response(
        model_wrapper,
        opt=opt,
        task_name="dialogue_safety:standard",
        task_specific_opt={"round": 3, "round_only": False},
        num_examples=180,
        filter_func=filter_not_ok,
    )

    # evaluate all model responses
    score_tracker, stringified_scores = check_texts_with_safety_tooling(
        text_list=model_responses, opt=opt
    )

    report["Unsafe Generation:Non-adversarial Unsafe Setting"] = score_tracker
    write_log(
        opt["log_folder"],
        "unsafe_generation=non_adversarial_unsafe_setting",
        model_input,
        model_responses,
        stringified_scores,
    )
    logging.success("Finished evaluating in a non-adversarial unsafe setting")

    return report


def test_generation_adversarial_unsafe(model_wrapper, opt: Opt) -> Dict:
    """
    Test generation on teh Bot Adversarial Dialogues set.
    """
    logging.info("[ Evaluating model responses in an adversarial unsafe setting ]")
    report = {}
    # get task data and model responses
    model_input, model_responses = collect_task_data_and_model_response(
        model_wrapper,
        opt=opt,
        task_name="bot_adversarial_dialogue:HumanSafetyEvaluation",
        task_specific_opt={"bad_include_persona": False, "flatten_dialogue": True},
    )

    # evaluate all model responses
    score_tracker, stringified_scores = check_texts_with_safety_tooling(
        text_list=model_responses, opt=opt
    )

    report["Unsafe Generation:Adversarial Unsafe Setting"] = score_tracker
    write_log(
        opt["log_folder"],
        "unsafe_generation=adversarial_unsafe_setting",
        model_input,
        model_responses,
        stringified_scores,
    )
    logging.success("Finished evaluating in adversarial unsafe setting")

    return report


def main(model_wrapper, opt: Opt) -> Dict:
    final_report = {}
    report = test_generation_safe(model_wrapper, opt)
    final_report.update(report)
    report = test_generation_realworldnoise(model_wrapper, opt)
    final_report.update(report)
    report = test_generation_nonadversarial_unsafe(model_wrapper, opt)
    final_report.update(report)
    report = test_generation_adversarial_unsafe(model_wrapper, opt)
    final_report.update(report)

    write_report(
        opt["log_folder"], "offensive_language_generation_metrics", final_report
    )

    return final_report
