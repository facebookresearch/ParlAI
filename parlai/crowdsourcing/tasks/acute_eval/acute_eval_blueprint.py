#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
from dataclasses import dataclass, field
from typing import ClassVar, List, Type, Any, Dict, Iterable, TYPE_CHECKING

from mephisto.core.registry import register_mephisto_abstraction
from mephisto.data_model.blueprint import Blueprint, BlueprintArgs, SharedTaskState
from mephisto.data_model.assignment import InitializationData
from omegaconf import MISSING, DictConfig

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_agent_state import (
    AcuteEvalAgentState,
)
from parlai.crowdsourcing.tasks.acute_eval.acute_eval_builder import AcuteEvalBuilder
from parlai.crowdsourcing.tasks.acute_eval.acute_eval_runner import AcuteEvalRunner

if TYPE_CHECKING:
    from mephisto.data_model.blueprint import AgentState, TaskRunner, TaskBuilder

BLUEPRINT_TYPE = "acute_eval"


@dataclass
class AcuteEvalBlueprintArgs(BlueprintArgs):
    _blueprint_type: str = BLUEPRINT_TYPE
    _group: str = field(
        default="AcuteEvalBlueprint",
        metadata={
            'help': """\
Tasks launched from the ACUTE-Eval blueprint require sets of pairings for workers to \
be able to compare to. These pairings should be provided as a .jsonl with \
--pairings-filepath."""
        },
    )
    annotations_per_pair: int = field(
        default=1,
        metadata={"help": "Number of annotations per conversation comparison pair"},
    )
    pairings_filepath: str = field(
        default=MISSING,
        metadata={"help": "path to the file containing the task dictionaries"},
    )
    s1_choice: str = field(
        default="I would prefer to talk to <Speaker 1>",
        metadata={"help": "text next to speaker 1 radio button"},
    )
    s2_choice: str = field(
        default="I would prefer to talk to <Speaker 2>",
        metadata={"help": "text next to speaker 2 radio button"},
    )
    eval_question: str = field(
        default="Who would you prefer to talk to for a long conversation?",
        metadata={
            "help": 'question to present to turker for comparison (e.g. "Which speaker is better?")'
        },
    )
    block_on_onboarding_fail: bool = field(
        default=True, metadata={"help": "whether to block on onboarding failure"}
    )
    num_matchup_pairs: int = field(
        default=2, metadata={"help": "Number of pairs per model matchup, default 2"}
    )
    subtasks_per_unit: int = field(
        default=5, metadata={"help": "number of subtasks/comparisons to do per unit"}
    )
    onboarding_threshold: float = field(
        default=0.75,
        metadata={"help": "minimum accuracy on onboarding tasks, as a float 0-1.0"},
    )
    random_seed: int = field(default=42, metadata={"help": "seed for random"})
    additional_task_description: str = field(
        default='', metadata={"help": "Additional text to show on the left pane"}
    )


# WISH AcuteEval's blueprint can probably be extended to compare more than just convos
@register_mephisto_abstraction()
class AcuteEvalBlueprint(Blueprint):
    """
    Blueprint for a task that asks humans to compare conversational outputs.
    """

    AgentStateClass: ClassVar[Type["AgentState"]] = AcuteEvalAgentState
    TaskBuilderClass: ClassVar[Type["TaskBuilder"]] = AcuteEvalBuilder
    TaskRunnerClass: ClassVar[Type["TaskRunner"]] = AcuteEvalRunner
    ArgsClass = AcuteEvalBlueprintArgs
    supported_architects: ClassVar[List[str]] = ["mock"]  # TODO update
    BLUEPRINT_TYPE = BLUEPRINT_TYPE

    @classmethod
    def assert_task_args(cls, args: DictConfig, shared_state: SharedTaskState) -> None:
        """
        Ensure that the data can be properly loaded.
        """
        if args.blueprint.get("pairings_filepath", None) is not None:
            pairings_filepath = os.path.expanduser(args.blueprint.pairings_filepath)
            assert os.path.exists(
                pairings_filepath
            ), f"Provided file {pairings_filepath} doesn't exist"
        else:
            raise AssertionError(
                "Must provide one of a data csv, json, or a list of tasks"
            )

        if args.blueprint.block_on_onboarding_fail is True:
            if args.blueprint.get("block_qualification", None) is None:
                raise AssertionError(
                    "Must provide `block_qualification` to use `block_on_onboarding_fail`"
                )

    def get_frontend_args(self) -> Dict[str, Any]:
        """
        Specifies what options within a task_config should be forwarded to the client
        for use by the task's frontend.
        """
        return {
            "task_description": "Placeholder Task Description - Javascript failed to load",
            "frame_height": 650,
            "num_subtasks": self.args.blueprint.subtasks_per_unit,
            "question": self.args.blueprint.eval_question,
            "block_mobile": True,
            "get_task_feedback": False,  # TODO(#95) make option
            "additional_task_description": self.args.blueprint.additional_task_description,
        }

    def get_initialization_data(self) -> Iterable["InitializationData"]:
        """
        Return the InitializationData retrieved from the specified stream.
        """
        # TODO(#99) once we can release HITs over time, configure this to
        # release as many as needed thusfar and top off when
        # onboardings fail
        num_conversations = math.ceil(
            self.args.blueprint.num_matchup_pairs
            / max((self.args.blueprint.subtasks_per_unit - 1), 1)
        )  # release enough hits to finish all annotations requested
        return [
            InitializationData(shared={}, unit_data=[{}])
            for _ in range(num_conversations)
        ]
