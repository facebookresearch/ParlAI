#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from mephisto.operations.registry import register_mephisto_abstraction
from mephisto.abstractions.blueprint import SharedTaskState
from mephisto.abstractions.blueprints.static_react_task.static_react_blueprint import (
    StaticReactBlueprint,
    StaticReactBlueprintArgs,
)
from omegaconf import DictConfig

if TYPE_CHECKING:
    from mephisto.data_model.task import TaskRun


def get_task_path():
    return os.path.dirname(__file__)


STATIC_BLUEPRINT_TYPE = 'turn_annotations_static_blueprint'
STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE = 'turn_annotations_static_inflight_qa_blueprint'


@dataclass
class TurnAnnotationsStaticBlueprintArgs(StaticReactBlueprintArgs):
    _blueprint_type: str = STATIC_BLUEPRINT_TYPE
    _group: str = field(
        default="TurnAnnotationsStaticBlueprint",
        metadata={
            'help': """This task renders conversations from a file and asks for turn by turn annotations of them."""
        },
    )
    random_seed: int = field(
        default=42, metadata={"help": 'Seed for random operations'}
    )
    annotation_question: str = field(
        default='Does this comment require any annotations? (Check all that apply)',
        metadata={
            "help": "The string displayed above the checkboxes for each annotation in the task."
        },
    )
    subtasks_per_unit: int = field(
        default=-1, metadata={"help": "Number of subtasks/comparisons to do per unit"}
    )
    annotation_indices_jsonl: Optional[str] = field(
        default=None,
        metadata={
            "help": "Specify which utterance indices to annotate per conversation in a JSONL file. Must be same length as conversations data-jsonl file. See example file in task_config/annotation_indices_example.jsonl"
        },
    )
    ask_reason: bool = field(
        default=False,
        metadata={
            "help": "If we want to ask the crowdworker for a reason for each of their annotations in a text field"
        },
    )
    conversation_count: Optional[int] = field(
        default=None,
        metadata={
            "help": "Specify a positive integer if you want to use only the first N conversations in the data file"
        },
    )
    onboarding_data: str = field(
        default=os.path.join(get_task_path(), 'task_config/onboarding.json'),
        metadata={
            "help": "Path to data and answers for onboarding task in JSON format"
        },
    )
    annotation_buckets: Optional[str] = field(
        default=None,
        metadata={
            "help": "As per Turn Annotations task, path to annotation buckets which will be checkboxes in the frontend for worker to annotate an utterance. If none provided, no checkboxes."
        },
    )
    response_field: bool = field(
        default=False,
        metadata={
            "help": "If we want a freeform textbox input for the crowdworker to respond to the message."
        },
    )


@register_mephisto_abstraction()
class TurnAnnotationsStaticBlueprint(StaticReactBlueprint):
    """
    This Blueprint has a subtasks number option to combine multiple conversations into
    "sub-HITs".

    It also has options for the onboarding data answers and the annotation bucket
    definitions.
    """

    ArgsClass = TurnAnnotationsStaticBlueprintArgs
    BLUEPRINT_TYPE = STATIC_BLUEPRINT_TYPE

    def __init__(
        self, task_run: "TaskRun", args: "DictConfig", shared_state: "SharedTaskState"
    ):
        super().__init__(task_run, args=args, shared_state=shared_state)
        random.seed(self.args.blueprint.random_seed)
        np.random.seed(self.args.blueprint.random_seed)
        self.subtasks_per_unit = self.args.blueprint.subtasks_per_unit
        self.conversation_count = self.args.blueprint.conversation_count

        if self.subtasks_per_unit <= 0:
            raise Exception(
                f'subtasks_per_unit must be greater than zero but was {self.subtasks_per_unit}'
            )

        self.raw_data = self._initialization_data_dicts

        # Load from file if needed specifying which utterances within each
        # conversation to annotate
        self.annotation_indices = None
        if self.args.blueprint.annotation_indices_jsonl:
            self.annotation_indices = []
            with open(
                self.args.blueprint.annotation_indices_jsonl, "r", encoding="utf-8-sig"
            ) as f:
                line = f.readline()
                while line:
                    conversation_indices = json.loads(line)
                    self.annotation_indices.append(conversation_indices)
                    line = f.readline()
            if len(self.annotation_indices) != len(self.raw_data):
                raise Exception(
                    f'Cannot specify a different length of annotation indices ({len(self.annotation_indices)}) than conversations ({len(self.raw_data)}).'
                )
            # TODO: should check that utterances specified are all bot
            # utterances (agent_idx == 1)

        if self.conversation_count:
            self.raw_data = self.raw_data[: self.conversation_count]
            if self.annotation_indices:
                self.annotation_indices = self.annotation_indices[
                    : self.conversation_count
                ]

        # Reorganize the self-chat data
        # self._initialization_data_dicts = self.process_data(
        #     self.raw_data, annotation_indices=self.annotation_indices
        # )

        # Now chunk the data into groups of <num_subtasks>
        grouped_data = []
        logging.info(
            f'Raw data length: {len(self.raw_data)}. self.subtasks_per_unit: {self.subtasks_per_unit}'
        )
        for i in range(0, len(self._initialization_data_dicts), self.subtasks_per_unit):
            chunk = self._initialization_data_dicts[i : i + self.subtasks_per_unit]
            grouped_data.append(chunk)
        self._initialization_data_dicts = grouped_data
        # Last group may have less unless an exact multiple
        logging.info(
            f'Grouped data into {len(self._initialization_data_dicts)} tasks with {self.subtasks_per_unit} subtasks each.'
        )

    def get_frontend_args(self) -> Dict[str, Any]:
        """
        Specifies what options within a task_config should be forwarded to the client
        for use by the task's frontend.
        """
        
        with open(self.args.blueprint.onboarding_data, "r", encoding="utf-8-sig") as f:
            onboarding_data = json.loads(f.read())

        # load the task configuration
        with open(os.path.join(get_task_path(), 'task_config/config.json')) as f:
            task_config = json.load(f)

        annotation_buckets = None
        if self.args.blueprint.annotation_buckets:
            with open(
                self.args.blueprint.annotation_buckets, "r", encoding="utf-8-sig"
            ) as f:
                annotation_buckets = json.loads(f.read())

        # combine the task configuration loaded from json with the settings
        # required by ParlAI.
        task_config.update({
            "task_description": self.args.task.get('task_description', None),
            "task_title": self.args.task.get('task_title', None),
            "annotation_question": self.args.blueprint.annotation_question,
            "onboarding_data": onboarding_data,
            "annotation_buckets": annotation_buckets,
            "ask_reason": self.args.blueprint.ask_reason,
            "response_field": self.args.blueprint.response_field,
            "frame_height": '100%',
            "num_subtasks": self.args.blueprint.subtasks_per_unit,
            "block_mobile": True
        })
        return task_config
