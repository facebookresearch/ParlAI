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
    annotation_last_only: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If you only want to annotate the last utterance of each conversation. Cannot be used with annotation_indices_jsonl."
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
    annotations_config_path: str = field(
        default="",
        metadata={
            "help": "As per Turn Annotations task, path to annotation buckets which will be checkboxes in the frontend for worker to annotate an utterance. Set to "
            " to disable checkboxes."
        },
    )
    response_field: bool = field(
        default=False,
        metadata={
            "help": "If we want a freeform textbox input for the crowdworker to respond to the message."
        },
    )
    task_description_file: str = field(
        default=os.path.join(get_task_path(), 'task_config/task_description.html'),
        metadata={"help": "Path to file of HTML to show on the task-description page"},
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
        self.annotation_last_only = self.args.blueprint.annotation_last_only

        if self.subtasks_per_unit <= 0:
            raise Exception(
                f'subtasks_per_unit must be greater than zero but was {self.subtasks_per_unit}'
            )

        self.raw_data = self._initialization_data_dicts

        # Load from file if needed specifying which utterances within each
        # conversation to annotate
        self.annotation_indices = None
        if self.args.blueprint.annotation_indices_jsonl:
            if self.annotation_last_only:
                raise RuntimeError(
                    'Cannot use flag annotation_last_only and supply a file with annotation indices.'
                )
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
        self._initialization_data_dicts = self.process_data(
            self.raw_data, annotation_indices=self.annotation_indices
        )

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

        # Load task description from file
        task_description = "<h1>" "You didn't specify a task_description_file" "</h1>"
        if self.args.blueprint.get("task_description_file", None) is not None:
            full_path = os.path.expanduser(self.args.blueprint.task_description_file)
            assert os.path.exists(
                full_path
            ), f"Target task description path {full_path} doesn't exist"
            with open(full_path, "r") as description_fp:
                task_description = description_fp.read()

        with open(self.args.blueprint.onboarding_data, "r", encoding="utf-8-sig") as f:
            onboarding_data = json.loads(f.read())

        annotation_buckets = None
        if self.args.blueprint.get('annotations_config_path', ''):
            with open(
                self.args.blueprint.annotations_config_path, "r", encoding="utf-8-sig"
            ) as f:
                annotation_buckets = json.loads(f.read())

        return {
            "task_description": task_description,
            "task_title": self.args.task.get('task_title', None),
            "annotation_question": self.args.blueprint.annotation_question,
            "onboarding_data": onboarding_data,
            "annotation_buckets": annotation_buckets,
            "ask_reason": self.args.blueprint.ask_reason,
            "response_field": self.args.blueprint.response_field,
            "frame_height": 0,
            "num_subtasks": self.args.blueprint.subtasks_per_unit,
            "block_mobile": True,
        }

    def process_data(self, data_dicts, annotation_indices=None):
        """
        Override this in a subclass if you want to change how data is processed from
        input file before being sent to the frontend.
        """
        output = []
        total_annotation_count = 0
        for conv_idx, d in enumerate(data_dicts):
            if annotation_indices:
                total_annotation_count += len(annotation_indices[conv_idx])
                # We only want to show the conversation up to the last
                # utterance we need annotations on, b/c otherwise may confuse
                # or bias the turkers
                if len(annotation_indices[conv_idx]) > 1:
                    logging.info(
                        f'Splitting {len(annotation_indices[conv_idx])} separate problematic utterance annotations in the same conversation into two separate conversations for this task. This avoids biasing the turkers with utterances that may come after one of the annotations.'
                    )
                for a in annotation_indices[conv_idx]:
                    processed_dialog = self._process_conversation(d, [a])
                    output.append(processed_dialog)
            elif self.annotation_last_only:
                # Annotate only the last utterance
                last_idx = len(d['dialog']) * 2 - 1
                processed_dialog = self._process_conversation(
                    d, annotation_indices=[last_idx]
                )
                total_annotation_count += 1
                output.append(processed_dialog)
            else:
                # Annotate all the model utterances
                total_annotation_count += len(d['dialog']) / 2
                processed_dialog = self._process_conversation(
                    d, annotation_indices=None
                )
                output.append(processed_dialog)

        print(
            f'process_data: Processed {len(data_dicts)} total conversations into {len(output)} conversations in the full data with {total_annotation_count} total turn annotations. (Does not account for units per assignment value - i.e. multiple annotations.)'
        )

        np.random.shuffle(output)
        return output

    def _process_conversation(self, d, annotation_indices: Optional[List[int]] = None):
        """
        Helper function for processing conversations.

        :param annotation_indices:
            Array of turn indices to annotate of the
            actual conversation not including the context [So 0 is the "Hi!" if
            that's the first non-context utterance of the conversation.] If this is not
            specified, annotate all bot turns.
        :return: modified dialogue object
        """
        new_dialogue = []
        if annotation_indices is not None:
            max_turn_to_show = max(annotation_indices)
        else:
            max_turn_to_show = None
        adjusted_turn_idx = 0
        for full_turn in d['dialog']:
            if len(full_turn) != 2:
                print(
                    f'Warning! Skipping incomplete conversation! full_turn was: {full_turn}'
                )
                continue
            if max_turn_to_show is not None and adjusted_turn_idx > max_turn_to_show:
                logging.info(
                    f'Skipping {adjusted_turn_idx}th utterance, b/c max_turn_to_show was {max_turn_to_show}.'
                )
                continue
            # If there is a persona, which is context as in the ConvAI2
            # task, we don't want to display the persona utterances
            # Persona strings have the format "your persona:"
            if 'your persona:' not in full_turn[0]['text']:
                do_annotate = False
                new_dialogue.append(
                    {
                        'text': full_turn[0]['text'],
                        'agent_idx': 0,
                        'do_annotate': do_annotate,
                        'other_metadata': full_turn[0].get('other_metadata'),
                    }
                )
                adjusted_turn_idx += 1
            if 'your persona:' not in full_turn[1]['text']:
                if annotation_indices:
                    do_annotate = adjusted_turn_idx in annotation_indices
                else:
                    do_annotate = True
                    # Default to annotating all bot utterances
                new_dialogue.append(
                    {
                        'text': full_turn[1]['text'],
                        'agent_idx': 1,
                        'do_annotate': do_annotate,
                        'other_metadata': full_turn[1].get('other_metadata'),
                    }
                )
                adjusted_turn_idx += 1
        if max_turn_to_show is not None and adjusted_turn_idx < max_turn_to_show:
            raise Exception(
                f'Conversation had {adjusted_turn_idx} but max_turn_to_show was {max_turn_to_show}'
            )
        assert any(
            nd['do_annotate'] for nd in new_dialogue
        ), f'Have to annotate at least one index in the conversation! But new_dialogue was: {new_dialogue}, raw dialogue was: {d["dialog"]}, annotation_indices was: {annotation_indices}, length of dialogue was {len(new_dialogue)}, adjusted_turn_idx was: {adjusted_turn_idx}, max_turn_to_show: {max_turn_to_show}'

        return new_dialogue


@dataclass
class TurnAnnotationsStaticInFlightQABlueprintArgs(TurnAnnotationsStaticBlueprintArgs):
    _blueprint_type: str = STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE
    _group: str = field(
        default="TurnAnnotationsStaticInFlightQABlueprint",
        metadata={
            'help': """This task mixes in a live onboarding as the last subtask (in addition to an onboarding at the start), and actually increases the number of subtasks per unit by 1."""
        },
    )
    onboarding_in_flight_data: str = field(
        default=os.path.join(get_task_path(), 'task_config/onboarding_in_flight.jsonl'),
        metadata={
            "help": "Path to data and answers for onboarding task in JSON-L format (one JSON object per line per onboarding)"
        },
    )


@register_mephisto_abstraction()
class TurnAnnotationsStaticInFlightQABlueprint(TurnAnnotationsStaticBlueprint):
    """
    This Blueprint mixes in a live onboarding as the last subtask (in addition to an
    onboarding at the start), and actually increases the number of subtasks per unit by
    1.
    """

    ArgsClass = TurnAnnotationsStaticInFlightQABlueprintArgs
    BLUEPRINT_TYPE = STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE

    def __init__(
        self, task_run: "TaskRun", args: "DictConfig", shared_state: "SharedTaskState"
    ):
        super().__init__(task_run, args=args, shared_state=shared_state)

        raw_qc_convos = []
        with open(self.args.blueprint.onboarding_in_flight_data, "r") as f:
            line = f.readline()
            while line:
                qc_convo = json.loads(line)
                raw_qc_convos.append(qc_convo)
                line = f.readline()
        # Annotate all the utterances in the quality controls
        # So annotation_indices=None here
        self.quality_control_convos = self.process_data(
            raw_qc_convos, annotation_indices=None
        )

        # Re-chunk the data to add a quality control convo as the last subtask
        # No shuffling of the data for reproducibility's sake
        # (quality control will always be last subtask)
        # TODO: I don't think we need to re-chunk this actually; just iterate
        # over the data and add the quality control task
        all_data = []
        for grp in self._initialization_data_dicts:
            all_data.extend(grp)

        grouped_data = []
        number_of_tasks = math.floor(len(all_data) / self.subtasks_per_unit)
        for i in range(0, number_of_tasks):
            data_index = i * self.subtasks_per_unit
            chunk = all_data[data_index : data_index + self.subtasks_per_unit]
            qc_convo_idx = i % len(self.quality_control_convos)
            chunk.append(self.quality_control_convos[qc_convo_idx])
            grouped_data.append(chunk)

        self._initialization_data_dicts = grouped_data
        self.subtasks_per_unit = len(chunk)

        print(
            f'{self.__class__.__name__}: Grouped data into {len(self._initialization_data_dicts)} tasks with {self.subtasks_per_unit} subtasks each (added in-flight qualification task).'
        )
