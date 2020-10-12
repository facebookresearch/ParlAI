#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import numpy as np
import math
import json
import logging
from mephisto.core.registry import register_mephisto_abstraction
from mephisto.server.blueprints.static_react_task.static_react_blueprint import (
    StaticReactBlueprint,
)
from mephisto.core.argparse_parser import str2bool
from typing import Any, Dict, TYPE_CHECKING
from argparse import _ArgumentGroup as ArgumentGroup

if TYPE_CHECKING:
    from mephisto.data_model.task import TaskRun


def get_task_path():
    return os.path.dirname(__file__)


@register_mephisto_abstraction()
class TurnAnnotationsStaticBlueprint(StaticReactBlueprint):
    """
    This Blueprint has a subtasks number option to combine multiple conversations into
    "sub-HITs".

    It also has options for the onboarding data answers and the annotation bucket
    definitions.
    """

    BLUEPRINT_TYPE = 'turn_annotations_static_blueprint'

    def __init__(self, task_run: "TaskRun", opts: Any):
        super().__init__(task_run, opts)
        print(f'Running {self.__class__.__name__} with opts: {self.opts}')
        random.seed(self.opts["random_seed"])
        np.random.seed(self.opts["random_seed"])
        self.subtasks_per_unit = self.opts['subtasks_per_unit']
        self.conversation_count = self.opts['conversation_count']

        if self.subtasks_per_unit <= 0:
            raise Exception(
                f'subtasks-per-unit must be greater than zero but was {self.subtasks_per_unit}'
            )

        self.raw_data = self._initialization_data_dicts

        # Load from file if needed specifying which utterances within each
        # conversation to annotate
        self.annotation_indices = None
        if self.opts['annotation_indices_jsonl']:
            self.annotation_indices = []
            with open(
                self.opts['annotation_indices_jsonl'], "r", encoding="utf-8-sig"
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

    @classmethod
    def add_args_to_group(cls, group: "ArgumentGroup") -> None:
        """
        Adds required options for TurnAnnotationStaticBlueprint.
        """
        super().add_args_to_group(group)
        group.add_argument(
            "--random-seed",
            dest="random_seed",
            type=int,
            default=42,
            help="seed for random",
        )
        group.add_argument(
            "--annotation-question",
            dest="annotation_question",
            type=str,
            default='Does this comment require any annotations? (Check all that apply)',
            help="The string displayed above the checkboxes for each annotation in the task.",
        )
        group.add_argument(
            "--subtasks-per-unit",
            dest="subtasks_per_unit",
            type=int,
            default=-1,
            help="number of subtasks/comparisons to do per unit",
        )
        group.add_argument(
            "--annotation-indices-jsonl",
            dest="annotation_indices_jsonl",
            type=str,
            default=None,
            help="Specify which utterance indices to annotate per conversation in a JSONL file. Must be same length as conversations data-jsonl file. See example file in task_config/annotation_indices_example.jsonl",
        )
        group.add_argument(
            "--ask-reason",
            dest="ask_reason",
            type=str2bool,  # Need to handle it being 'False' in arg_string
            default=False,
            help="If we want to ask the crowdworker for a reason for each of their annotations in a text field",
        )
        group.add_argument(
            "--conversation-count",
            dest="conversation_count",
            type=int,
            default=None,
            help="Specify a positive integer if you want to use only the first N conversations in the data file",
        )
        group.add_argument(
            "--onboarding-data",
            dest="onboarding_data",
            type=str,
            default=os.path.join(get_task_path(), 'task_config/onboarding.json'),
            help="Path to data and answers for onboarding task in JSON format",
        )
        group.add_argument(
            "--annotation-buckets",
            dest="annotation_buckets",
            type=str,
            default=os.path.join(
                get_task_path(), 'task_config/annotation_buckets.json'
            ),
            help="As per Turn Annotations task, path to annotation buckets which will be checkboxes in the frontend for worker to annotate an utterance.",
        )

    def get_frontend_args(self) -> Dict[str, Any]:
        """
        Specifies what options within a task_config should be forwarded to the client
        for use by the task's frontend.
        """

        with open(self.opts['onboarding_data'], "r", encoding="utf-8-sig") as f:
            onboarding_data = json.loads(f.read())

        with open(self.opts['annotation_buckets'], "r", encoding="utf-8-sig") as f:
            annotation_buckets = json.loads(f.read())

        return {
            "task_description": self.opts['task_description'],
            "task_title": self.opts['task_title'],
            "annotation_question": self.opts['annotation_question'],
            "onboarding_data": onboarding_data,
            "annotation_buckets": annotation_buckets,
            "ask_reason": self.opts['ask_reason'],
            "frame_height": '100%',
            "num_subtasks": self.opts["subtasks_per_unit"],
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
            max_turn_to_show = len(d['dialog']) - 1
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
            else:
                processed_dialog = self._process_conversation(d, [max_turn_to_show])
                output.append(processed_dialog)
        print(
            f'Processed {len(data_dicts)} total conversations into {len(output)} conversations to be used in crowdsourcing task with {total_annotation_count} total annotations.'
        )
        np.random.shuffle(output)
        return output

    def _process_conversation(self, d, annotation_indices):
        """
        Helper function for processing conversations.

        :param annotation_indices:
            Array of turn indices to annotate of the
            actual conversation not including the context [So 0 is the "Hi!" if
            that's the first non-context utterance of the conversation.]
        :return: modified dialogue object
        """
        new_dialogue = []
        max_turn_to_show = max(annotation_indices)
        adjusted_turn_idx = 0
        for full_turn in d['dialog']:
            if len(full_turn) != 2:
                print(
                    f'Warning! Skipping incomplete conversation! full_turn was: {full_turn}'
                )
                continue
            if adjusted_turn_idx > max_turn_to_show:
                logging.info(
                    f'Skipping {adjusted_turn_idx}th utterance, b/c max_turn_to_show was {max_turn_to_show}.'
                )
                continue
            # If there is a persona, which is context as in the ConvAI2
            # task, we don't want to display the persona utterances
            if 'persona' not in full_turn[0]['text']:
                do_annotate = False
                new_dialogue.append(
                    {
                        'text': full_turn[0]['text'],
                        'agent_idx': 0,
                        'do_annotate': do_annotate,
                        'other_metadata': full_turn[0]['other_metadata'],
                    }
                )
                adjusted_turn_idx += 1
            if 'persona' not in full_turn[1]['text']:
                do_annotate = True
                if annotation_indices:
                    do_annotate = adjusted_turn_idx in annotation_indices
                new_dialogue.append(
                    {
                        'text': full_turn[1]['text'],
                        'agent_idx': 1,
                        'do_annotate': do_annotate,
                        'other_metadata': full_turn[1]['other_metadata'],
                    }
                )
                adjusted_turn_idx += 1
        if adjusted_turn_idx < max_turn_to_show:
            raise Exception(
                f'Conversation had {adjusted_turn_idx} but max_turn_to_show was {max_turn_to_show}'
            )
        return new_dialogue


@register_mephisto_abstraction()
class TurnAnnotationsStaticInFlightQABlueprint(TurnAnnotationsStaticBlueprint):
    """
    This Blueprint mixes in a live onboarding as the last subtask (in addition to an
    onboarding at the start), and actually increases the number of subtasks per unit by
    1.
    """

    BLUEPRINT_TYPE = 'turn_annotations_static_inflight_qa_blueprint'

    def __init__(self, task_run: "TaskRun", opts: Any):
        super().__init__(task_run, opts)

        raw_qc_convos = []
        with open(self.opts['onboarding_in_flight_data'], "r") as f:
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

    @classmethod
    def add_args_to_group(cls, group: "ArgumentGroup") -> None:
        """
        Adds required options for TurnAnnotationsStaticInFlightQABlueprint.
        """
        super().add_args_to_group(group)
        group.add_argument(
            "--onboarding-in-flight-data",
            dest="onboarding_in_flight_data",
            type=str,
            default=os.path.join(
                get_task_path(), 'task_config/onboarding_in_flight.jsonl'
            ),
            help="Path to data and answers for onboarding task in JSON-L format (one JSON object per line per onboarding)",
        )
