#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from mephisto.core.registry import register_mephisto_abstraction
from mephisto.server.blueprints.static_react_task.static_react_blueprint import (
    StaticReactBlueprint,
)
from typing import Any, Dict, TYPE_CHECKING
from argparse import _ArgumentGroup as ArgumentGroup

if TYPE_CHECKING:
    from mephisto.data_model.task import TaskRun


@register_mephisto_abstraction()
class TurnAnnotationsStaticBlueprint(StaticReactBlueprint):
    """
    This Blueprint has a subtasks number option to combine multiple conversations into "sub-HITs". It also has options for the onboarding data answers and the annotation bucket definitions.
    """

    BLUEPRINT_TYPE = 'turn_annotations_static_blueprint'

    def __init__(self, task_run: "TaskRun", opts: Any):
        super().__init__(task_run, opts)
        self.subtasks_per_unit = opts['subtasks_per_unit']

        if self.subtasks_per_unit <= 0:
            raise Exception(
                f'subtasks-per-unit must be greater than zero but was {self.subtasks_per_unit}'
            )
        grouped_data = []

        # Reorganize the self-chat data
        self._initialization_data_dicts = self.process_data(
            self._initialization_data_dicts
        )

        # Now chunk the data into groups of <num_subtasks>
        logging.info(
            f'Raw data length: {len(self._initialization_data_dicts)}. self.subtasks_per_unit: {self.subtasks_per_unit}'
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
            "--subtasks-per-unit",
            dest="subtasks_per_unit",
            type=int,
            default=-1,
            help="number of subtasks/comparisons to do per unit",
        )
        group.add_argument(
            "--annotate-last-utterance-only",
            dest="annotate_last_utterance_only",
            type=bool,
            default=False,
            help="If we only want the crowdworker to annotate the last utterance in the conversation",
        )
        group.add_argument(
            "--onboarding-data",
            dest="onboarding_data",
            type=str,
            default='./json/onboarding.json',
            help="Path to data and answers for onboarding task in JSON format",
        )
        group.add_argument(
            "--annotation-buckets",
            dest="annotation_buckets",
            type=str,
            default='./json/annotation_buckets.json',
            help="As per Turn Annotations task, path to annotation buckets which will be checkboxes for worker to annotate an utterance.",
        )

    def get_frontend_args(self) -> Dict[str, Any]:
        """
        Specifies what options within a task_config should be forwarded to the client for use by the task's frontend.
        """
        with open(self.opts['onboarding_data'], "r", encoding="utf-8-sig") as f:
            onboarding_data = json.loads(f.read())

        with open(self.opts['annotation_buckets'], "r", encoding="utf-8-sig") as f:
            annotation_buckets = json.loads(f.read())

        return {
            "task_description": self.opts['task_description'],
            "task_title": self.opts['task_title'],
            "onboarding_data": onboarding_data,
            "annotation_buckets": annotation_buckets,
            "annotate_last_utterance_only": self.opts['annotate_last_utterance_only'],
            "frame_height": '100%',
            "num_subtasks": self.opts["subtasks_per_unit"],
            "block_mobile": True,
        }

    def process_data(self, data_dicts):
        """
        Override this in a subclass if you want to change how data is processed from input file before being sent to the frontend.
        """
        output = []
        for d in data_dicts:
            new_dialogue = []
            for utt in d['dialog']:
                # If there is a persona, which is context as in the ConvAI2
                # task, we don't want to display the persona utterances
                if 'persona' not in utt[0]['text']:
                    new_dialogue.append({'text': utt[0]['text'], 'agent_idx': 0})
                if 'persona' not in utt[1]['text']:
                    new_dialogue.append({'text': utt[1]['text'], 'agent_idx': 1})
            output.append(new_dialogue)
        return output


@register_mephisto_abstraction()
class TurnAnnotationsStaticInFlightQABlueprint(TurnAnnotationsStaticBlueprint):
    """
    This Blueprint mixes in a live onboarding as the last subtask (in addition to an onboarding at the start), and actually
    increases the number of subtasks per unit by 1.
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
        self.quality_control_convos = self.process_data(raw_qc_convos)

        # No shuffling of the data for reproducibility's sake
        # (quality control will always be last subtask)
        all_data = []
        for grp in self._initialization_data_dicts:
            all_data.extend(grp)

        grouped_data = []
        for i in range(0, len(all_data), self.subtasks_per_unit):
            chunk = all_data[i : i + self.subtasks_per_unit]
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
            default='./json/onboarding_in_flight.jsonl',
            help="Path to data and answers for onboarding task in JSON-L format (one JSON object per line per onboarding)",
        )
