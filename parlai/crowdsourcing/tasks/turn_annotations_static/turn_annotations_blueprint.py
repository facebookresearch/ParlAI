#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import logging
from argparse import _ArgumentGroup as ArgumentGroup
from typing import Any, Dict, TYPE_CHECKING

from mephisto.core.registry import register_mephisto_abstraction
from mephisto.data_model.blueprint import SharedTaskState
from mephisto.server.blueprints.static_react_task.static_react_blueprint import (
    StaticReactBlueprint,
)
from omegaconf import MISSING, DictConfig

if TYPE_CHECKING:
    from mephisto.data_model.task import TaskRun


def get_task_path():
    return os.path.dirname(__file__)


STATIC_BLUEPRINT_TYPE = 'turn_annotations_static_blueprint'
STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE = 'turn_annotations_static_inflight_qa_blueprint'


@dataclass
class TurnAnnotationsStaticBlueprintArgs(StaticReactBlueprintArgs):
    _blueprint_type: str = STATIC_BLUEPRINT_TYPE


#     _group: str = field(
#         default="AcuteEvalBlueprint",
#         metadata={
#             'help': """\
# Tasks launched from the ACUTE-Eval blueprint require sets of pairings for workers to \
# be able to compare to. These pairings should be provided as a .jsonl with \
# --pairings-filepath."""
#         },
#     )
#     annotations_per_pair: int = field(
#         default=1,
#         metadata={"help": "Number of annotations per conversation comparison pair"},
#     )
#     pairings_filepath: str = field(
#         default=MISSING,
#         metadata={"help": "path to the file containing the task dictionaries"},
#     )
#     s1_choice: str = field(
#         default="I would prefer to talk to <Speaker 1>",
#         metadata={"help": "text next to speaker 1 radio button"},
#     )
#     s2_choice: str = field(
#         default="I would prefer to talk to <Speaker 2>",
#         metadata={"help": "text next to speaker 2 radio button"},
#     )
#     eval_question: str = field(
#         default="Who would you prefer to talk to for a long conversation?",
#         metadata={
#             "help": 'question to present to turker for comparison (e.g. "Which speaker is better?")'
#         },
#     )
#     block_on_onboarding_fail: bool = field(
#         default=True, metadata={"help": "whether to block on onboarding failure"}
#     )
#     num_matchup_pairs: int = field(
#         default=2, metadata={"help": "Number of pairs per model matchup, default 2"}
#     )
#     subtasks_per_unit: int = field(
#         default=5, metadata={"help": "number of subtasks/comparisons to do per unit"}
#     )
#     onboarding_threshold: float = field(
#         default=0.75,
#         metadata={"help": "minimum accuracy on onboarding tasks, as a float 0-1.0"},
#     )
#     random_seed: int = field(default=42, metadata={"help": "seed for random"})
#     additional_task_description: str = field(
#         default='', metadata={"help": "Additional text to show on the left pane"}
#     )


@register_mephisto_abstraction()
class TurnAnnotationsStaticBlueprint(StaticReactBlueprint):
    """
    This Blueprint has a subtasks number option to combine multiple conversations into
    "sub-HITs".

    It also has options for the onboarding data answers and the annotation bucket
    definitions.
    """

    ArgsClass = TaskAnnotationsStaticBlueprintArgs
    BLUEPRINT_TYPE = STATIC_BLUEPRINT_TYPE

    def __init__(
        self, task_run: "TaskRun", args: "DictConfig", shared_state: "SharedTaskState"
    ):
        super().__init__(task_run, args=args, shared_state=shared_state)
        self.subtasks_per_unit = args.blueprint.subtasks_per_unit

        if self.subtasks_per_unit <= 0:
            raise Exception(
                f'subtasks_per_unit must be greater than zero but was {self.subtasks_per_unit}'
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
            type=str2bool,  # Need to handle it being 'False' in arg_string
            default=False,
            help="If we only want the crowdworker to annotate the last utterance in the conversation",
        )
        group.add_argument(
            "--ask-reason",
            dest="ask_reason",
            type=str2bool,  # Need to handle it being 'False' in arg_string
            default=False,
            help="If we want to ask the crowdworker for a reason for each of their annotations in a text field",
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

        with open(self.args.blueprint.onboarding_data, "r", encoding="utf-8-sig") as f:
            onboarding_data = json.loads(f.read())

        with open(self.args.blueprint.annotation_buckets, "r", encoding="utf-8-sig") as f:
            annotation_buckets = json.loads(f.read())

        return {
            "task_description": self.args.task.get('task_description', None),
            "task_title": self.args.task.get('task_title', None),
            "onboarding_data": onboarding_data,
            "annotation_buckets": annotation_buckets,
            "annotate_last_utterance_only": self.args.blueprint.annotate_last_utterance_only,
            "ask_reason": self.args.blueprint.ask_reason,
            "frame_height": '100%',
            "num_subtasks": self.args.blueprint.subtasks_per_unit,
            "block_mobile": True,
        }

    def process_data(self, data_dicts):
        """
        Override this in a subclass if you want to change how data is processed from
        input file before being sent to the frontend.
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


@dataclass
class TurnAnnotationsStaticInFlightQABlueprintArgs(TurnAnnotationsStaticBlueprintArgs):
    _blueprint_type: str = STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE


#     _group: str = field(
#         default="AcuteEvalBlueprint",
#         metadata={
#             'help': """\
# Tasks launched from the ACUTE-Eval blueprint require sets of pairings for workers to \
# be able to compare to. These pairings should be provided as a .jsonl with \
# --pairings-filepath."""
#         },
#     )
#     annotations_per_pair: int = field(
#         default=1,
#         metadata={"help": "Number of annotations per conversation comparison pair"},
#     )
#     pairings_filepath: str = field(
#         default=MISSING,
#         metadata={"help": "path to the file containing the task dictionaries"},
#     )
#     s1_choice: str = field(
#         default="I would prefer to talk to <Speaker 1>",
#         metadata={"help": "text next to speaker 1 radio button"},
#     )
#     s2_choice: str = field(
#         default="I would prefer to talk to <Speaker 2>",
#         metadata={"help": "text next to speaker 2 radio button"},
#     )
#     eval_question: str = field(
#         default="Who would you prefer to talk to for a long conversation?",
#         metadata={
#             "help": 'question to present to turker for comparison (e.g. "Which speaker is better?")'
#         },
#     )
#     block_on_onboarding_fail: bool = field(
#         default=True, metadata={"help": "whether to block on onboarding failure"}
#     )
#     num_matchup_pairs: int = field(
#         default=2, metadata={"help": "Number of pairs per model matchup, default 2"}
#     )
#     subtasks_per_unit: int = field(
#         default=5, metadata={"help": "number of subtasks/comparisons to do per unit"}
#     )
#     onboarding_threshold: float = field(
#         default=0.75,
#         metadata={"help": "minimum accuracy on onboarding tasks, as a float 0-1.0"},
#     )
#     random_seed: int = field(default=42, metadata={"help": "seed for random"})
#     additional_task_description: str = field(
#         default='', metadata={"help": "Additional text to show on the left pane"}
#     )


@register_mephisto_abstraction()
class TurnAnnotationsStaticInFlightQABlueprint(TurnAnnotationsStaticBlueprint):
    """
    This Blueprint mixes in a live onboarding as the last subtask (in addition to an
    onboarding at the start), and actually increases the number of subtasks per unit by
    1.
    """

    ArgsClass = TaskAnnotationsStaticInFlightQABlueprintArgs
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
        self.quality_control_convos = self.process_data(raw_qc_convos)

        # No shuffling of the data for reproducibility's sake
        # (quality control will always be last subtask)
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
