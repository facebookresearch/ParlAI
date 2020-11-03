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
from threading import Semaphore
from mephisto.core.registry import register_mephisto_abstraction
from mephisto.data_model.blueprint import SharedTaskState
from mephisto.server.blueprints.parlai_chat.parlai_chat_blueprint import (
    ParlAIChatBlueprint,
    ParlAIChatAgentState,
    SharedParlAITaskState,
    ParlAIChatBlueprintArgs,
)
from omegaconf import DictConfig, MISSING

if TYPE_CHECKING:
    from mephisto.data_model.task import TaskRun


def get_task_path():
    return os.path.dirname(os.path.realpath(__file__))


BLUEPRINT_TYPE = 'turn_annotations_blueprint'


@dataclass
class SharedTurnAnnotationTaskState(SharedParlAITaskState):
    shared_models: Dict[str, Any] = field(default_factory=dict)
    conversations_needed: Dict[str, Any] = field(default_factory=dict)
    run_statistics: Dict[str, int] = field(default_factory=dict)
    onboard_statistics: Dict[str, int] = field(default_factory=dict)
    generation_semaphore: Optional[Semaphore] = None


# annotations_intro => annotation_question
# task_description_path => task_description_file


@dataclass
class TurnAnnotationsBlueprintArgs(ParlAIChatBlueprintArgs):
    _blueprint_type: str = BLUEPRINT_TYPE
    _group: str = field(
        default="TurnAnnotationsBlueprint",
        metadata={
            'help': "This task runs conversations between a human and one of a set of "
            "provided models, asking workers to evaluate individual turns and "
            "the overall model quality."
        },
    )
    world_file: str = field(
        default=os.path.join(get_task_path(), 'worlds.py'),
        metadata={"help": "Path to file containing turn annotations parlai world"},
    )
    custom_source_dir: str = field(
        default=os.path.join(get_task_path(), 'webapp'),
        metadata={"help": "Path to turn annotations frontend code"},
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
    conversations_needed_string: str = field(
        default=MISSING,
        metadata={
            "help": 'Number of convos needed for each model. For example: "modelA:50,modelB:20"'
        },
    )
    task_model_parallel: bool = field(
        default=True,
        metadata={
            "help": 'Whether to load models to be used with model_parallel True.'
        },
    )
    max_resp_time: int = field(
        default=180, metadata={"help": "time limit for entering a dialog message"}
    )
    max_onboard_time: int = field(
        default=300, metadata={"help": "time limit accepting onboarding"}
    )
    base_save_folder: str = field(
        default=MISSING,
        metadata={
            "help": "Additional folder to dump crowdsourcing results (outside mephisto)"
        },
    )
    base_model_folder: str = field(
        default=MISSING, metadata={"help": "base folder for loading model files from"}
    )
    onboard_worker_answer_folder: str = field(
        default="${mephisto.blueprint.base_save_folder}/onboard_answers",
        metadata={
            "help": "base folder for saving all worker answer results during onboarding"
        },
    )
    check_acceptability: bool = field(
        default=False,
        metadata={
            "help": "Check worker's responses against several metrics of acceptability"
        },
    )
    include_persona: bool = field(
        default=False, metadata={"help": "Show persona to the bot"}
    )
    conversation_start_mode: str = field(
        default='hi',
        metadata={
            "help": 'Whether to show "Hi!" or two previous utterances (as in BlendedSkillTalk) at the beginning of the conversation',
            "choices": ['hi', 'bst'],
        },
    )
    context_seed: int = field(
        default=MISSING,
        metadata={"help": "Set seed for pulling the context info (for testing)"},
    )
    task_config_path: str = field(
        default=os.path.join(get_task_path(), 'task_config'),
        metadata={"help": "Base path to pull task configuration information"},
    )
    task_description_file: str = field(
        default="${mephisto.blueprint.task_config_path}/task_description.html",
        metadata={"help": "Path to file of HTML to show on the task-description page"},
    )
    left_pane_text_path: str = field(
        default="${mephisto.blueprint.task_config_path}/left_pane_text.html",
        metadata={
            "help": "Path to file of HTML to show on the left-hand pane of the chat window"
        },
    )
    annotations_config_path: str = field(
        default="${mephisto.blueprint.task_config_path}/annotations_config.json",
        metadata={"help": "Path to JSON of annotation categories"},
    )
    onboard_task_data_path: str = field(
        default="${mephisto.blueprint.task_config_path}/onboard_task_data.json",
        metadata={"help": "Path to JSON containing settings for running onboarding"},
    )
    final_rating_question: str = field(
        default='Please rate your partner on a scale of 1-5.',
        metadata={"help": "Text to show when asking worker to make their final rating"},
    )
    max_concurrent_responses: int = field(
        default=1,
        metadata={"help": "Limit on the number of models that can generate at once"},
    )


@register_mephisto_abstraction()
class TurnAnnotationsBlueprint(ParlAIChatBlueprint):
    """
    This Blueprint uses somewhat specialized arguments for Turn Annotations, 
    manages their validation, and also has specialized data storage for the 
    result format.

    It also has options for the onboarding data answers and the annotation bucket
    definitions.
    """

    ArgsClass = TurnAnnotationsBlueprintArgs
    SharedStateClass = SharedTurnAnnotationsTaskState
    BLUEPRINT_TYPE = BLUEPRINT_TYPE

    @classmethod
    def assert_task_args(
        cls, args: "DictConfig", shared_state: "SharedTaskState"
    ) -> None:
        """Ensure that arguments are properly configured to launch this task"""
        super().assert_task_args(args, shared_state)
        assert args.blueprint.get('conversations_needed_string', None) is not None, (
            "Must provide a string of needed conversations per model",
        )
        try:
            conversations_needed = {}
            for part in parts:
                model_name, num_string = part.split(':')
                conversations_needed[model_name] = int(num_string)
        except Exception as e:
            raise Exception(
                "Could not create conversations needed dict from given string. "
                f"Error was {e}.\n"
                "Be sure the format is like modelA:50,modelB:20"
            )
        assert (
            args.blueprint.get("task_description_file", None) is not None
        ), "Must provide a task description file"
        full_path = os.path.expanduser(args.blueprint.task_description_file)
        assert os.path.exists(
            full_path
        ), f"Target task description path {full_path} doesn't exist"

        assert (
            args.blueprint.get("left_pane_text_path", None) is not None
        ), "Must provide a left pane text file"
        full_path = os.path.expanduser(args.blueprint.left_pane_text_path)
        assert os.path.exists(
            full_path
        ), f"Target left pane text path {full_path} doesn't exist"

        assert (
            args.blueprint.get("annotations_config_path", None) is not None
        ), "Must provide an annotation config file"
        full_path = os.path.expanduser(args.blueprint.annotations_config_path)
        assert os.path.exists(
            full_path
        ), f"Target annotation config path {full_path} doesn't exist"

        assert (
            args.blueprint.get("onboard_task_data_path", None) is not None
        ), "Must provide an onboarding data file"
        full_path = os.path.expanduser(args.blueprint.onboard_task_data_path)
        assert os.path.exists(
            full_path
        ), f"Target onboarding data path {full_path} doesn't exist"

    def __init__(
        self, task_run: "TaskRun", args: "DictConfig", shared_state: "SharedTaskState"
    ):
        # Set the number of conversations needed
        conversations_needed_string = args.blueprint.conversations_needed_string
        parts = conversations_needed_string.split(',')
        conversations_needed = {}
        tot_conversations = 0
        for part in parts:
            model_name, num_string = part.split(':')
            conversations_needed[model_name] = int(num_string)
            tot_conversations += int(num_string)
        self.conversations_needed = conversations_needed
        shared_state.conversations_needed = conversations_needed
        args.blueprint.num_conversations = tot_conversations

        # Default conve
        super().__init__(task_run, args=args, shared_state=shared_state)
        random.seed(self.args.blueprint.random_seed)
        np.random.seed(self.args.blueprint.random_seed)

        # Load task configuration data beyond the task desscription, as the super does that
        left_pane_path = os.path.expanduser(args.blueprint.left_pane_text_path)
        with open(left_pane_path, "r") as left_pane_file:
            self.left_pane_text = left_pane_file.read()
        annotations_config_path = os.path.expanduser(
            args.blueprint.annotations_config_path
        )
        with open(annotations_config_path, "r") as annotations_config_file:
            self.annotations_config = annotations_config_file.read()
        onboard_task_data_path = os.path.expanduser(
            args.blueprint.onboard_task_data_path
        )
        with open(onboard_task_data_path, "r") as onboard_task_data_file:
            self.onboard_task_data = onboard_task_data_file.read()

        run_statistics = copy.deepcopy(self.conversations_needed)
        run_statistics = {r: 0 for (r, v) in run_statistics.items()}
        onboard_statistics = {}

    def get_frontend_args(self) -> Dict[str, Any]:
        """
        Specifies what options within a task_config should be forwarded to the client
        for use by the task's frontend.
        """

        with open(self.args.blueprint.onboarding_data, "r", encoding="utf-8-sig") as f:
            onboarding_data = json.loads(f.read())

        with open(
            self.args.blueprint.annotation_buckets, "r", encoding="utf-8-sig"
        ) as f:
            annotation_buckets = json.loads(f.read())

        return {
            "task_description": self.args.task.get('task_description', None),
            "task_title": self.args.task.get('task_title', None),
            "annotation_question": self.args.blueprint.annotation_question,
            "onboarding_data": onboarding_data,
            "annotation_buckets": annotation_buckets,
            "ask_reason": self.args.blueprint.ask_reason,
            "frame_height": '100%',
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
            else:
                processed_dialog = self._process_conversation(
                    d, annotation_indices=None
                )
                output.append(processed_dialog)
        print(
            f'Processed {len(data_dicts)} total conversations into {len(output)} conversations to be used in crowdsourcing task with {total_annotation_count} total annotations.'
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
            if 'persona' not in full_turn[0]['text']:
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
            if 'persona' not in full_turn[1]['text']:
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
        return new_dialogue
