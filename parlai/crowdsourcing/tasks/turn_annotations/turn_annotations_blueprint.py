#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
from dataclasses import dataclass, field
from threading import Semaphore, Condition
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
from mephisto.operations.registry import register_mephisto_abstraction
from mephisto.abstractions.blueprint import SharedTaskState
from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    ParlAIChatBlueprint,
    SharedParlAITaskState,
    ParlAIChatBlueprintArgs,
)
from omegaconf import DictConfig, MISSING

from parlai.core.params import ParlaiParser
from parlai.crowdsourcing.tasks.turn_annotations.bot_agent import TurkLikeAgent
from parlai.tasks.blended_skill_talk.agents import ContextGenerator

if TYPE_CHECKING:
    from mephisto.data_model.task import TaskRun


def get_task_path():
    return os.path.dirname(os.path.realpath(__file__))


BLUEPRINT_TYPE = 'turn_annotations_blueprint'


@dataclass
class SharedTurnAnnotationsTaskState(SharedParlAITaskState):
    shared_models: Dict[str, Any] = field(default_factory=dict)
    conversations_needed: Dict[str, Any] = field(default_factory=dict)
    run_statistics: Dict[str, int] = field(default_factory=dict)
    onboard_statistics: Dict[str, int] = field(default_factory=dict)
    statistics_condition: Optional[Condition] = None
    generation_semaphore: Optional[Semaphore] = None
    context_generator: Optional[Any] = None


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
        default=os.path.join(get_task_path(), 'frontend'),
        metadata={"help": "Path to turn annotations frontend code"},
    )
    num_turns: int = field(default=6, metadata={"help": 'minimum number of turns'})
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
    base_model_folder: str = field(
        default=MISSING, metadata={"help": "base folder for loading model files from"}
    )
    chat_data_folder: str = field(
        default=MISSING,
        metadata={"help": "Folder in which to save collected conversation data"},
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
    override_opt: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Additional args to pass to initialize the models and persona generator "
            "in order to override the parlai parser defaults."
        },
    )


@register_mephisto_abstraction()
class TurnAnnotationsBlueprint(ParlAIChatBlueprint):
    """
    This Blueprint uses somewhat specialized arguments for Turn Annotations, manages
    their validation, and also has specialized data storage for the result format.

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
        """
        Ensure that arguments are properly configured to launch this task.
        """
        if len(shared_state.conversations_needed) == 0:
            assert (
                args.blueprint.get('conversations_needed_string', None) is not None
            ), (
                "Must provide a string of needed conversations per model if not providing "
                "a conversations needed dict"
            )
            try:
                conversations_needed = {}
                parts = args.blueprint.conversations_needed_string.split(',')
                for part in parts:
                    model_name, num_string = part.split(':')
                    conversations_needed[model_name] = int(num_string)
            except Exception as e:
                raise Exception(
                    "Could not create conversations needed dict from given string. "
                    f"Error was {e}.\n"
                    "Be sure the format is like modelA:50,modelB:20"
                )
        else:
            conversations_needed = shared_state.conversations_needed
        args.blueprint.num_conversations = sum(conversations_needed.values())
        super().assert_task_args(args, shared_state)
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
        conversations_needed = {}
        parts = conversations_needed_string.split(',')
        for part in parts:
            model_name, num_string = part.split(':')
            conversations_needed[model_name] = int(num_string)
        self.conversations_needed = conversations_needed
        shared_state.conversations_needed = conversations_needed
        args.blueprint.num_conversations = sum(conversations_needed.values())

        # Default conversation initialization
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
            self.onboard_task_data = json.load(onboard_task_data_file)

        run_statistics = {r: 0 for (r, v) in self.conversations_needed.items()}
        shared_state.run_statistics = run_statistics

        # Initialize models
        models_needed = list(conversations_needed.keys())
        active_models = [m for m in models_needed if conversations_needed[m] > 0]
        shared_bot_agents = TurkLikeAgent.get_bot_agents(args, active_models)
        shared_state.shared_models = shared_bot_agents

        # Context need parlai options
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        if len(args.blueprint.override_opt) > 0:
            argparser.set_params(**args.blueprint.override_opt)
        opt = argparser.parse_args([])

        if (
            args.blueprint.include_persona
            or args.blueprint.conversation_start_mode == 'bst'
        ):
            context_generator = ContextGenerator(opt, datatype='test', seed=0)
            # We pull from the test set so that the model can't regurgitate
            # memorized conversations
        else:
            context_generator = None
        shared_state.context_generator = context_generator

        # Limits the number of models that can generate at once
        max_concurrent_responses = 1
        semaphore = Semaphore(max_concurrent_responses)

        # Lock for editing run statistics between threads
        statistics_condition = Condition()

        # Move shared state into the world and onboarding opts, such that these
        # can be used by the worlds
        shared_state.onboarding_world_opt.update(
            {
                'onboard_statistics': shared_state.onboard_statistics,
                'statistics_condition': statistics_condition,
                'max_onboard_time': args.blueprint.max_onboard_time,
                'onboard_task_data': self.onboard_task_data,
                'onboarding_qualification': args.blueprint.onboarding_qualification,
            }
        )
        shared_state.world_opt.update(
            {
                'annotations_config': self.annotations_config,
                'block_qualification': args.blueprint.block_qualification,
                'conversations_needed': conversations_needed,
                'run_statistics': shared_state.run_statistics,
                'context_generator': context_generator,
                'semaphore': semaphore,
                'shared_bot_agents': shared_bot_agents,
                'num_turns': args.blueprint.num_turns,
                'max_resp_time': args.blueprint.max_resp_time,
                'is_sandbox': args.provider.requester_name == 'MOCK_REQUESTER',
                'statistics_condition': statistics_condition,
                'check_acceptability': args.blueprint.check_acceptability,
                'include_persona': args.blueprint.include_persona,
                'conversation_start_mode': args.blueprint.conversation_start_mode,
                'chat_data_folder': args.blueprint.chat_data_folder,
            }
        )

    def get_frontend_args(self) -> Dict[str, Any]:
        """
        Specifies what options within a task_config should be forwarded to the client
        for use by the task's frontend.
        """
        with open(
            self.args.blueprint.annotations_config_path, "r", encoding="utf-8-sig"
        ) as f:
            annotation_buckets = json.loads(f.read())

        return {
            "min_num_turns": self.args.blueprint.num_turns,
            "task_description": self.full_task_description,
            "task_title": self.args.task.get('task_title', None),
            "annotation_question": self.args.blueprint.annotation_question,
            "annotation_buckets": annotation_buckets,
            "onboarding_data": self.onboard_task_data,
            "left_pane_text": self.left_pane_text,
            "frame_height": '650px',
            "final_rating_question": self.args.blueprint.final_rating_question,
            "block_mobile": True,
        }
