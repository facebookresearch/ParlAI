#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING

from mephisto.abstractions.blueprint import SharedTaskState
from mephisto.operations.registry import register_mephisto_abstraction
from omegaconf import DictConfig

from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import (
    ModelChatBlueprint,
    ModelChatBlueprintArgs,
    BaseModelChatBlueprint,
)
from parlai.crowdsourcing.tasks.pairwise_per_turn_eval.bot_agent import (
    PerTurnEvalTurkLikeAgent,
)

if TYPE_CHECKING:
    from mephisto.data_model.task import TaskRun


def get_task_path():
    return os.path.dirname(os.path.realpath(__file__))


BLUEPRINT_TYPE = 'per_turn_eval_blueprint'


@dataclass
class PerTurnEvalBlueprintArgs(ModelChatBlueprintArgs):
    _blueprint_type: str = BLUEPRINT_TYPE

    custom_source_dir: str = field(
        default=os.path.join(get_task_path(), 'frontend'),
        metadata={"help": "Path to frontend code"},
    )

    random_seed: int = field(
        default=42, metadata={"help": 'Seed for random operations'}
    )

    task_question: str = field(
        default="Which next response from your partner would be more interesting?",
        metadata={"help": "The task question to choose bot's responses based on"},
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


@register_mephisto_abstraction()
class PerTurnEvalBlueprint(ModelChatBlueprint):
    """
    Blueprint for per turn evaluation method.

    This blueprint subclasses ModelChatBlueprint to provide logic for keeping track of
    how many more conversations are needed per model; this logic is not shared with
    other blueprints.
    """

    ArgsClass = PerTurnEvalBlueprintArgs
    BLUEPRINT_TYPE = BLUEPRINT_TYPE

    @classmethod
    def assert_task_args(
        cls, args: "DictConfig", shared_state: "SharedTaskState"
    ) -> None:
        """
        Ensure that arguments are properly configured to launch this task.
        """

        if (
            not isinstance(shared_state.conversations_needed, dict)
            or len(shared_state.conversations_needed) == 0
        ):
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
                    model1, model2, num_string = part.split(':')
                    models_alphabetize = [model1, model2]
                    models_alphabetize.sort()
                    conversations_needed[
                        f'{models_alphabetize[0]}:{models_alphabetize[1]}'
                    ] = int(num_string)
            except Exception as e:
                raise Exception(
                    "Could not create conversations needed dict from given string. "
                    f"Error was {e}.\n"
                    "Be sure the format is like modelA:modelB:50,modelC:modelD:20"
                )
        else:
            conversations_needed = shared_state.conversations_needed
        args.blueprint.num_conversations = sum(conversations_needed.values())
        BaseModelChatBlueprint.assert_task_args(args=args, shared_state=shared_state)

        if args.blueprint.get("annotations_config_path", "") != "":
            # We are going to do annotations, so check for the presence of an onboarding
            # data file that will be used to onboard users into knowing how to do the
            # annotations properly
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
        conversations_needed = self._process_conversations_needed(args)
        self.conversations_needed = conversations_needed
        shared_state.conversations_needed = conversations_needed
        args.blueprint.num_conversations = sum(conversations_needed.values())

        shared_state.world_opt.update({'task_question': args.blueprint.task_question})

        super().__init__(task_run=task_run, args=args, shared_state=shared_state)

    def format_left_pane_text(self, args: "DictConfig"):
        """
        Adds the user's intended persona into the left pane of the frontend by modifying
        self.left_pane_text for code injection.
        """
        self.left_pane_text = self.left_pane_text.format(
            task_question=args.blueprint.task_question
        )

    def _process_conversations_needed(self, args: "DictConfig") -> Dict[str, int]:
        """
        Formats the pair of models and sets the number of conversations needed.
        """

        conversations_needed_string = args.blueprint.conversations_needed_string
        conversations_needed = {}
        parts = conversations_needed_string.split(',')
        for part in parts:
            model1, model2, num_string = part.split(':')
            models_alphabetize = [model1, model2]
            models_alphabetize.sort()  # alphabetizing for consistency
            conversations_needed[
                f'{models_alphabetize[0]}:{models_alphabetize[1]}'
            ] = int(
                num_string
            )  # format is model_1_name:model_2_name

        return conversations_needed

    def _get_shared_models(self, args: "DictConfig") -> Dict[str, dict]:
        """
        Loads the appropriate model pair and sends it into bot agent.
        """

        with open(args.blueprint.model_opt_path) as f:
            all_model_opts = yaml.safe_load(f.read())

        active_model_opts = {}
        for model_pair in self.conversations_needed:
            for model, opt in all_model_opts.items():
                if (
                    model in model_pair.split(':')
                    and self.conversations_needed[model_pair] > 0
                ):
                    active_model_opts[model] = opt

        return PerTurnEvalTurkLikeAgent.get_bot_agents(
            args=args, model_opts=active_model_opts
        )
