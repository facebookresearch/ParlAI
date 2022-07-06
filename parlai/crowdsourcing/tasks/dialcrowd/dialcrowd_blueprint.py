#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING

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


STATIC_BLUEPRINT_TYPE = 'dialcrowd_static_blueprint'


@dataclass
class DialCrowdStaticBlueprintArgs(StaticReactBlueprintArgs):
    _blueprint_type: str = STATIC_BLUEPRINT_TYPE
    _group: str = field(
        default="DialCrowdStaticBlueprint",
        metadata={
            'help': """This task renders conversations from a file and asks for turn by turn annotations of them."""
        },
    )
    subtasks_per_unit: int = field(
        default=-1, metadata={"help": "Number of subtasks/comparisons to do per unit"}
    )


@register_mephisto_abstraction()
class DialCrowdStaticBlueprint(StaticReactBlueprint):
    """
    This Blueprint has a subtasks number option to combine multiple conversations into
    "sub-HITs".

    It also has options for the onboarding data answers and the annotation bucket
    definitions.
    """

    ArgsClass = DialCrowdStaticBlueprintArgs
    BLUEPRINT_TYPE = STATIC_BLUEPRINT_TYPE

    def __init__(
        self, task_run: "TaskRun", args: "DictConfig", shared_state: "SharedTaskState"
    ):
        super().__init__(task_run, args=args, shared_state=shared_state)
        self.subtasks_per_unit = self.args.blueprint.subtasks_per_unit

        if self.subtasks_per_unit <= 0:
            raise Exception(
                f'subtasks_per_unit must be greater than zero but was {self.subtasks_per_unit}'
            )

        self.raw_data = self._initialization_data_dicts

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

        # load the task configuration
        with open(os.path.join(get_task_path(), 'task_config/config.json')) as f:
            task_config = json.load(f)

        # combine the task configuration loaded from json with the settings
        # required by ParlAI.
        task_config.update(
            {
                "task_description": self.args.task.get('task_description', None),
                "task_title": self.args.task.get('task_title', None),
                "frame_height": 0,
                "num_subtasks": self.args.blueprint.subtasks_per_unit,
                "block_mobile": True,
            }
        )
        return task_config
