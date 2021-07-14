#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config
from omegaconf import DictConfig, OmegaConf

from parlai.crowdsourcing.utils.mturk import soft_block_mturk_workers
from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import (
    SharedModelChatTaskState,
)
import parlai.crowdsourcing.tasks.model_chat.worlds as world_module
import parlai.utils.logging as logging

try:
    from mephisto.operations.operator import Operator
    from mephisto.tools.scripts import load_db_and_process_config
    from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
        BLUEPRINT_TYPE,
        SharedParlAITaskState,
    )
    from mephisto.data_model.qualification import make_qualification_dict, QUAL_EXISTS
except ImportError:
    logging.info("unable to import from mephisto")
    Operator = None
    load_db_and_process_config = None
    SharedParlAITaskState = None
    make_qualification_dict = None
    QUAL_EXISTS = None
    BLUEPRINT_TYPE = None

from parlai_internal.crowdsourcing.projects.reverse_persona.parlai_chat_task_demo.constants import (
    ALLOWLIST_QUAL_NAME,
)
import json


def check_override_opt(args):
    with open(args.blueprint.override_opt_path) as f:
        override_opt = json.load(f)
    if (
        override_opt['bot_model_name']
        != args.blueprint.conversations_needed_string.split(":")[0]
    ):
        raise AssertionError(
            f"YOU bot model name in {args.blueprint.override_opt_path} doesnt match with {args.blueprint.conversations_needed_string}"
        )


def run_task(cfg: DictConfig, task_directory: str):
    """
    Run task, given configuration.
    """

    frontend_source_dir = os.path.join(task_directory, "webapp")
    frontend_build_dir = os.path.join(frontend_source_dir, "build")
    _ = frontend_build_dir  # Unused at the moment

    db, cfg = load_db_and_process_config(cfg)
    print(f'\nHydra config:\n{OmegaConf.to_yaml(cfg)}')
    logging.warning(f'QUAL_TYPE = allow, task_name={cfg.mephisto.task.task_name}')
    if cfg.mephisto.provider._provider_type == 'mock':
        cfg.mephisto.task.task_name = cfg.mephisto.task.task_name + '_mock'
    if 'sandbox' in cfg.mephisto.provider.requester_name:
        cfg.mephisto.task.task_name = cfg.mephisto.task.task_name + '_sandbox'
    check_override_opt(cfg.mephisto)

    random.seed(42)

    # Update task name when on sandbox or local to ensure data is split.
    task_name = cfg.mephisto.task.get('task_name', 'model_chat')
    architect_type = cfg.mephisto.architect._architect_type
    if architect_type == 'local':
        task_name = f"{task_name}_local"
    elif architect_type == 'mturk_sandbox':
        task_name = f"{task_name}_sandbox"
    cfg.mephisto.task.task_name = task_name

    soft_block_qual_name = cfg.mephisto.blueprint.get(
        'block_qualification', f'{task_name}_block'
    )
    # Default to a task-specific name to avoid soft-block collisions
    # soft_block_mturk_workers(cfg=cfg, db=db, soft_block_qual_name=soft_block_qual_name)

    existing_qualifications = [
        make_qualification_dict(ALLOWLIST_QUAL_NAME, QUAL_EXISTS, None)
    ]

    # Init
    shared_state = SharedModelChatTaskState(
        world_module=world_module, qualifications=existing_qualifications
    )

    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=shared_state)
    operator.wait_for_runs_then_shutdown(
        skip_input=True, log_rate=cfg.monitoring_log_rate
    )
