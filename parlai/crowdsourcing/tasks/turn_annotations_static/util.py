#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config
from omegaconf import DictConfig, OmegaConf

from parlai.crowdsourcing.utils.frontend import build_task
from parlai.crowdsourcing.utils.mturk import soft_block_mturk_workers


def run_static_task(cfg: DictConfig, task_directory: str):
    """
    Run static task, given configuration.
    """

    db, cfg = load_db_and_process_config(cfg)
    print(f'\nHydra config:\n{OmegaConf.to_yaml(cfg)}')

    random.seed(42)

    task_name = cfg.mephisto.task.get('task_name', 'turn_annotations_static')
    soft_block_qual_name = cfg.mephisto.blueprint.get(
        'block_qualification', f'{task_name}_block'
    )
    # Default to a task-specific name to avoid soft-block collisions
    soft_block_mturk_workers(cfg=cfg, db=db, soft_block_qual_name=soft_block_qual_name)

    build_task(task_directory)

    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
    operator.wait_for_runs_then_shutdown(
        skip_input=True, log_rate=cfg.monitoring_log_rate
    )
