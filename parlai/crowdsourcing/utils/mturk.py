#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from dataclasses import dataclass, field
from typing import Optional

from mephisto.abstractions.database import MephistoDB
from mephisto.abstractions.providers.mturk.mturk_agent import MTurkAgent
from mephisto.abstractions.providers.mturk.utils.script_utils import (
    direct_soft_block_mturk_workers,
)
from mephisto.operations.hydra_config import RunScriptConfig
from omegaconf import DictConfig, MISSING


@dataclass
class MTurkConfig:
    """
    Add MTurk-specific flags.
    """

    worker_blocklist_paths: Optional[str] = field(
        default=MISSING,
        metadata={
            "help": (
                'Path(s) to a list of IDs of workers to soft-block, separated by newlines. Use commas to indicate multiple lists'
            )
        },
    )


@dataclass
class MTurkRunScriptConfigMixin:
    """
    Add useful flags for running MTurk tasks.
    """

    current_time: int = int(time.time())  # For parametrizing block_qualification
    mturk: MTurkConfig = MTurkConfig()


@dataclass
class MTurkRunScriptConfig(MTurkRunScriptConfigMixin, RunScriptConfig):
    """
    Add useful flags for running MTurk tasks.

    Use this instead of MTurkRunScriptConfigMixin when there are no task-specific fields
    that need to be set in the script config.
    """


def get_mturk_id_from_mephisto_wrapper(agent):
    """
    Returns the MTurk worker ID from a ParlAI-Wrapped Mephisto Agent.
    """
    if not isinstance(agent, MTurkAgent):
        return f"--NOT-MTURK-AGENT-{agent.mephisto_agent.get_worker().worker_name}"
    return agent.mephisto_agent.get_worker().get_mturk_worker_id()


def soft_block_mturk_workers(
    cfg: DictConfig, db: MephistoDB, soft_block_qual_name: str
):
    """
    Soft-block all MTurk workers listed in the input paths.
    """
    if cfg.mephisto.provider.get('_provider_type', 'mock') == 'mturk':
        if cfg.mturk.get('worker_blocklist_paths', None) is None:
            print(
                'Skipping soft-blocking workers because no blocklist path(s) are given.'
            )
        else:
            blocklist_paths = cfg.mturk.worker_blocklist_paths.split(',')
            worker_blocklist = set()
            for path in blocklist_paths:
                with open(path) as f:
                    worker_blocklist |= set(f.read().strip().split('\n'))
            print(
                f'About to soft-block {len(worker_blocklist):d} workers by '
                f'giving them the qualification "{soft_block_qual_name}".'
            )
            direct_soft_block_mturk_workers(
                db=db,
                worker_list=list(worker_blocklist),
                soft_block_qual_name=soft_block_qual_name,
                requester_name=cfg.mephisto.provider.get("requester_name", None),
            )
