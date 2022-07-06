#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import os
from collections import defaultdict
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import List, Dict, Any

from parlai.crowdsourcing.projects.wizard_of_internet import constants
from parlai.crowdsourcing.projects.wizard_of_internet.wizard_internet_blueprint import (
    WIZARD_INTERNET_PARLAICHAT_BLUEPRINT,
)
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig
import parlai.utils.logging as logging

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    SharedParlAITaskState,
)
from mephisto.operations.operator import Operator
from mephisto.operations.hydra_config import register_script_config
from mephisto.tools.scripts import load_db_and_process_config

"""
Read parlai/crowdsourcing/README.md to learn how to launch
crowdsourcing tasks with this script.
"""

_ = WIZARD_INTERNET_PARLAICHAT_BLUEPRINT

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = ['_self_', {"conf": "dev"}]


@dataclass
class ScriptConfig(MTurkRunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY
    turn_timeout: int = field(
        default=300,
        metadata={
            'help': 'Maximum response time before kicking '
            'a worker out, default 300 seconds'
        },
    )


register_script_config(name='scriptconfig', module=ScriptConfig)


def load_apprentice_persona_list(personas_fpath: str, shuffle: bool):
    """
    Reads a list of curated apprentice personas.
    """
    logging.info('Loading personas.')
    with open(personas_fpath, 'r') as pf:
        personas = [p.strip() for p in pf if p.strip()]
    logging.info(f'{len(personas)} personas loaded.')
    if shuffle:
        random.shuffle(personas)
    return personas


def load_previously_used_personas_counts(fpath: str):
    """
    Loads an existing count for how many times each persona was used.

    This is useful if the task was restarted after some initial data collection.
    """
    logging.info('Loading the previous runs persona counts.')
    personas_count = defaultdict(int)
    try:
        with open(fpath, 'r') as fi:
            for pl in fi:
                if not pl.strip():
                    continue
                persona, count = pl.strip().lower().split(';')
                personas_count[persona.strip()] = int(count)
    except FileNotFoundError:
        logging.info(
            f'Persona count file not found in {fpath}. Starting new persona use counter.'
        )

    logging.info(f'{len(personas_count)} previously used persona counts loaded.')
    return personas_count


def get_persona_locations(locations_fpath: str):
    """
    Reads a list of locations.
    """
    locations = []
    logging.info('Loading the locations file.')
    with open(locations_fpath) as lf:
        for line in lf:
            s = line.strip()
            if not s:
                continue
            locations.append(s)
    logging.info(f'{len(locations)} location loaded')
    return locations


def remove_overused_persona(
    personas: List[str], persona_use_count: Dict[str, int], max_persona_use: int
):
    """
    Removes personas that were used too often from the list of personas.
    """
    if not max_persona_use or not persona_use_count:
        return personas
    cleaned_personas = []
    for p in personas:
        if persona_use_count[p.lower()] < max_persona_use:
            cleaned_personas.append(p)
    logging.info(
        f'{len(cleaned_personas)} out of {len(personas)} personas accepted for use, '
        f'based on use count being less than maximum allowed of {max_persona_use}'
    )
    return cleaned_personas


def get_world_opt(config: DictConfig):
    """
    Generates the main chat world opt from Mephisto config.
    """
    blueprint_data = config.mephisto.blueprint
    previous_personas_count = load_previously_used_personas_counts(
        blueprint_data.persona_counts_file
    )
    num_max_persona_use = blueprint_data.max_times_persona_use
    personas = load_apprentice_persona_list(
        blueprint_data.personas_file, blueprint_data.shuffle_persona
    )
    personas = remove_overused_persona(
        personas, previous_personas_count, num_max_persona_use
    )
    locations = get_persona_locations(blueprint_data.locations_file)
    return {
        'send_task_data': True,
        'min_turns': blueprint_data.min_turns,
        'wizard_time_out': blueprint_data.wizard_time_out,
        'apprentice_time_out': blueprint_data.apprentice_time_out,
        'search_warning_turn': blueprint_data.search_warning_turn,
        'search_warning_threshold': blueprint_data.search_warning_threshold,
        'select_warning_turn': blueprint_data.select_warning_turn,
        'select_warning_threshold': blueprint_data.select_warning_threshold,
        'personas': personas,
        'prev_persona_count': previous_personas_count,
        'max_times_persona_use': num_max_persona_use,
        'locations': locations,
        'pick_persona_with_replacement': blueprint_data.use_personas_with_replacement,
        'search_server': blueprint_data.search_server,
        'num_passages_retrieved': blueprint_data.num_passages_retrieved,
        'soft_block_qname': blueprint_data.block_qualification,
        constants.ROLE_QUALIFICATION_NAME_KEY: blueprint_data.role_qualification,
    }


def get_onboarding_world_opt(config: DictConfig):
    """
    Generates onboarding world opt from Mephisto config.
    """
    blueprint_data = config.mephisto.blueprint
    return {
        'wizard_time_out': blueprint_data.wizard_time_out,
        'apprentice_time_out': blueprint_data.apprentice_time_out,
        'send_task_data': False,
        'is_onboarding': True,
        'search_server': blueprint_data.search_server,
        'num_passages_retrieved': blueprint_data.num_passages_retrieved,
        'onboarding_qualification': blueprint_data.onboarding_qualification,
        constants.ROLE_QUALIFICATION_NAME_KEY: blueprint_data.role_qualification,
    }


def get_worker_eval_function(role_qname: str, onboarding_qname: str):
    """
    Returns the callback function that is used for checking worker qualification.

    Check `worker_can_do_unit` of `SharedTaskState` in Mephisto.
    """

    def worker_eval_function(worker, unit):
        """
        Checks the worker qualification for the task, based on their existing records.
        """
        worker_qualification = worker.get_granted_qualification(role_qname)
        if not worker_qualification:
            # has not done any onboarding training yet
            logging.debug('Worker does not have any qualifications (new worker).')
            return True

        qualification_status = worker_qualification.value
        logging.debug(f'Worker role qualification is {qualification_status}')
        if qualification_status in (
            constants.WIZARD_IN_TRAINING,
            constants.APPRENTICE_IN_TRAINING,
        ):
            # The agent had started the onboarding training but was not finished
            onboarding_qual = worker.get_granted_qualification(onboarding_qname)
            return not onboarding_qual or not onboarding_qual.value

        # The agent has successfully finished the onboarding training
        if unit.unit_index == 0:
            return qualification_status == constants.WIZARD
        else:
            return qualification_status == constants.APPRENTICE

    return worker_eval_function


def check_role_training_qualification(
    db: LocalMephistoDB, qname: str, requester_name: str
):
    """
    Initializes the qualification name in DB, if it does not exist.
    """

    logging.info(f'Checking for "{qname}"" qualification.')
    if not db.find_qualifications(qname):
        logging.info('Creating the qualification.')
        db.make_qualification(qname)
        reqs = db.find_requesters(requester_name=requester_name, provider_type='mturk')
        requester = reqs[-1]
        requester._create_new_mturk_qualification(qname)
    else:
        logging.info('Qualification exists.')


def update_persona_use_counts_file(
    fptah: str, counts: Dict[str, int], sorted_order=True
):
    """
    Writes the persona use counts to file.

    This is to keep track of use counts for the next time that the task was restarted.
    See `load_previously_used_personas_counts` function above.
    """
    logging.info(f'Writting new persona counts to {fptah}')
    items = counts.items()
    if sorted_order:
        items = sorted(items, key=lambda x: x[1], reverse=True)
    saved_count = 0
    with open(fptah, 'w') as fo:
        for p, c in items:
            if c > 0:
                saved_count += 1
                fo.write(f'{p} ; {c}\n')
    logging.info(f'Saved {saved_count} recent persona counts successfully.')


def add_banned_words_frontend_conf(task_state, fpath: str = None):
    """
    Adds the list of banned words to the task config to be used later in the frontend.

    It reads the text file specified in fpath to populate a list banned words. Then adds
    this list to Mephisto `task_config` to make it accessible for the front-end app. The
    file specified by `fpath` is a plain text file where each line contains a single
    banned word/phrase.
    """
    banned_words = []
    if fpath and os.path.exists(fpath):
        with open(fpath, 'r') as fin:
            banned_words.extend([w.strip().lower() for w in fin if w.strip()])

    task_state.task_config['bannedWords'] = banned_words


@hydra.main(config_path="hydra_configs", config_name='scriptconfig')
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)
    world_opt = get_world_opt(cfg)
    onboarding_world_opt = get_onboarding_world_opt(cfg)
    shared_state = SharedParlAITaskState(
        world_opt=world_opt, onboarding_world_opt=onboarding_world_opt
    )

    check_role_training_qualification(
        db=db,
        qname=world_opt[constants.ROLE_QUALIFICATION_NAME_KEY],
        requester_name=cfg.mephisto.provider.requester_name,
    )

    shared_state.task_config['minTurns'] = world_opt['min_turns']
    shared_state.task_config['onboardingPersona'] = constants.ONBOARDING_PERSONA
    shared_state.worker_can_do_unit = get_worker_eval_function(
        world_opt[constants.ROLE_QUALIFICATION_NAME_KEY],
        onboarding_world_opt['onboarding_qualification'],
    )

    banned_words_fpath = cfg.mephisto.blueprint.banned_words_file
    add_banned_words_frontend_conf(shared_state, banned_words_fpath)

    operator = Operator(db)
    operator.validate_and_run_config(cfg.mephisto, shared_state)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=300)
    update_persona_use_counts_file(
        cfg.mephisto.blueprint.persona_counts_file, world_opt['prev_persona_count']
    )


if __name__ == '__main__':
    main()
