# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.mturk.core.mturk_manager import MTurkManager
import parlai.mturk.core.mturk_utils as mturk_utils

from worlds import ControllableDialogEval, PersonasGenerator, PersonaAssignWorld
from task_config import task_config
import model_configs as mcf

from threading import Lock
import gc
import datetime
import json
import logging
import os
import sys
import copy
import random
import pprint


# update this with models you want to run. these names correspond to variables
# in model_configs.py
SETTINGS_TO_RUN = """
baseline_model
greedy_model
inquisitive_model_ct_setting00
inquisitive_model_ct_setting01
inquisitive_model_ct_setting04
inquisitive_model_ct_setting07
inquisitive_model_ct_setting10
inquisitive_model_ct_setting10_better
interesting_nidf_model_bfw_setting_04
interesting_nidf_model_bfw_setting_06
interesting_nidf_model_bfw_setting_08
interesting_nidf_model_bfw_setting_minus_04
interesting_nidf_model_bfw_setting_minus_10
interesting_nidf_model_ct_setting0
interesting_nidf_model_ct_setting2
interesting_nidf_model_ct_setting4
interesting_nidf_model_ct_setting7
interesting_nidf_model_ct_setting9
repetition_model_setting05
repetition_model_setting12
repetition_model_setting35
repetition_model_setting35_settinginf
repetition_model_settinginf
responsiveness_model_bfw_setting_00
responsiveness_model_bfw_setting_05
responsiveness_model_bfw_setting_10
responsiveness_model_bfw_setting_13
responsiveness_model_bfw_setting_minus_10
""".strip().split()


def main():
    """This task consists of an MTurk agent evaluating a Controllable Dialog model.
    """
    start_time = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
    argparser = ParlaiParser(False, add_model_args=True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument('--max-resp-time', default=240,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--max-choice-time', type=int,
                           default=300, help='time limit for turker'
                           'choosing the topic')
    argparser.add_argument('--ag-shutdown-time', default=120,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--num-turns', default=6, type=int,
                           help='number of turns of dialogue')
    argparser.add_argument('--human-eval', type='bool', default=False,
                           help='human vs human eval, no models involved')
    argparser.add_argument('--auto-approve-delay', type=int,
                           default=3600 * 24 * 2,
                           help='how long to wait for auto approval')
    argparser.add_argument('--only-masters', type='bool', default=False,
                           help='Set to true to use only master turks for '
                                'this test eval')
    argparser.add_argument('--create-model-qualif', type='bool', default=True,
                           help='Create model qualif so unique eval between'
                                'models.')
    argparser.add_argument('--limit-workers', type=int, default=len(SETTINGS_TO_RUN),
                           help='max HITs a worker can complete')
    argparser.add_argument('--mturk-log', type=str,
                           default=(
                                '$HOME/ParlAI/data/mturklogs/controllable/{}.log'
                                .format(start_time)))
    argparser.add_argument('--short-eval', type='bool', default=True,
                           help='Only ask engagingness question and persona'
                                'question.')
    # persona specific arguments
    argparser.add_argument('--persona-type', type=str, default='self',
                           choices=['self', 'other', 'none'])
    argparser.add_argument('--persona-datatype', type=str, default='valid',
                           choices=['train', 'test', 'valid'])
    argparser.add_argument('--max-persona-time', type=int, default=360,
                           help='max time to view persona')

    def get_logger(opt):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fmt = logging.Formatter(
            '%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        if 'mturk_log' in opt:
            logfn = opt['mturk_log'].replace('$HOME', os.environ['HOME'])
            if not os.path.isdir(os.path.dirname(logfn)):
                raise OSError("Please run `mkdir -p {}`".format(os.path.dirname(logfn)))
            logfile = logging.FileHandler(logfn, 'a')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)
        logger.info('COMMAND: %s' % ' '.join(sys.argv))
        logger.info('-' * 100)
        logger.info('CONFIG:\n%s' % json.dumps(opt, indent=4, sort_keys=True))

        return logger

    start_opt = argparser.parse_args()

    task_config['task_description'] = task_config['task_description'].format(
        start_opt['reward']
    )

    # set options
    start_opt['limit_workers'] = len(SETTINGS_TO_RUN)
    start_opt['allowed_conversations'] = 1
    start_opt['max_hits_per_worker'] = start_opt['limit_workers']
    start_opt['task'] = os.path.basename(
        os.path.dirname(os.path.abspath(__file__)))

    start_opt.update(task_config)

    get_logger(start_opt)

    model_share_params = {}
    worker_models_seen = {}
    model_opts = {}
    model_counts = {}

    lock = Lock()

    for setup in SETTINGS_TO_RUN:
        assert 'human' not in setup
        model_counts[setup] = 0
        agent_config = getattr(mcf, setup)
        combined_config = copy.deepcopy(start_opt)
        for k, v in agent_config.items():
            combined_config[k] = v
            combined_config['override'][k] = v
        folder_name = '{}-{}'.format(setup, start_time)
        combined_config['save_data_path'] = os.path.join(
            start_opt['datapath'],
            'local_controllable_dialogue',
            folder_name
        )
        model_opts[setup] = combined_config
        bot = create_agent(combined_config, True)
        model_share_params[setup] = bot.share()

    if not start_opt.get('human_eval'):
        mturk_agent_ids = ['PERSON_1']
    else:
        mturk_agent_ids = ['PERSON_1', 'PERSON_2']

    mturk_manager = MTurkManager(
        opt=start_opt,
        mturk_agent_ids=mturk_agent_ids
    )

    personas_generator = PersonasGenerator(start_opt)

    directory_path = os.path.dirname(os.path.abspath(__file__))

    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()
        agent_qualifications = []
        # assign qualifications
        if start_opt['create_model_qualif']:
            qual_name = 'ControlEvalRound2'
            qual_desc = (
                'Qualification to ensure workers complete only a certain'
                'number of these HITs'
            )
            qualification_id = mturk_utils.find_or_create_qualification(
                qual_name, qual_desc, False
            )
            print('Created qualification: ', qualification_id)
            start_opt['unique_qualif_id'] = qualification_id

        def run_onboard(worker):
            worker.personas_generator = personas_generator
            world = PersonaAssignWorld(start_opt, worker)
            world.parley()
            world.shutdown()

        def check_worker_eligibility(worker):
            worker_id = worker.worker_id
            lock.acquire()
            retval = len(worker_models_seen.get(worker_id, [])) < len(SETTINGS_TO_RUN)
            lock.release()
            return retval

        def assign_worker_roles(workers):
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()
        mturk_manager.create_hits(qualifications=agent_qualifications)

        def run_conversation(mturk_manager, opt, workers):
            conv_idx = mturk_manager.conversation_index

            # gotta find a bot this worker hasn't seen yet
            assert len(workers) == 1
            worker_id = workers[0].worker_id
            lock.acquire()
            if worker_id not in worker_models_seen:
                worker_models_seen[worker_id] = set()
            print("MODELCOUNTS:")
            print(pprint.pformat(model_counts))
            logging.info("MODELCOUNTS\n" + pprint.pformat(model_counts))
            model_options = [
                (model_counts[setup_name] + 10 * random.random(), setup_name)
                for setup_name in SETTINGS_TO_RUN
                if setup_name not in worker_models_seen[worker_id]
            ]
            if not model_options:
                lock.release()
                logging.error(
                    "Worker {} already finished all settings! Returning none"
                    .format(worker_id)
                )
                return None
            _, model_choice = min(model_options)

            worker_models_seen[worker_id].add(model_choice)
            model_counts[model_choice] += 1
            lock.release()

            world = ControllableDialogEval(
                opt=model_opts[model_choice],
                agents=workers,
                num_turns=start_opt['num_turns'],
                max_resp_time=start_opt['max_resp_time'],
                model_agent_opt=model_share_params[model_choice],
                world_tag='conversation t_{}'.format(conv_idx),
                agent_timeout_shutdown=opt['ag_shutdown_time'],
                model_config=model_choice,
            )
            world.reset_random()
            while not world.episode_done():
                world.parley()
            world.save_data()

            lock.acquire()
            if not world.convo_finished:
                model_counts[model_choice] -= 1
                worker_models_seen[worker_id].remove(model_choice)
            lock.release()

            world.shutdown()
            gc.collect()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )

    except BaseException:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
