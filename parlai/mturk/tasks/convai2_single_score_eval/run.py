# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import Convai2GeneralEval
from parlai.mturk.tasks.convai2_model_eval.worlds import PersonaProfileWorld, \
    PersonasGenerator
from task_config import task_config
import gc
import datetime
import config
import logging
import sys
import json

import os

MASTER_QUALIF = {
    'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
    'Comparator': 'Exists',
    # 'LocaleValues': [{'Country': 'US'}],
    'RequiredToPreview': True
}

MASTER_QUALIF_SDBOX = {
    'QualificationTypeId': '2ARFPLSP75KLA8M8DH1HTEQVJT3SY6',
    'Comparator': 'Exists',
    # 'LocaleValues': [{'Country': 'US'}],
    'RequiredToPreview': True
}


def main():
    """This task consists of an MTurk agent evaluating a chit-chat model. They
    are asked to chat to the model adopting a specific persona. After their
    conversation, they are asked to evaluate their partner on several metrics.
    """
    start_time = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
    argparser = ParlaiParser(False, add_model_args=True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument('-mt', '--max-turns', default=10, type=int,
                           help='maximal number of chat turns')
    argparser.add_argument('--max-resp-time', default=240,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--max-persona-time', type=int,
                           default=300, help='time limit for turker'
                           'entering the persona')
    argparser.add_argument('--ag-shutdown-time', default=120,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--persona-type', default='both', type=str,
                           choices=['both', 'self', 'other'],
                           help='Which personas to load from personachat')
    argparser.add_argument('--revised', default=False, type='bool',
                           help='Whether to use revised personas')
    argparser.add_argument('-rt', '--range-turn', default='5,6',
                           help='sample range of number of turns')
    argparser.add_argument('--auto-approve-delay', type=int,
                           default=3600 * 24 * 1,
                           help='how long to wait for auto approval')
    argparser.add_argument('--only-masters', type='bool', default=True,
                           help='Set to true to use only master turks for '
                                'this test eval')
    argparser.add_argument('--model-config', type=str, required=True,
                           help='Filename for file containing dict with model opt')
    argparser.add_argument('--mturk-log', type=str,
                           default='logs/{}.log'.format(start_time))

    def inject_override(opt, override_dict):
        opt['override'] = override_dict
        opt['model'] = override_dict['model']
        opt['model_file'] = override_dict['model_file']
        opt['log_level'] = 50

    def get_logger(opt):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fmt = logging.Formatter(
            '%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        if 'mturk_log' in opt:
            logfile = logging.FileHandler(opt['mturk_log'], 'a')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)
        logger.info('COMMAND: %s' % ' '.join(sys.argv))
        logger.info('-' * 100)
        logger.info('CONFIG:\n%s' % json.dumps(opt, indent=4, sort_keys=True))

        return logger

    start_opt = argparser.parse_args()

    inject_override(start_opt, getattr(config, start_opt['model_config']))

    bot = create_agent(start_opt)
    shared_bot_params = bot.share()

    get_logger(bot.opt)

    folder_name = '{}-{}'.format(start_opt['model_config'], start_time)

    start_opt['task'] = os.path.basename(
        os.path.dirname(os.path.abspath(__file__)))
    if 'data_path' not in start_opt:
        start_opt['data_path'] = os.getcwd() + '/data/' + folder_name
    start_opt.update(task_config)

    mturk_agent_ids = ['PERSON_1']

    mturk_manager = MTurkManager(
        opt=start_opt,
        mturk_agent_ids=mturk_agent_ids
    )

    persona_generator = PersonasGenerator(start_opt)
    directory_path = os.path.dirname(os.path.abspath(__file__))
    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()
        agent_qualifications = []
        if start_opt['only_masters'] is True:
            if start_opt['is_sandbox']:
                agent_qualifications.append(MASTER_QUALIF_SDBOX)
            else:
                agent_qualifications.append(MASTER_QUALIF)
        mturk_manager.create_hits(qualifications=agent_qualifications)

        # if not opt['is_sandbox']:
        #    # ADD BLOCKED WORKERS HERE
        #    blocked_worker_list = bad_workers_list
        #    #blocked_worker_list = []
        #    for w in blocked_worker_list:
        #        print('Soft Blocking {}\n'.format(w))
        #        mturk_manager.soft_block_worker(w)
        #        time.sleep(0.1)

        def run_onboard(worker):
            worker.persona_generator = persona_generator
            world = PersonaProfileWorld(start_opt, worker)
            world.parley()
            world.shutdown()
        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        def run_conversation(mturk_manager, opt, workers):
            # this function only supports human-bot
            agents = workers[0]
            conv_idx = mturk_manager.conversation_index
            world = Convai2GeneralEval(
                opt=start_opt,
                agents=[agents],
                range_turn=[int(s)
                            for s in start_opt['range_turn'].split(',')],
                max_turn=start_opt['max_turns'],
                max_resp_time=start_opt['max_resp_time'],
                model_agent_opt=shared_bot_params,
                world_tag='conversation t_{}'.format(conv_idx),
                agent_timeout_shutdown=opt['ag_shutdown_time'],
            )
            world.reset_random()
            while not world.episode_done():
                world.parley()
            world.save_data()

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
