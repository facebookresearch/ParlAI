#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.utils.misc import AttrDict
from parlai.mturk.core.mturk_manager import MTurkManager
import parlai.mturk.core.mturk_utils as mturk_utils

from worlds import WizardEval, TopicsGenerator, TopicChooseWorld
from task_config import task_config

from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
    KnowledgeRetrieverAgent,
)

import gc
import datetime
import json
import os
import sys
from parlai.utils.logging import ParlaiLogger, INFO

MASTER_QUALIF = {
    'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
    'Comparator': 'Exists',
    'RequiredToPreview': True,
}


def main():
    """
    This task consists of an MTurk agent evaluating a wizard model.

    They are assigned a topic and asked to chat.
    """
    start_time = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
    argparser = ParlaiParser(False, add_model_args=True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '-mt', '--max-turns', default=10, type=int, help='maximal number of chat turns'
    )
    argparser.add_argument(
        '--max-resp-time',
        default=240,
        type=int,
        help='time limit for entering a dialog message',
    )
    argparser.add_argument(
        '--generative-setup',
        default=False,
        help='mimic setup for the WoW generator task (use knowledge token)',
    )
    argparser.add_argument(
        '--max-choice-time',
        type=int,
        default=300,
        help='time limit for turker' 'choosing the topic',
    )
    argparser.add_argument(
        '--ag-shutdown-time',
        default=120,
        type=int,
        help='time limit for entering a dialog message',
    )
    argparser.add_argument(
        '-rt', '--range-turn', default='3,5', help='sample range of number of turns'
    )
    argparser.add_argument(
        '--human-eval',
        type='bool',
        default=False,
        help='human vs human eval, no models involved',
    )
    argparser.add_argument(
        '--auto-approve-delay',
        type=int,
        default=3600 * 24 * 1,
        help='how long to wait for auto approval',
    )
    argparser.add_argument(
        '--only-masters',
        type='bool',
        default=False,
        help='Set to true to use only master turks for ' 'this test eval',
    )
    argparser.add_argument(
        '--unique-workers',
        type='bool',
        default=False,
        help='Each worker must be unique',
    )
    argparser.add_argument(
        '--prepend-gold-knowledge',
        type='bool',
        default=False,
        help='Add the gold knowledge to the input text from the human for '
        'the model observation.',
    )
    argparser.add_argument(
        '--mturk-log',
        type=str,
        default='data/mturklogs/wizard_of_wikipedia/{}.log'.format(start_time),
    )

    def inject_override(opt, override_dict):
        opt['override'] = override_dict
        for k, v in override_dict.items():
            opt[k] = v

    def get_logger(opt):
        fmt = '%(asctime)s: [ %(message)s ]'
        logfile = None
        if 'mturk_log' in opt:
            logfile = opt['mturk_log']
            if not os.path.isdir(os.path.dirname(logfile)):
                os.makedirs(os.path.dirname(logfile))
        logger = ParlaiLogger(
            "mturk_woz",
            console_level=INFO,
            file_level=INFO,
            console_format=fmt,
            file_format=fmt,
            filename=logfile,
        )
        logger.info('COMMAND: %s' % ' '.join(sys.argv))
        logger.info('-' * 100)
        logger.info('CONFIG:\n%s' % json.dumps(opt, indent=4, sort_keys=True))

        return logger

    # MODEL CONFIG
    # NOTE: please edit this to test your own models
    config = {
        'model_file': 'models:wizard_of_wikipedia/end2end_generator/model',
        'generative_setup': True,
        'prepend_gold_knowledge': True,
        'model': 'projects:wizard_of_wikipedia:generator',
        'beam_size': 10,  # add inference arguments here
        'inference': 'beam',
        'beam_block_ngram': 3,
    }

    # add dialogue model args
    argparser.add_model_subargs(config['model'])
    # add knowledge retriever args
    argparser.add_model_subargs('projects:wizard_of_wikipedia:knowledge_retriever')
    start_opt = argparser.parse_args()

    inject_override(start_opt, config)

    if not start_opt.get('human_eval'):
        # make dialogue responder model
        bot = create_agent(start_opt)
        shared_bot_params = bot.share()
        # make knowledge retriever
        knowledge_opt = {
            'model': 'projects:wizard_of_wikipedia:knowledge_retriever',
            'add_token_knowledge': not start_opt['generative_setup'],
            'datapath': start_opt['datapath'],
            'interactive_mode': False,  # interactive mode automatically sets fixed cands
        }
        # add all existing opt to the knowledge opt, without overriding
        # the above arguments
        for k, v in start_opt.items():
            if k not in knowledge_opt and k not in config:
                knowledge_opt[k] = v

        knowledge_agent = KnowledgeRetrieverAgent(knowledge_opt)
        knowledge_agent_shared_params = knowledge_agent.share()

    else:
        shared_bot_params = None

    if not start_opt['human_eval']:
        get_logger(bot.opt)
    else:
        get_logger(start_opt)

    if start_opt['human_eval']:
        folder_name = 'human_eval-{}'.format(start_time)
    else:
        folder_name = '{}-{}'.format(start_opt['model'], start_time)

    start_opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if 'data_path' not in start_opt:
        start_opt['data_path'] = os.path.join(
            os.getcwd(), 'data', 'wizard_eval', folder_name
        )
    start_opt.update(task_config)

    if not start_opt.get('human_eval'):
        mturk_agent_ids = ['PERSON_1']
    else:
        mturk_agent_ids = ['PERSON_1', 'PERSON_2']

    mturk_manager = MTurkManager(opt=start_opt, mturk_agent_ids=mturk_agent_ids)

    topics_generator = TopicsGenerator(start_opt)
    directory_path = os.path.dirname(os.path.abspath(__file__))
    mturk_manager.setup_server(task_directory_path=directory_path)
    worker_roles = {}
    connect_counter = AttrDict(value=0)

    try:
        mturk_manager.start_new_run()
        agent_qualifications = []
        if not start_opt['is_sandbox']:
            # assign qualifications
            if start_opt['only_masters']:
                agent_qualifications.append(MASTER_QUALIF)
            if start_opt['unique_workers']:
                qual_name = 'UniqueChatEval'
                qual_desc = (
                    'Qualification to ensure each worker completes a maximum '
                    'of one of these chat/eval HITs'
                )
                qualification_id = mturk_utils.find_or_create_qualification(
                    qual_name, qual_desc, False
                )
                print('Created qualification: ', qualification_id)
                UNIQUE_QUALIF = {
                    'QualificationTypeId': qualification_id,
                    'Comparator': 'DoesNotExist',
                    'RequiredToPreview': True,
                }
                start_opt['unique_qualif_id'] = qualification_id
                agent_qualifications.append(UNIQUE_QUALIF)
        mturk_manager.create_hits(qualifications=agent_qualifications)

        def run_onboard(worker):
            if start_opt['human_eval']:
                role = mturk_agent_ids[connect_counter.value % len(mturk_agent_ids)]
                connect_counter.value += 1
                worker_roles[worker.worker_id] = role
            else:
                role = 'PERSON_1'
            worker.topics_generator = topics_generator
            world = TopicChooseWorld(start_opt, worker, role=role)
            world.parley()
            world.shutdown()

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()

        def check_single_worker_eligibility(worker):
            return True

        def check_multiple_workers_eligibility(workers):
            valid_workers = {}
            for worker in workers:
                worker_id = worker.worker_id
                if worker_id not in worker_roles:
                    print('Something went wrong')
                    continue
                role = worker_roles[worker_id]
                if role not in valid_workers:
                    valid_workers[role] = worker
                if len(valid_workers) == 2:
                    break
            return valid_workers.values() if len(valid_workers) == 2 else []

        if not start_opt['human_eval']:
            eligibility_function = {
                'func': check_single_worker_eligibility,
                'multiple': False,
            }
        else:
            eligibility_function = {
                'func': check_multiple_workers_eligibility,
                'multiple': True,
            }

        def assign_worker_roles(workers):
            if start_opt['human_eval']:
                for worker in workers:
                    worker.id = worker_roles[worker.worker_id]
            else:
                for index, worker in enumerate(workers):
                    worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        def run_conversation(mturk_manager, opt, workers):
            conv_idx = mturk_manager.conversation_index
            world = WizardEval(
                opt=start_opt,
                agents=workers,
                range_turn=[int(s) for s in start_opt['range_turn'].split(',')],
                max_turn=start_opt['max_turns'],
                max_resp_time=start_opt['max_resp_time'],
                model_agent_opt=shared_bot_params,
                world_tag='conversation t_{}'.format(conv_idx),
                agent_timeout_shutdown=opt['ag_shutdown_time'],
                knowledge_retriever_opt=knowledge_agent_shared_params,
            )
            while not world.episode_done():
                world.parley()
            world.save_data()

            world.shutdown()
            gc.collect()

        mturk_manager.start_task(
            eligibility_function=eligibility_function,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation,
        )

    except BaseException:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
