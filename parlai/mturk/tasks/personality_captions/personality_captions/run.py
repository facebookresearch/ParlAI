#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.tasks.personality_captions.agents import PersonalityCaptionsTeacher
from parlai.tasks.personality_captions.build import build as build_pc_data
from worlds import (
    MTurkPersonalityCaptionsWorld,
    RoleOnboardWorld,
    PersonalityGenerator,
    ImageGenerator,
    COMMENTER,
    PersonalityAndImageGenerator,
    TASK_TYPE_TO_CONFIG,
)
import os


def main():
    """
    Personality-Captions Data Collection Task.

    This is the task setup used when collecting the Personality-Captions dataset
    (https://arxiv.org/abs/1810.10665).
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    PersonalityCaptionsTeacher.add_cmdline_args(argparser)
    argparser.add_argument(
        '-ni',
        '--num_images',
        type=int,
        default=10,
        help='number of images to show \
                           to turker',
    )
    argparser.add_argument(
        '-mx_rsp_time',
        '--max_resp_time',
        default=1800,
        type=int,
        help='time limit for entering a dialog message',
    )
    argparser.add_argument(
        '-mx_onb_time',
        '--max_onboard_time',
        type=int,
        default=300,
        help='time limit for turker' 'in onboarding',
    )
    argparser.add_argument(
        '--auto-approve-delay',
        type=int,
        default=3600 * 24 * 5,
        help='how long to wait for  \
                           auto approval',
    )
    argparser.add_argument(
        '--multiple-personality',
        type='bool',
        default=False,
        help='for getting captions with ' 'multiple personalities for same image',
    )
    argparser.add_argument(
        '--task-type',
        type=str,
        default='personality',
        choices=['personality', 'no_personality', 'caption'],
        help='Task Type - specify `personality` for '
        'original task, `no_personality` for the same task '
        'instructions but with no personality, and '
        '`caption` for the task but asking for a normal '
        'caption.',
    )

    opt = argparser.parse_args()

    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)
    if 'data_path' not in opt:
        opt['data_path'] = os.getcwd() + '/data/' + opt['task']
    opt.update(TASK_TYPE_TO_CONFIG[opt['task_type']])
    build_pc_data(opt)
    mturk_agent_ids = [COMMENTER]
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_ids, use_db=True)
    personality_generator = PersonalityGenerator(opt)
    image_generator = ImageGenerator(opt)
    personality_and_image_generator = PersonalityAndImageGenerator(opt)
    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()

        def run_onboard(worker):
            worker.personality_generator = personality_generator
            worker.image_generator = image_generator
            worker.personality_and_image_generator = personality_and_image_generator
            world = RoleOnboardWorld(opt, worker)
            world.parley()
            world.shutdown()

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()
        mturk_manager.create_hits()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for w in workers:
                w.id = mturk_agent_ids[0]

        def run_conversation(mturk_manager, opt, workers):
            agents = workers[:]
            conv_idx = mturk_manager.conversation_index
            world = MTurkPersonalityCaptionsWorld(
                opt, agents=agents, world_tag='conversation t_{}'.format(conv_idx)
            )
            while not world.episode_done():
                world.parley()
            world.save_data()

            world.shutdown()
            world.review_work()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
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
