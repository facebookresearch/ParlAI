#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import TalkTheWalkWorld, InstructionWorld
from task_config import task_config


"""
This task consists of two local human agents and two MTurk agents,
chatting with each other in a free-form format.
You can end the conversation by sending a message ending with
`[DONE]` from human_1.
"""


def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument('--replay', action='store_true',
                           help='Set to replay old interactions')
    argparser.add_argument('--replay-log-file', type=str, default='',
                           help='location of log to use if replay')
    argparser.add_argument('--real-time', action='store_true',
                           help='Set to replay in real time ')
    argparser.add_argument('--replay-bot', action='store_true',
                           help='Set to replay bot actions instead of human')
    argparser.add_argument('--model-file', type=str, default='',
                           help='language generator model file')
    argparser.add_argument('--world-idx', type=int, default=-1,
                           help='specify world to load')
    argparser.add_argument('--start-idx', type=int, default=0,
                           help='where to start replay, if replaying actions')
    argparser.add_argument('--bot-type', type=str, default='discrete',
                           choices=['discrete', 'natural'],
                           help='which bot log to use')
    opt = argparser.parse_args()
    opt.update(task_config)

    mturk_agent_1_id = 'Tourist'
    mturk_agent_2_id = 'Guide'
    mturk_agent_ids = [mturk_agent_1_id, mturk_agent_2_id]
    task_directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(task_directory_path)
    opt['data_path'] = os.getcwd() + '/data/' + opt['task']
    mturk_manager = MTurkManager(opt=opt,
                                 mturk_agent_ids=mturk_agent_ids)
    mturk_manager.setup_server(task_directory_path=task_directory_path)

    try:
        mturk_manager.start_new_run()
        mturk_manager.create_hits()

        def run_onboard(worker):
            world = InstructionWorld(opt=opt, mturk_agent=worker)
            while not world.episode_done():
                world.parley()
                world.shutdown()

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        global worker_count
        worker_count = 0

        def assign_worker_roles(workers):
            workers[0].id = mturk_agent_ids[0]
            workers[1].id = mturk_agent_ids[1]
            return [workers[0], workers[1]]

        def run_conversation(mturk_manager, opt, workers):
            # Create mturk agents
            mturk_agent_1 = workers[0]
            mturk_agent_2 = workers[1]
            conv_idx = mturk_manager.conversation_index
            world = TalkTheWalkWorld(opt=opt,
                                     agents=[mturk_agent_1, mturk_agent_2],
                                     world_tag=conv_idx)

            while not world.episode_done():
                world.parley()
            world.shutdown()
            world.review_work()
            if not opt.get('replay'):
                world.save()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )
    except Exception:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
