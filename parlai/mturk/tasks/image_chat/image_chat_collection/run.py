# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.tasks.image_chat.agents import ImageChatTeacher
from worlds import \
    MTurkImageChatWorld, RoleOnboardWorld, PersonalityGenerator, \
    ExampleGenerator, RESPONDER
from task_configs.task_config_first_response import task_config as config_first
from task_configs.task_config_second_response import task_config as config_second
import os


def main():
    """
        Image Chat data collection task.

        A worker is shown an image and part of a conversation, and is given a
        personality with which the worker should continue the conversation.
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument('-min_t', '--min_turns', default=3, type=int,
                           help='minimum number of turns')
    argparser.add_argument('-mt', '--max_turns', default=5, type=int,
                           help='maximal number of chat turns')
    argparser.add_argument('-mx_rsp_time', '--max_resp_time', default=1800,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('-mx_onb_time', '--max_onboard_time', type=int,
                           default=300, help='time limit for turker'
                           'in onboarding')
    argparser.add_argument('-ni', '--num_images', type=int,
                           default=10, help='number of images to show \
                           to turker')
    argparser.add_argument('--auto-approve-delay', type=int,
                           default=3600*24*5, help='how long to wait for  \
                           auto approval')
    argparser.add_argument('--second-response', type='bool',
                           default=False, help='Specify if getting responses \
                           to responses to original comment')
    ImageChatTeacher.add_cmdline_args(argparser)

    opt = argparser.parse_args()
    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)
    if 'data_path' not in opt:
        opt['data_path'] = os.getcwd() + '/data/' + opt['task']
    opt.update(config_second if opt['second_response'] else config_first)

    mturk_agent_ids = [RESPONDER]
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=mturk_agent_ids
    )

    personality_generator = PersonalityGenerator(opt)
    example_generator = ExampleGenerator(opt)
    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()
        mturk_manager.ready_to_accept_workers()

        def run_onboard(worker):
            worker.personality_generator = personality_generator
            worker.example_generator = example_generator
            world = RoleOnboardWorld(opt, worker)
            world.parley()
            world.shutdown()

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.create_hits()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for w in workers:
                w.id = mturk_agent_ids[0]

        def run_conversation(mturk_manager, opt, workers):
            agents = workers[:]
            conv_idx = mturk_manager.conversation_index
            world = MTurkImageChatWorld(
                opt,
                agents=agents,
                world_tag='conversation t_{}'.format(conv_idx),
            )
            while not world.episode_done():
                world.parley()
            world.save_data()

            world.shutdown()
            world.review_work()

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
