# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import MTurkIGCEvalWorld, RoleOnboardWorld, IGCExampleGenerator, RATER
from task_configs.task_config_questions import task_config as tc_questions
from task_configs.task_config_responses import task_config as tc_responses

import os

round_choices = ['questions', 'responses']


def main():
    """
    IGC Human Evaluation.

    Specify the `--eval-data-path` to load examples for evaluation.

    The data in `--eval-data-path` should be formatted as a dictionary
    mapping IGC image ids to dicts with the following fields:
    {
        'questions': list of (<generator_name>, <generated_question>) tuples,
        'responses': list of (<generator_name>, <generated_response>) tuples,
        'question': question to use when evaluating responses,
        'context': context for the image
    }

    If not data path specified, loads a demo_example specified in worlds.py

    Specify `--image-path` for the path to the IGC images, where each example
    is saved as <image_id>.jpg


    NOTE: You can download the IGC Test Set from
        https://www.microsoft.com/en-us/download/details.aspx?id=55324

    And you can use the `download_igc_images.py` script to download the images
    (please put the IGC_crowd_test.csv file in this directory to use the script)
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '-min_t', '--min_turns', default=3, type=int, help='minimum number of turns'
    )
    argparser.add_argument(
        '-mt', '--max_turns', default=5, type=int, help='maximal number of chat turns'
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
        '-ni',
        '--num_images',
        type=int,
        default=5,
        help='number of images to show \
                           to turker',
    )
    argparser.add_argument(
        '--auto-approve-delay',
        type=int,
        default=3600 * 24,
        help='how long to wait for  \
                           auto approval',
    )
    argparser.add_argument(
        '--data-path', type=str, default='', help='where to save data'
    )
    argparser.add_argument(
        '--eval-data-path',
        type=str,
        default='',
        help='path to file with candidates to ' 'evaluate',
    )
    argparser.add_argument(
        '--image-path', type=str, default='', help='path to IGC images'
    )
    argparser.add_argument(
        '-rnd',
        '--dialog-round',
        type=str,
        default='questions',
        choices=round_choices,
        help='which dialog round to show',
    )

    opt = argparser.parse_args()
    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)
    if 'data_path' not in opt or opt['data_path'] == '':
        opt['data_path'] = "{}/data/{}_evals".format(os.getcwd(), opt['dialog_round'])
    opt['task_dir'] = os.getcwd()
    if opt['dialog_round'] == 'questions':
        opt.update(tc_questions)
    else:
        opt.update(tc_responses)

    mturk_agent_ids = [RATER]
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_ids)

    example_generator = IGCExampleGenerator(opt)
    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()

        def run_onboard(worker):
            worker.example_generator = example_generator
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
            world = MTurkIGCEvalWorld(
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
