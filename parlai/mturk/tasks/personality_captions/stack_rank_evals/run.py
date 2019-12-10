#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import (
    MTurkPersonalityCaptionsStackRankWorld,
    RoleOnboardWorld,
    ExampleGenerator,
    CHOOSER,
)
from parlai.tasks.personality_captions.agents import PersonalityCaptionsTeacher
from task_config import task_config
import os


def main():
    """
    Human Evaluation of various image captions/comments.

    A turker is shown an image and two possible comments/captions, and
    optionally the personality used to create these captions. Then, the
    turker is asked to choose which caption they think is more engaging.

    In this example, we will just be comparing the original comment twice
    (this is just to demonstrate the task for future use).

    To use your own data, please specify `--eval-data-path` to an
    appropriate json file with a list of examples, where each example
    has the following structure:
        {
            'image_hash': <hash of image>,
            'personality': <personality, if applicable>,
            '<compare_key_1>': <first option to compare>,
            '<compare_key_2>': <second option to compare>,
            .
            .
            .
        }
    Note that compare_key_1 and compare_key_2 can be any field, as long as they
    map to a string comment/caption.

    Example Scenario:
        Suppose you have the original Personality-Captions dataset, and
        you would like to compare the outputs of your model called `model`.

        Your data may look like the following:
        [{
            'image_hash': hashforimageofcat,
            'personality': 'Sweet',
            'comment': 'Look at the cute cat!', # Human Comment
            'model_comment': 'That's a weird looking dog' # Model Comment
        }, ...]

        Thus, you would specify `-ck1 comment -ck2 model_comment` to evaluate
        the outputs of the model vs. the human comments from Personality-Captions
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
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
        default=10,
        help='number of images to show \
                           to turker',
    )
    argparser.add_argument(
        '--data-path', type=str, default='', help='where to save data'
    )
    argparser.add_argument(
        '--eval-data-path',
        type=str,
        default='',
        help='where to load data to rank from. Leave '
        'blank to use Personality-Captions data',
    )
    argparser.add_argument(
        '-ck1',
        '--compare-key-1',
        type=str,
        default='comment',
        help='key of first option to compare',
    )
    argparser.add_argument(
        '-ck2',
        '--compare-key-2',
        type=str,
        default='comment',
        help='key of second option to compare',
    )
    argparser.add_argument(
        '--show-personality',
        default=True,
        type='bool',
        help='whether to show the personality',
    )
    PersonalityCaptionsTeacher.add_cmdline_args(argparser)
    opt = argparser.parse_args()
    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)
    if 'data_path' not in opt or opt['data_path'] == '':
        opt['data_path'] = os.getcwd() + '/data/' + opt['task']
    if opt.get('eval_data_path') == '':
        opt['eval_data_path'] = os.path.join(
            opt['datapath'], 'personality_captions/train.json'
        )
    opt.update(task_config)

    mturk_agent_ids = [CHOOSER]
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_ids)

    example_generator = ExampleGenerator(opt)
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
            world = MTurkPersonalityCaptionsStackRankWorld(
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
