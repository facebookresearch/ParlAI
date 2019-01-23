# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import \
    MTurkImageChatStackRankWorld, CHOOSER, ExampleGenerator, RoleOnboardWorld
from parlai.tasks.image_chat.agents import ImageChatTeacher
from parlai.tasks.image_chat.build import build as build_ic
from task_configs.task_config_first_response import task_config as config_first
from task_configs.task_config_second_response import task_config as config_second
import os

round_choices = ['first_response', 'second_response']


def main():
    """
        Human Evaluation of various responses to comments on images.

        A turker is shown an image and some dialog history. Then, the
        turker is asked to choose which response they think is more engaging.

        If no `--eval-data-path` is given, the data from the original
        Image-Chat dataset is used.

        To use your own data, please specify `--eval-data-path`, a path to an
        appropriate json file with a list of examples, where each example
        has the following structure:
            {
                'image_hash': <hash of image>,
                'dialog': [(personality, text), ...] - list of personality, text tuples
                'personality': <personality of responses to compare>
                '<compare_key_1>': <first response to compare>,
                '<compare_key_2>': <second option to compare>,
                .
                .
                .
            }
        Note that compare_key_1 and compare_key_2 can be any field, as long as they
        map to a string response.

        Example Scenario:
            Suppose you have the original Image-Chat dataset, and
            you would like to compare the outputs of your model called `model`.

            Your data may look like the following:
            [{
                'image_hash': hashforimageofcat,
                'dialog': [
                    ('Sweet', 'What a cute cat!'),
                    ('Neutral', 'Just looks like a plain cat to me')
                ]
                'personality': 'Sweet',
                'comment': 'It really is adorable if you look!', # Human Comment
                'model_comment': 'You'll love it if you pet it!' # Model Comment
            }, ...]

            Thus, you would specify `-ck1 comment -ck2 model_comment` to evaluate
            the outputs of the model vs. the human comments from Personality-Captions

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
                           default=3600*24, help='how long to wait for  \
                           auto approval')
    argparser.add_argument('--data-path', type=str,
                           default='', help='where to save data')
    argparser.add_argument('--eval-data-path', type=str, default='',
                           help='where to load data to rank from. Leave '
                                'blank to use Image-Chat data')
    argparser.add_argument('-ck1', '--compare-key-1', type=str,
                           default='comment',
                           help='key of first comparable')
    argparser.add_argument('-ck2', '--compare-key-2', type=str,
                           default='comment',
                           help='key of first comparable')
    argparser.add_argument('-rnd', '--dialog-round', type=str, default='first_response',
                           choices=round_choices,
                           help='which dialog round to show')
    argparser.add_argument('--show-personality', default=True, type='bool',
                           help='whether to show the personality')
    ImageChatTeacher.add_cmdline_args(argparser)
    opt = argparser.parse_args()
    build_ic(opt)
    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)
    if 'data_path' not in opt or opt['data_path'] == '':
        opt['data_path'] = os.getcwd() + '/data/' + opt['task']
    if opt.get('eval_data_path') == '':
        opt['eval_data_path'] = os.path.join(
            opt['datapath'],
            'image_chat/test.json')
    config = config_first if opt['dialog_round'] == 'first_response' else config_second
    opt.update(config)

    mturk_agent_ids = [CHOOSER]
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=mturk_agent_ids
    )

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
            world = MTurkImageChatStackRankWorld(
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
