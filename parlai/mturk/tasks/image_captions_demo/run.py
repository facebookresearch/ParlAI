# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld

import base64
import os
from parlai.mturk.tasks.image_captions_demo.task_config import task_config

from io import BytesIO
from PIL import Image


def load_image(path):
    return Image.open(path).convert('RGB')


def get_image_encoded_from_path(path):
    img = load_image(path)
    buffered = BytesIO()
    img.save(buffered, format='JPEG')
    encoded = str(base64.b64encode(buffered.getvalue()).decode('ascii'))
    return encoded


def main():
    """
    Image captioning to demonstrate the capabilities of the static ParlAI-MTurk
    interface.
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['frontend_version'] = 1
    opt.update(task_config)

    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)
    if 'data_path' not in opt:
        opt['data_path'] = os.getcwd() + '/data/' + opt['task']
    # build_pc_data(opt)
    mturk_manager = StaticMTurkManager(opt=opt)
    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()

        mturk_manager.ready_to_accept_workers()
        mturk_manager.create_hits()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for w in workers:
                w.id = 'worker'

        def run_conversation(mturk_manager, opt, workers):
            image = '/Users/jju/ParlAI/data/yfcc_images/d06b45856ea94a4a26aa112f79a292e2.jpg'
            image_data = get_image_encoded_from_path(image)
            world = StaticMTurkTaskWorld(
                opt,
                mturk_agent=workers[0],
                task_data={'image': image_data, 'word_min': 5, 'word_max': 7},
            )
            while not world.episode_done():
                world.parley()

            world.shutdown()

            return world.prep_save_data(workers)

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
