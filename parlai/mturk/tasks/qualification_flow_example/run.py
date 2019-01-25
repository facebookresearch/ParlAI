#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.qualification_flow_example.worlds import \
    QualificationFlowOnboardWorld, QualificationFlowSoloWorld
from parlai.mturk.core.mturk_manager import MTurkManager
import parlai.mturk.core.mturk_utils as mturk_utils
from parlai.mturk.tasks.qualification_flow_example.task_config import \
    task_config
import os
import random


def main():
    completed_workers = []
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)

    mturk_agent_id = 'Worker'
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id]
    )
    mturk_manager.setup_server()
    qual_name = 'ParlAIExcludeQual{}t{}'.format(
        random.randint(10000, 99999), random.randint(10000, 99999))
    qual_desc = (
        'Qualification for a worker not correctly completing the '
        'first iteration of a task. Used to filter to different task pools.'
    )
    qualification_id = \
        mturk_utils.find_or_create_qualification(qual_name, qual_desc,
                                                 opt['is_sandbox'])
    print('Created qualification: ', qualification_id)

    def run_onboard(worker):
        world = QualificationFlowOnboardWorld(opt, worker)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    mturk_manager.set_onboard_function(onboard_function=run_onboard)

    try:
        mturk_manager.start_new_run()
        agent_qualifications = [{
            'QualificationTypeId': qualification_id,
            'Comparator': 'DoesNotExist',
            'RequiredToPreview': True
        }]
        mturk_manager.create_hits(qualifications=agent_qualifications)

        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(worker):
            worker[0].id = mturk_agent_id

        global run_conversation

        def run_conversation(mturk_manager, opt, workers):
            mturk_agent = workers[0]
            world = QualificationFlowSoloWorld(
                opt=opt,
                mturk_agent=mturk_agent,
                qualification_id=qualification_id,
                firstTime=(mturk_agent.worker_id not in completed_workers),
            )
            while not world.episode_done():
                world.parley()
            completed_workers.append(mturk_agent.worker_id)
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
        mturk_utils.delete_qualification(qualification_id, opt['is_sandbox'])
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
