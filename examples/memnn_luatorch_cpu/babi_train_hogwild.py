# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.agents.remote_agent.agents import RemoteAgent
from parlai.tasks.babi.teachers import TaskTeacher
from parlai.core.worlds import DialogPartnerWorld, HogwildWorld
from parlai.core.params import ParlaiParser

import sys
import time

def main():
    # Get command line arguments
    argparser = ParlaiParser()
    RemoteAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    if opt['numthreads'] == 1:
        print('WARNING: hogwild does not work with 1 thread (port mismatch)')
        sys.exit(-1)

    opt['datatype'] = 'train'
    teacher_train = TaskTeacher(opt)
    opt['numexs'] = len(teacher_train)
    agent = RemoteAgent(opt)

    valid_opt = opt.copy()
    valid_opt['datatype'] = 'valid'
    teacher_valid = TaskTeacher(valid_opt)

    start = time.time()
    # train / valid loop with synchronized barriers (and fun `with` statement)
    with DialogPartnerWorld(valid_opt, [teacher_valid, agent]) as world_valid, (
            HogwildWorld(opt, [teacher_train, agent])) as world_train:
        for _ in range(3):
            print('[ training ]')
            for _ in range(len(teacher_train) * 50):  # do fifty epochs of train
                world_train.parley()

            world_train.synchronize()
            print('[ training summary. ]')
            print(teacher_train.report())

            print('[ validating ]')
            for _ in range(len(teacher_valid)):  # check valid accuracy
                world_valid.parley()

            print('[ validating summary. ]')
            report_valid = teacher_valid.report()
            print(report_valid)
            if report_valid['accuracy'] > 0.95:
                break

    print('finished in {} s'.format(round(time.time() - start, 2)))

if __name__ == '__main__':
    main()
