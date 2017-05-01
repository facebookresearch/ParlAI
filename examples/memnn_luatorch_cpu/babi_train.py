# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.agents.remote_agent.agents import RemoteAgent
from parlai.tasks.babi.teachers import TaskTeacher
from parlai.core.worlds import DialogPartnerWorld
from parlai.core.params import ParlaiParser

import time

def main():
    # Get command line arguments
    argparser = ParlaiParser()
    RemoteAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    opt['datatype'] = 'train'
    teacher_train = TaskTeacher(opt)
    opt['numexs'] = len(teacher_train)
    agent = RemoteAgent(opt)

    opt['datatype'] = 'valid'
    teacher_valid = TaskTeacher(opt)

    start = time.time()
    # train / valid loop using with statements (fun!) runs world.shutdown() after
    with DialogPartnerWorld(opt, [teacher_valid, agent]) as world_valid, (
            DialogPartnerWorld(opt, [teacher_train, agent])) as world_train:
        for _ in range(10):
            print('[ training ]')
            for _ in range(len(teacher_train)):  # do one epoch of train
                world_train.parley()

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

        # show some example dialogs after training:
        for _k in range(3):
                world_valid.parley()
                print(world_valid.query['text'])
                print("A: " + world_valid.reply['text'])

    print('finished in {} s'.format(round(time.time() - start, 2)))

if __name__ == '__main__':
    main()
