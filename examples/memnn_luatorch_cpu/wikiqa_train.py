# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.agents.remote_agent.agents import RemoteAgent
from parlai.tasks.wikiqa.teachers import DefaultTeacher
from parlai.core.worlds import DialogPartnerWorld
from parlai.core.params import ParlaiParser

import time

def main():
    # Get command line arguments
    argparser = ParlaiParser()
    RemoteAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    opt['datatype'] = 'train'
    teacher_train = DefaultTeacher(opt)
    opt['numexs'] = len(teacher_train)
    agent = RemoteAgent(opt)

    opt['datatype'] = 'valid'
    teacher_valid = DefaultTeacher(opt)

    world_train = DialogPartnerWorld(opt, [teacher_train, agent])
    world_valid = DialogPartnerWorld(opt, [teacher_valid, agent])

    start = time.time()
    # train / valid loop
    for _ in range(100):
        print('[ training ]')
        for _ in range(len(teacher_train)):  # do one epoch of train
            world_train.parley()

        print('[ training summary. ]')
        print(teacher_train.report())

        print('[ validating ]')
        for _ in range(len(teacher_valid)):  # check valid accuracy
            world_valid.parley()

        print('[ validating summary. ]')
        print(teacher_valid.report())

    world_train.shutdown()
    world_valid.shutdown()

    print('finished in {} s'.format(round(time.time() - start, 2)))

if __name__ == '__main__':
    main()
