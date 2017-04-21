# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.agents.remote_agent.agents import RemoteAgent
from parlai.tasks.wikiqa.teachers import DefaultTeacher
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

    try:
        opt['datatype'] = 'train'
        teacher_train = DefaultTeacher(opt)
        opt['numexs'] = len(teacher_train)
        agent = RemoteAgent(opt)

        valid_opt = opt.copy()
        valid_opt['datatype'] = 'test'
        teacher_valid = DefaultTeacher(valid_opt)

        world_train = HogwildWorld(opt, [teacher_train, agent])
        world_valid = DialogPartnerWorld(valid_opt, [teacher_valid, agent])

        start = time.time()
        # train / valid loop with synchronized barriers
        for _ in range(10):
            print('[ training ]')
            for _ in range(len(teacher_train) * 10):  # do ten epochs of train
                world_train.parley()

            world_train.synchronize()
            print('[ training summary. ]')
            print(teacher_train.report())

            print('[ validating ]')
            for _ in range(len(teacher_valid)):  # check valid accuracy
                world_valid.parley()

            print('[ validating summary. ]')
            print(teacher_valid.report())
    # eat some of the errors caused by hitting ctrl-c to clean up exit message
    except KeyboardInterrupt:
        pass
    except ConnectionResetError:
        pass
    finally:
        world_train.shutdown()
        world_valid.shutdown()

    print('finished in {} s'.format(round(time.time() - start, 2)))

if __name__ == '__main__':
    main()
