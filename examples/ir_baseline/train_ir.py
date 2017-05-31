# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train IR model by building dictionary and applying TFIDF style weighting.
   After training, computes validation error.
   Run with, e.g.:
   python examples/ir_baseline/train_ir.py -t dialog_babi:Task:1
   or
   python examples/ir_baseline/train_ir.py -t dialog_babi:Task:1 -mf '/tmp/ir.model'
"""

from parlai.agents.ir_baseline.agents import IrBaselineAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import DialogPartnerWorld
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

def main():
    # Get command line arguments
    parser = ParlaiParser(True, True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    opt = parser.parse_args()
    # Create model and assign it to the specified task
    opt['model'] = 'ir_baseline'
    agent = create_agent(opt)
    opt['datatype'] = 'train:ordered'
    world = create_task(opt, agent)

    # Show some example dialogs:
    for _ in range(len(world)):
        world.parley()
        if opt['display_examples']:
            print(world.display() + "\n~~")
        if world.epoch_done():
            print("TRAIN EPOCH DONE")
            break
    world.shutdown()
    if opt['model_file']:
        agent.save(opt['model_file'])

    # Now eval on validation data.
    opt['datatype'] = 'valid'
    valid_world = create_task(opt, agent)
    for _ in range(len(valid_world)):
        valid_world.parley()
        print("---")
        if opt['display_examples']:
            print(valid_world.display() + "\n~~")
        print(valid_world.report())
        if valid_world.epoch_done():
            print("VALIDATION DONE")
            break
    valid_world.shutdown()


if __name__ == '__main__':
    main()


