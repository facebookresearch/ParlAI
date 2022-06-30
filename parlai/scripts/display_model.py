#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which iterates through the tasks specified and runs the given model on
them.

## Examples ```shell parlai display_model --task babi:task1k:1 --model repeat_label
parlai display_model --task convai2 --model-file "/path/to/model_file"  --datatype test
```
"""  # noqa: E501

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
from parlai.core.logs import ClearMLLogger

from parlai.utils.distributed import is_primary_worker

import random


def simple_display(opt, world, turn, clearml_logger, _k):
    if opt['batchsize'] > 1:
        raise RuntimeError('Simple view only support batchsize=1')
    teacher, response = world.get_acts()
    if turn == 0:
        text = "- - - NEW EPISODE: " + teacher.get('id', "[no agent id]") + "- - -"
        print(colorize(text, 'highlight'))
    text = teacher.get('text', '[no text field]')
    print(colorize(text, 'text'))
    response_text = response.get('text', 'No response')
    labels = teacher.get('labels', teacher.get('eval_labels', ['[no labels field]']))
    labels = '|'.join(labels)

    if opt['clearml_log'] and is_primary_worker():
        debug_sample = (
            text + "\n" + '    labels: ' + labels + "\n" + ' model: ' + response_text
        )
        clearml_logger.log_debug_samples(opt['task'], debug_sample, _k)

    print(colorize('    labels: ' + labels, 'labels'))
    print(colorize('     model: ' + response_text, 'text2'))


def setup_args():
    parser = ParlaiParser(True, True, 'Display model predictions.')
    parser.add_argument('-n', '-ne', '--num-examples', default=10)
    parser.add_argument(
        '--display-add-fields',
        type=str,
        default='',
        help='Display these fields when verbose is off (e.g., "--display-add-fields label_candidates,beam_texts")',
    )

    ClearMLLogger.add_cmdline_args(parser, partial_opt=None)

    # by default we want to display info about the validation set
    parser.set_defaults(datatype='valid')

    return parser


def display_model(opt):
    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)
    agent.opt.log()

    if opt['clearml_log'] and is_primary_worker():
        clearml_logger = ClearMLLogger(opt)
    else:
        clearml_logger = None

    # Show some example dialogs.
    turn = 0
    with world:
        for _k in range(int(opt['num_examples'])):
            world.parley()
            if opt['verbose'] or opt.get('display_add_fields', ''):
                print(world.display() + "\n~~")
                if opt['clearml_log'] and is_primary_worker():
                    clearml_logger.log_debug_samples(opt['task'], world.display(), _k)
            else:
                simple_display(opt, world, turn, clearml_logger, _k)
            turn += 1
            if world.get_acts()[0]['episode_done']:
                turn = 0
            if world.epoch_done():
                logging.info("epoch done")
                turn = 0
                break

    if opt['clearml_log'] and is_primary_worker():
        # Upload the Model as artifact
        clearml_logger.upload_artifact("Model", opt["model_file"])
        # Close ClearML Task
        clearml_logger.close()


@register_script('display_model', aliases=['dm'])
class DisplayModel(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        display_model(self.opt)


if __name__ == '__main__':
    DisplayModel.main()
