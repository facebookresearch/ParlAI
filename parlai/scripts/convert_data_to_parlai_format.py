#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Convert a dataset into the ParlAI text format.

## Examples

```shell
parlai convert_data_to_parlai_format -t babi:task1k:1 --outfile /tmp/dump
```
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import msg_to_str, TimeLogger
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script
import random
import tempfile


def dump_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    opt.log()
    ignorefields = opt.get('ignore_fields', '')
    if opt['outfile'] is None:
        outfile = tempfile.mkstemp(
            prefix='{}_{}_'.format(opt['task'], opt['datatype']), suffix='.txt'
        )[1]
    else:
        outfile = opt['outfile']

    if opt['num_examples'] == -1:
        num_examples = world.num_examples()
    else:
        num_examples = opt['num_examples']
    log_timer = TimeLogger()

    logging.debug('starting to convert...')
    logging.info(f'saving output to {outfile}')
    fw = open(outfile, 'w')
    for _ in range(num_examples):
        world.parley()
        acts = world.get_acts()
        value = acts[0].get('labels', acts[0].pop('eval_labels', None))
        acts[0].force_set('labels', value)
        txt = msg_to_str(acts[0], ignore_fields=ignorefields)
        fw.write(txt + '\n')
        if acts[0].get('episode_done', False):
            fw.write('\n')

        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            logging.info(text)

        if world.epoch_done():
            logging.info('epoch done')
            break
    fw.close()


def setup_args():
    # Get command line arguments
    parser = ParlaiParser(description='Dump a task to a standardized format')
    parser.add_argument(
        '-n',
        '--num-examples',
        default=-1,
        type=int,
        help='Total number of exs to convert, -1 to convert all examples',
    )
    parser.add_argument(
        '-of',
        '--outfile',
        default=None,
        type=str,
        help='Output file where to save, by default will be created in tmp',
    )
    parser.add_argument(
        '-if',
        '--ignore-fields',
        default='id',
        type=str,
        help='Ignore these fields from the message (returned with .act() )',
    )
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.set_defaults(datatype='train:stream')
    return parser


@register_script('convert_to_parlai', hidden=True)
class ConvertDataToParlaiFormat(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return dump_data(self.opt)


if __name__ == '__main__':
    random.seed(42)
    ConvertDataToParlaiFormat.main()
