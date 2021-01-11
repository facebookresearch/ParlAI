#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generates a dictionary file from the training data.

## Examples

```bash
# learn the vocabulary from one task, then train on another task.
parlai build_dict -t convai2 --dict-file premade.dict
parlai train_model -t squad --dict-file premade.dict -m seq2seq
```
"""

from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser, str2class
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.utils.distributed import is_distributed
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
import copy
import tqdm


def setup_args(parser=None, hidden=True):
    if parser is None:
        parser = ParlaiParser(True, True, 'Build a dictionary.')
    dict_loop = parser.add_argument_group('Dictionary Loop Arguments')
    dict_loop.add_argument(
        '--dict-maxexs',
        default=-1,
        type=int,
        help='max number of examples to build dict on',
        hidden=hidden,
    )
    dict_loop.add_argument(
        '--dict-include-valid',
        default=False,
        type='bool',
        help='Include validation set in dictionary building ' 'for task.',
        hidden=hidden,
    )
    dict_loop.add_argument(
        '--dict-include-test',
        default=False,
        type='bool',
        help='Include test set in dictionary building for task.',
        hidden=hidden,
    )
    dict_loop.add_argument(
        '-ltim', '--log-every-n-secs', type=float, default=10, hidden=hidden
    )
    DictionaryAgent.add_cmdline_args(parser, partial_opt=None)
    return parser


def build_dict(opt, skip_if_built=False):
    if isinstance(opt, ParlaiParser):
        logging.error('Should be passed opt not Parser')
        opt = opt.parse_args()
    if not opt.get('dict_file'):
        logging.error(
            'Tried to build dictionary but `--dict-file` is not set. Set '
            'this param so the dictionary can be saved.'
        )
        return
    if skip_if_built and PathManager.exists(opt['dict_file']):
        # Dictionary already built, skip all loading or setup
        logging.debug("dictionary already built.")
        return None

    if opt.get('dict_class'):
        # Custom dictionary class
        dictionary = str2class(opt['dict_class'])(opt)
    else:
        # Default dictionary class
        dictionary = DictionaryAgent(opt)

    if PathManager.exists(opt['dict_file']) or (
        hasattr(dictionary, 'is_prebuilt') and dictionary.is_prebuilt()
    ):
        # Dictionary already built, return loaded dictionary agent
        logging.debug("dictionary already built.")
        return dictionary

    if is_distributed():
        raise ValueError('Dictionaries should be pre-built before distributed train.')

    ordered_opt = copy.deepcopy(opt)
    cnt = 0
    # we use train set to build dictionary

    ordered_opt['batchsize'] = 1
    # Set this to none so that image features are not calculated when Teacher is
    # instantiated while building the dict
    ordered_opt['image_mode'] = 'no_image_model'

    ordered_opt.log()

    datatypes = ['train:ordered:stream']
    if opt.get('dict_include_valid'):
        datatypes.append('valid:stream')
    if opt.get('dict_include_test'):
        datatypes.append('test:stream')
    cnt = 0
    for dt in datatypes:
        ordered_opt['datatype'] = dt
        world_dict = create_task(ordered_opt, dictionary)
        # pass examples to dictionary
        log_time = TimeLogger()
        total = world_dict.num_examples()
        if opt['dict_maxexs'] >= 0:
            total = min(total, opt['dict_maxexs'])

        log_every_n_secs = opt.get('log_every_n_secs', None)
        if log_every_n_secs:
            pbar = tqdm.tqdm(
                total=total, desc='Building dictionary', unit='ex', unit_scale=True
            )
        else:
            pbar = None
        while not world_dict.epoch_done():
            cnt += 1
            if cnt > opt['dict_maxexs'] and opt['dict_maxexs'] >= 0:
                logging.info('Processed {} exs, moving on.'.format(opt['dict_maxexs']))
                # don't wait too long...
                break
            world_dict.parley()
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()

    dictionary.save(opt['dict_file'], sort=True)
    logging.info(
        f'dictionary built with {len(dictionary)} tokens '
        f'in {log_time.total_time():.1f}s'
    )
    return dictionary


@register_script('build_dict', hidden=True)
class BuildDict(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args(hidden=False)

    def run(self):
        return build_dict(self.opt)


if __name__ == '__main__':
    BuildDict.main()
