#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic example which iterates through the tasks specified and load/extract the image
features.

For more options, check `parlai.core.image_featurizers`

## Examples

To extract the image feature of COCO images:

```shell
parlai extract_image_feature -t vqa_v1 -im resnet152
```
"""
import copy
import tqdm

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, False, 'Load/extract image features')
    return parser


def extract_feats(opt):
    if isinstance(opt, ParlaiParser):
        logging.error('extract_feats should be passed opt not parser')
        opt = opt.parse_args()
    # Get command line arguments
    opt = copy.deepcopy(opt)
    dt = opt['datatype'].split(':')[0] + ':ordered'
    opt['datatype'] = dt
    opt['no_cuda'] = False
    opt['gpu'] = 0
    opt['num_epochs'] = 1
    opt['num_load_threads'] = 20
    opt.log()
    logging.info("Loading Images")
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    total_exs = world.num_examples()
    pbar = tqdm.tqdm(unit='ex', total=total_exs)
    while not world.epoch_done():
        world.parley()
        pbar.update()
    pbar.close()

    logging.info("Finished extracting images")


@register_script('extract_image_feature', hidden=True)
class ExtractImgFeatures(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return extract_feats(self.opt)


if __name__ == '__main__':
    ExtractImgFeatures.main()
