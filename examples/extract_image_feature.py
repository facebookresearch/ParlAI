# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and load/extract the 
image features. 

For example, to extract the image feature of COCO images:
`python examples/extract_image_feature.py -t vqa_v1 -im resnet152`.

The CNN model and layer is specified at `--image-cnntype` and `--image-layernum` 
in `parlai.core.image_featurizers`. 

For more options, check `parlai.core.image_featurizers`
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.image_featurizers import ImageLoader

import random

def main():
    random.seed(42)

    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument('-n', '--num-examples', default=10)
    parser.set_defaults(datatype='train:ordered')

    ImageLoader.add_cmdline_args(parser)
    opt = parser.parse_args()

    opt['no_cuda'] = False
    opt['gpu'] = 0
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs.
    with world:
        for k in range(int(opt['num_examples'])):
            world.parley()
            print(world.display() + '\n~~')
            if world.epoch_done():
                print('EPOCH DONE')
                break

if __name__ == '__main__':
    main()
