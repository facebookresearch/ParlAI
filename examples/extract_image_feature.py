# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and load/extract the
image features.

For example, to extract the image feature of COCO images:
`python examples/extract_image_feature.py -t vqa_v1 -im resnet152`.

For more options, check `parlai.core.image_featurizers`
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.utils import ProgressLogger


def main():
    # Get command line arguments
    parser = ParlaiParser(True, False)
    parser.set_defaults(datatype='train:ordered')

    opt = parser.parse_args()
    bsz = opt.get('batchsize', 1)
    opt['no_cuda'] = False
    opt['gpu'] = 0
    opt['num_epochs'] = 1
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    logger = ProgressLogger(should_humanize=False)
    print("Beginning image extraction...")
    exs_seen = 0
    total_exs = world.num_examples()
    while not world.epoch_done():
        world.parley()
        exs_seen += bsz
        logger.log(exs_seen, total_exs)
    print("Finished extracting images")


if __name__ == '__main__':
    main()
