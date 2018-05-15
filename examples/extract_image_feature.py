# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and load/extract
the image features.

For example, to extract the image feature of COCO images:
`python examples/extract_image_feature.py -t vqa_v1 -im resnet152`.

For more options, check `parlai.core.image_featurizers`
"""
from parlai.scripts.extract_image_feature import setup_args, extract_feats

if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    extract_feats(opt)
