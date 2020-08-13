#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import torch

from parlai.core.params import ParlaiParser
from parlai.core.teachers import create_task_agent_from_taskname
import parlai.utils.testing as testing_utils

BASE_IMAGE_ARGS = {
    "no_cuda": False,
    "task": "integration_tests:ImageTeacher",
    "image_size": 256,
    "image_cropsize": 224,
    "image_features_dim": 2048,
}

IMAGE_MODE_TO_DIM = {
    "resnet152": torch.Size([2048]),
    "resnet152_spatial": torch.Size([1, 2048, 7, 7]),
    "resnext101_32x48d_wsl": torch.Size([2048]),
    "resnext101_32x48d_wsl_spatial": torch.Size([1, 2048, 7, 7]),
}


@testing_utils.skipUnlessTorch
@testing_utils.skipUnlessGPU
class TestImageLoader(unittest.TestCase):
    """
    Unit Tests for the ImageLoader.
    """

    def test_image_loader(self):
        """
        Test that model correctly handles text task.
        """
        opt = ParlaiParser().parse_args([])
        opt.update(BASE_IMAGE_ARGS)
        for image_mode, dim in IMAGE_MODE_TO_DIM.items():
            opt["image_mode"] = image_mode
            teacher = create_task_agent_from_taskname(opt)[0]
            teacher_act = teacher.get(0)
            self.assertEquals(
                teacher_act["image"].size(),
                dim,
                f"dim mismatch for image mode {image_mode}",
            )


if __name__ == '__main__':
    unittest.main()
