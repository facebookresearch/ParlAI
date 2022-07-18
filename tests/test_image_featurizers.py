#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from PIL import Image
import unittest
import torch
import zipfile

from parlai.core.image_featurizers import ImageLoader
from parlai.core.params import ParlaiParser
from parlai.core.teachers import create_task_agent_from_taskname
from parlai.utils.io import PathManager
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


@unittest.skip
@testing_utils.skipUnlessVision
class TestImageLoader(unittest.TestCase):
    """
    Unit Tests for the ImageLoader.
    """

    def _base_test_loader(self, image_mode_partial: str, no_cuda: bool = False):
        """
        Test for given partial image mode.
        """
        opt = ParlaiParser().parse_args([])
        opt.update(BASE_IMAGE_ARGS)
        opt['no_cuda'] = no_cuda
        for image_mode, dim in IMAGE_MODE_TO_DIM.items():
            if image_mode_partial not in image_mode:
                continue
            opt["image_mode"] = image_mode
            teacher = create_task_agent_from_taskname(opt)[0]
            teacher_act = teacher.get(0)
            self.assertEquals(
                teacher_act["image"].size(),
                dim,
                f"dim mismatch for image mode {image_mode}",
            )
        torch.cuda.empty_cache()

    @testing_utils.skipUnlessGPU
    def test_resnet(self):
        self._base_test_loader("resnet")

    @testing_utils.skipUnlessGPU
    def test_resnext(self):
        self._base_test_loader("resnext")

    def test_other_image_modes(self):
        """
        Test non-featurized image modes.
        """
        with testing_utils.tempdir() as tmp:
            image_file = 'tmp.jpg'
            image_path = os.path.join(tmp, image_file)
            image_zip_path = os.path.join(tmp, 'tmp.zip')
            image = Image.new('RGB', (16, 16), color=0)

            with PathManager.open(image_path, 'wb') as fp:
                image.save(fp, 'JPEG')

            with zipfile.ZipFile(
                PathManager.open(image_zip_path, 'wb'), mode='w'
            ) as zipf:
                zipf.write(image_path, arcname=image_file)

            for im in ['raw', 'ascii']:
                loader = ImageLoader({"image_mode": im})
                loader.load(image_path)
                loader.load(f"{image_zip_path}/{image_file}")


if __name__ == '__main__':
    unittest.main()
