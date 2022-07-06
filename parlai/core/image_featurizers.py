#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Provide functionality for loading images.
"""

import parlai.core.build_data as build_data
import parlai.utils.logging as logging
from parlai.utils.io import PathManager

import os
from PIL import Image
import torch
from zipfile import ZipFile

_greyscale = '  .,:;crsA23hHG#98&@'
_cache_size = 84000

# Mapping from image mode to (torch_instantiation_str, layer_cutoff_idx)
IMAGE_MODE_SWITCHER = {
    'resnet152': ['resnet152', -1],
    'resnet101': ['resnet101', -1],
    'resnet50': ['resnet50', -1],
    'resnet34': ['resnet34', -1],
    'resnet18': ['resnet18', -1],
    'resnet152_spatial': ['resnet152', -2],
    'resnet101_spatial': ['resnet101', -2],
    'resnet50_spatial': ['resnet50', -2],
    'resnet34_spatial': ['resnet34', -2],
    'resnet18_spatial': ['resnet18', -2],
    'resnext101_32x8d_wsl': ['resnext101_32x8d_wsl', -1],
    'resnext101_32x16d_wsl': ['resnext101_32x16d_wsl', -1],
    'resnext101_32x32d_wsl': ['resnext101_32x32d_wsl', -1],
    'resnext101_32x48d_wsl': ['resnext101_32x48d_wsl', -1],
    'resnext101_32x8d_wsl_spatial': ['resnext101_32x8d_wsl', -2],
    'resnext101_32x16d_wsl_spatial': ['resnext101_32x16d_wsl', -2],
    'resnext101_32x32d_wsl_spatial': ['resnext101_32x32d_wsl', -2],
    'resnext101_32x48d_wsl_spatial': ['resnext101_32x48d_wsl', -2],
}


class ImageLoader:
    """
    Extract image feature using pretrained CNN network.
    """

    def __init__(self, opt):
        self.opt = opt.copy()
        self.use_cuda = False
        self.netCNN = None
        self.image_mode = opt.get('image_mode', 'no_image_model')
        self.use_cuda = not self.opt.get('no_cuda', False) and torch.cuda.is_available()
        if self.image_mode not in ['no_image_model', 'raw', 'ascii']:
            if 'image_mode' not in opt or 'image_size' not in opt:
                raise RuntimeError(
                    'Need to add image arguments to opt. See '
                    'parlai.core.params.ParlaiParser.add_image_args'
                )
            self.image_size = opt['image_size']
            self.crop_size = opt['image_cropsize']
            self._init_transform()
            if 'resnet' in self.image_mode:
                self._init_resnet_cnn()
            elif 'resnext' in self.image_mode:
                self._init_resnext_cnn()
            else:
                raise RuntimeError(
                    'Image mode {} not supported'.format(self.image_mode)
                )

    @classmethod
    def is_spatial(cls, image_mode: str):
        """
        Return if image mode has spatial dimensionality.
        """
        return any([s in image_mode for s in ['spatial']])

    def _init_transform(self):
        # initialize the transform function using torch vision.
        try:
            import torchvision
            import torchvision.transforms

            self.torchvision = torchvision
            self.transforms = torchvision.transforms

        except ImportError:
            raise ImportError('Please install torchvision; see https://pytorch.org/')

        self.transform = self.transforms.Compose(
            [
                self.transforms.Resize(self.image_size),
                self.transforms.CenterCrop(self.crop_size),
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _init_resnet_cnn(self):
        """
        Lazily initialize preprocessor model.

        When image_mode is one of the ``resnet`` varieties
        """
        cnn_type, layer_num = self._image_mode_switcher()
        # initialize the pretrained CNN using pytorch.
        CNN = getattr(self.torchvision.models, cnn_type)

        # cut off the additional layer.
        self.netCNN = torch.nn.Sequential(
            *list(CNN(pretrained=True).children())[:layer_num]
        )

        if self.use_cuda:
            self.netCNN.cuda()

    def _init_resnext_cnn(self):
        """
        Lazily initialize preprocessor model.

        When image_mode is one of the ``resnext101_..._wsl`` varieties
        """
        try:
            cnn_type, layer_num = self._image_mode_switcher()
            model = torch.hub.load('facebookresearch/WSL-Images', cnn_type)
            # cut off layer for ImageNet classification
            # if spatial, cut off another layer for spatial features
            self.netCNN = torch.nn.Sequential(*list(model.children())[:layer_num])
        except RuntimeError as e:
            # Perhaps specified one of the wrong model names
            model_names = [m for m in IMAGE_MODE_SWITCHER if 'resnext101' in m]
            logging.error(
                'If you have specified one of the resnext101 wsl models, '
                'please make sure it is one of the following: \n'
                f"{', '.join(model_names)}"
            )
            raise e
        except AttributeError:
            # E.g. "module 'torch' has no attribute 'hub'"
            raise RuntimeError(
                'Please install the latest pytorch distribution to have access '
                'to the resnext101 wsl models (pytorch 1.1.0, torchvision 0.3.0)'
            )

        if self.use_cuda:
            self.netCNN.cuda()

    def _image_mode_switcher(self):
        if self.image_mode not in IMAGE_MODE_SWITCHER:
            raise NotImplementedError(
                'image preprocessing mode'
                + '{} not supported yet'.format(self.image_mode)
            )

        return IMAGE_MODE_SWITCHER.get(self.image_mode)

    @classmethod
    def get_available_model_names(cls):
        """
        Get a list of the available model variants in this ImageLoader.
        """
        return list(IMAGE_MODE_SWITCHER.keys())

    def extract(self, image, path=None):
        # check whether initialize CNN network.
        # extract the image feature
        if 'faster_r_cnn' not in self.image_mode:
            transform = self.transform(image).unsqueeze(0)
            if self.use_cuda:
                transform = transform.cuda()
            with torch.no_grad():
                feature = self.netCNN(transform)
        else:
            raise RuntimeError("detectron support has been removed.")
        # save the feature
        if path is not None:
            import parlai.utils.torch as torch_utils

            torch_utils.atomic_save(feature.cpu(), path)
        return feature

    def _img_to_ascii(self, im):
        im.thumbnail((60, 40), Image.BICUBIC)
        im = im.convert('L')
        asc = []
        for y in range(0, im.size[1]):
            for x in range(0, im.size[0]):
                lum = 255 - im.getpixel((x, y))
                asc.append(_greyscale[lum * len(_greyscale) // 256])
            asc.append('\n')
        return ''.join(asc)

    def _breakup_zip_filename(self, path):
        # assume format path/to/file.zip/image_name.jpg
        assert '.zip' in path
        sep = path.index('.zip') + 4
        zipname = path[:sep]
        file_name = path[sep + 1 :]
        return zipname, file_name

    def _get_prepath(self, path):
        if '.zip' in path:
            zipname, file_name = self._breakup_zip_filename(path)
            task = self.opt['task']
            prepath = os.path.join(self.opt['datapath'], task)
            imagefn = ''.join(zipname.strip('.zip').split('/')[-2:]) + path.name
            return prepath, imagefn
        else:
            prepath, imagefn = os.path.split(path)
            return prepath, imagefn

    def _load_image(self, path):
        """
        Return the loaded image in the path.
        """
        if '.zip' in path:
            zipname, file_name = self._breakup_zip_filename(path)
            with ZipFile(PathManager.open(zipname, 'rb')) as zipf:
                with zipf.open(file_name) as fh:
                    return Image.open(fh).convert('RGB')
        else:
            # raw just returns RGB values
            with PathManager.open(path, 'rb') as f:
                return Image.open(f).convert('RGB')

    def load(self, path):
        """
        Load from a given path.
        """
        mode = self.opt.get('image_mode', 'raw')
        if mode is None or mode == 'no_image_model':
            # don't need to load images
            return None
        elif mode == 'raw':
            return self._load_image(path)
        elif mode == 'ascii':
            # convert images to ascii ¯\_(ツ)_/¯
            return self._img_to_ascii(self._load_image(path))

        # otherwise, looks for preprocessed version under 'mode' directory
        prepath, imagefn = self._get_prepath(path)
        dpath = os.path.join(prepath, mode)
        if not PathManager.exists(dpath):
            build_data.make_dir(dpath)
        imagefn = imagefn.split('.')[0]
        new_path = os.path.join(prepath, mode, imagefn)
        if not PathManager.exists(new_path):
            return self.extract(self._load_image(path), new_path)
        else:
            with PathManager.open(new_path, 'rb') as f:
                return torch.load(f)
