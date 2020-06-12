#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Provide functionality for loading images.
"""

import parlai.core.build_data as build_data
import parlai.utils.logging as logging

import os
from PIL import Image
from zipfile import ZipFile

_greyscale = '  .,:;crsA23hHG#98&@'
_cache_size = 84000


class ImageLoader:
    """
    Extract image feature using pretrained CNN network.
    """

    def __init__(self, opt):
        self.opt = opt.copy()
        self.use_cuda = False
        self.netCNN = None
        self.im = opt.get('image_mode', 'no_image_model')
        if self.im not in ['no_image_model', 'raw', 'ascii']:
            if 'image_mode' not in opt or 'image_size' not in opt:
                raise RuntimeError(
                    'Need to add image arguments to opt. See '
                    'parlai.core.params.ParlaiParser.add_image_args'
                )
            self.image_mode = opt['image_mode']
            self.image_size = opt['image_size']
            self.crop_size = opt['image_cropsize']
            self._lazy_import_torch()
            self._init_transform()
            if 'resnet' in self.image_mode:
                self._init_resnet_cnn()
            elif 'resnext' in self.image_mode:
                self._init_resnext_cnn()
            else:
                raise RuntimeError(
                    'Image mode {} not supported'.format(self.image_mode)
                )

    def _lazy_import_torch(self):
        try:
            import torch
        except ImportError:
            raise ImportError('Need to install Pytorch: go to pytorch.org')
        import torchvision
        import torchvision.transforms as transforms
        import torch.nn as nn

        self.use_cuda = not self.opt.get('no_cuda', False) and torch.cuda.is_available()
        if self.use_cuda:
            logging.debug(f'Using CUDA')
            torch.cuda.set_device(self.opt.get('gpu', -1))
        self.torch = torch
        self.torchvision = torchvision
        self.transforms = transforms
        self.nn = nn

    def _init_transform(self):
        # initialize the transform function using torch vision.
        self.transform = self.transforms.Compose(
            [
                self.transforms.Scale(self.image_size),
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
        self.netCNN = self.nn.Sequential(
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
            model = self.torch.hub.load('facebookresearch/WSL-Images', self.image_mode)
            # cut off layer for ImageNet classification
            self.netCNN = self.nn.Sequential(*list(model.children())[:-1])
        except RuntimeError as e:
            # Perhaps specified one of the wrong model names
            logging.error(
                'If you have specified one of the resnext101 wsl models, '
                'please make sure it is one of the following: \n'
                'resnext101_32x8d_wsl, resnext101_32x16d_wsl, '
                'resnext101_32x32d_wsl, resnext101_32x48d_wsl'
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
        switcher = {
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
        }

        if self.image_mode not in switcher:
            raise NotImplementedError(
                'image preprocessing mode'
                + '{} not supported yet'.format(self.image_mode)
            )

        return switcher.get(self.image_mode)

    @classmethod
    def get_available_model_names(cls):
        """
        Get a list of the available model variants in this ImageLoader.
        """
        return [
            'resnet152',
            'resnet101',
            'resnet50',
            'resnet34',
            'resnet18',
            'resnet152_spatial',
            'resnet101_spatial',
            'resnet50_spatial',
            'resnet34_spatial',
            'resnet18_spatial',
            'resnext101_32x8d_wsl',
            'resnext101_32x16d_wsl',
            'resnext101_32x32d_wsl',
            'resnext101_32x48d_wsl',
        ]

    def extract(self, image, path=None):
        # check whether initialize CNN network.
        if not self.netCNN:
            self.init_cnn(self.opt)
        # extract the image feature
        transform = self.transform(image).unsqueeze(0)
        if self.use_cuda:
            transform = transform.cuda()
        with self.torch.no_grad():
            feature = self.netCNN(transform)
        # save the feature
        if path is not None:
            self.torch.save(feature.cpu(), path)
        return feature

    def _img_to_ascii(self, path):
        im = Image.open(path)
        im.thumbnail((60, 40), Image.BICUBIC)
        im = im.convert('L')
        asc = []
        for y in range(0, im.size[1]):
            for x in range(0, im.size[0]):
                lum = 255 - im.getpixel((x, y))
                asc.append(_greyscale[lum * len(_greyscale) // 256])
            asc.append('\n')
        return ''.join(asc)

    def load(self, path):
        """
        Load from a given path.
        """
        opt = self.opt
        mode = opt.get('image_mode', 'raw')
        is_zip = False
        if mode is None or mode == 'no_image_model':
            # don't need to load images
            return None
        elif '.zip' in path:
            # assume format path/to/file.zip/image_name.jpg
            is_zip = True
            sep = path.index('.zip') + 4
            zipname = path[:sep]
            file_name = path[sep + 1 :]
            path = ZipFile(zipname, 'r').open(file_name)
            task = opt['task']
            prepath = os.path.join(opt['datapath'], task)
            imagefn = ''.join(zipname.strip('.zip').split('/')[-2:]) + path.name
        if mode == 'raw':
            # raw just returns RGB values
            return Image.open(path).convert('RGB')
        elif mode == 'ascii':
            # convert images to ascii ¯\_(ツ)_/¯
            return self._img_to_ascii(path)
        else:
            # otherwise, looks for preprocessed version under 'mode' directory
            if not is_zip:
                prepath, imagefn = os.path.split(path)
            dpath = os.path.join(prepath, mode)
            if not os.path.exists(dpath):
                build_data.make_dir(dpath)
            imagefn = imagefn.split('.')[0]
            new_path = os.path.join(prepath, mode, imagefn)
            if not os.path.isfile(new_path):
                return self.extract(Image.open(path).convert('RGB'), new_path)
            else:
                return self.torch.load(new_path)
