#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Provide functionality for loading images.
"""

import parlai.core.build_data as build_data
from parlai.core.opt import Opt
import parlai.utils.logging as logging
from parlai.utils.io import PathManager
from parlai.zoo.detectron.build import build

import os
from PIL import Image
import numpy as np
import torch
from typing import Dict, Tuple, List
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
    'faster_r_cnn_152_32x8d': ['', -1],
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
            elif 'faster_r_cnn_152_32x8d' in self.image_mode:
                self._init_faster_r_cnn()
            else:
                raise RuntimeError(
                    'Image mode {} not supported'.format(self.image_mode)
                )

    @classmethod
    def is_spatial(cls, image_mode: str):
        """
        Return if image mode has spatial dimensionality.
        """
        return any([s in image_mode for s in ['spatial', 'faster_r_cnn']])

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

    def _init_faster_r_cnn(self):
        """
        Initialize Detectron Model.
        """
        self.netCNN = DetectronFeatureExtractor(self.opt, self.use_cuda)

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
            feature = self.netCNN.get_detectron_features([image])[0]
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
                return self.torch.load(f)


class DetectronFeatureExtractor:
    """
    Code adapted from https://github.com/facebookresearch/mmf/blob/master/tools/scripts/
    features/extract_features_vmb.py.

    Docstrings and type annotations added post hoc.
    """

    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self, opt: Opt, use_cuda: bool = False):
        self.opt = opt
        self.use_cuda = use_cuda
        self.num_features = 100

        try:
            import cv2

            self.cv2 = cv2
        except ImportError:
            raise ImportError("Please install opencv: pip install opencv-python")
        try:
            import maskrcnn_benchmark  # noqa
        except ImportError:
            raise ImportError(
                'Please install vqa-maskrcnn-benchmark to use faster_r_cnn_152_32x8d features: '
                '1. git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git\n'
                '2. cd vqa-maskrcnn-benchmark\n'
                '3. git checkout 4c168a637f45dc69efed384c00a7f916f57b25b8 -b stable\n'
                '4. python setup.py develop'
            )
        self._build_detection_model()

    def _build_detection_model(self):
        """
        Build the detection model.

        Builds a CNN using the vqa-maskrcnn-benchmark repository.
        """
        from maskrcnn_benchmark.config import cfg
        from maskrcnn_benchmark.modeling.detector import build_detection_model
        from maskrcnn_benchmark.utils.model_serialization import load_state_dict

        dp = self.opt['datapath']
        build(dp)
        cfg_path = os.path.join(dp, 'models/detectron/detectron_config.yaml')
        model_path = os.path.join(dp, 'models/detectron/detectron_model.pth')

        cfg.merge_from_file(cfg_path)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        if self.use_cuda:
            model.to("cuda")
        model.eval()
        self.detection_model = model

    def _image_transform(
        self, img: "Image"
    ) -> Tuple[torch.Tensor, float, Dict[str, int]]:
        """
        Using Open-CV, perform image transform on a raw image.

        :param img:
            raw image to transform

        :return (img, scale, info):
            img: tensor representation of image
            scale: scale of image WRT self.MIN_SIZE & self.MAX_SIZE
            info: dict containing values for img width & height
        """
        im = np.array(img).astype(np.float32)

        if im.shape[-1] > 3:
            im = np.array(img.convert("RGB")).astype(np.float32)

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = self.cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=self.cv2.INTER_LINEAR,
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self,
        output: torch.Tensor,
        im_scales: List[float],
        im_infos: List[Dict[str, int]],
        feature_name: str = "fc6",
        conf_thresh: int = 0,
    ):
        """
        Post-process feature extraction from the detection model.

        :param output:
            output from the detection model
        :param im_scales:
            list of scales for the processed images
        :param im_infos:
            list of dicts containing width/height for images
        :param feature_name:
            which feature to extract for the image
        :param conf_thresh:
            threshold for bounding box scores (?)

        :return (feature_list, info_list):
            return list of processed image features, and list of information for each image
        """
        from maskrcnn_benchmark.layers import nms

        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros(scores.shape[0]).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater
                    # than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][:, start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "cls_prob": scores[keep_boxes][:, start_index:].cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, images: List["Image"]) -> List[torch.Tensor]:
        """
        Extract detectron features.

        :param images:
            a list of PIL Images

        :return features:
            return a list of features
        """
        from maskrcnn_benchmark.structures.image_list import to_image_list

        img_tensor, im_scales, im_infos = [], [], []

        for image in images:
            im, im_scale, im_info = self._image_transform(image)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        if self.use_cuda:
            current_img_list = current_img_list.to("cuda")

        with torch.no_grad():
            output = self.detection_model(current_img_list)

        features, _ = self._process_feature_extraction(output, im_scales, im_infos)

        return features
