#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Image-based utils for chat services.
"""
import base64
import io
from typing import Dict, Union

import PIL
import torch
from parlai.core.image_featurizers import ImageLoader
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.typing import TShared


class ImageLicense(object):
    """
    Representation of image license information.
    """

    def __init__(
        self, username: str, image_url: str, image_license: str, image_license_url: str
    ):
        self._username = username
        self._image_url = image_url
        self._image_license = image_license
        self._image_license_url = image_license_url

    def get_username(self):
        return self._username

    def get_image_url(self):
        return self._image_url

    def get_image_license(self):
        return self._image_license

    def get_image_license_url(self):
        return self._image_license_url

    def get_attribution_message(self):
        return (
            f"This image was originally under the license *{self._image_license}"
            f"({self._image_license_url})* by user *{self._username}*. "
            f"The original image link is *{self._image_url}*."
        )


class ImageInformation(object):
    """
    Representation of image information.
    """

    def __init__(
        self, image_id: str, image_location_id: str, image: Union["PIL.Image", str]
    ):
        """
        When image is str, it is a serialized; need to deserialize.
        """
        self._image_id = image_id
        self._image_location_id = image_location_id
        self._image = image
        if isinstance(self._image, str):
            self._image = PIL.Image.open(io.BytesIO(base64.b64decode(self._image)))

    def get_image_id(self) -> str:
        return self._image_id

    def get_image_location_id(self) -> str:
        return self._image_location_id

    def get_image(self) -> "PIL.Image":
        return self._image

    def offload_state(self) -> Dict[str, str]:
        """
        Return serialized state.

        :return state_dict:
            serialized state that can be used in json.dumps
        """
        byte_arr = io.BytesIO()
        image = self.get_image()
        image.save(byte_arr, format="JPEG")
        serialized = base64.encodebytes(byte_arr.getvalue()).decode("utf-8")
        return {
            "image_id": self.get_image_id(),
            "image_location_id": self.get_image_location_id(),
            "image": serialized,
        }


class ImageFeaturesGenerator(object):
    """
    Features generator for images.

    Uses ParlAI Image Loader.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        self.opt = opt
        self.image_model = opt.get("image_mode")
        if shared:
            self.image_loader = shared["image_loader"]
        else:
            opt.setdefault("image_mode", self.image_model)
            new_opt = ParlaiParser(True, False).parse_args([])
            for k, v in new_opt.items():
                if k not in opt:
                    opt[k] = v

            self.image_loader = ImageLoader(opt)

    def get_image_features(self, image_id: str, image: "PIL.Image") -> torch.Tensor:
        """
        Get image features for given image id and Image.

        :param image_id:
            id for image
        :param image:
            PIL Image object

        :return image_features:
            Image Features Tensor
        """
        image = image.convert("RGB")
        return self.image_loader.extract(image)


class ObjectionableContentError(Exception):
    """
    Error if an image is objectionable.
    """

    pass
