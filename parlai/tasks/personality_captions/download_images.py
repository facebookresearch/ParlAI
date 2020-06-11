#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
from parlai.core.build_data import download_multiprocess
from parlai.core.params import ParlaiParser
import parlai.core.build_data as build_data


def download_images(opt, task='personality_captions'):
    dpath = os.path.join(opt['datapath'], task)
    image_path = os.path.join(opt['datapath'], 'yfcc_images')
    version = '1.0'
    response = input(
        'Please confirm that you have obtained permission '
        'to work with the YFCC100m dataset, as outlined by the steps '
        'listed at '
        'https://multimediacommons.wordpress.com/yfcc100m-core-dataset/ [Y/y]: '
    )
    if response.lower() != 'y':
        raise RuntimeError(
            'In order to use the images from this dataset, '
            'you must obtain permission from the webpage above.'
        )
    response = input(
        'NOTE: This script will download each image individually from the '
        's3 server on which the images are hosted. This will take a *very '
        'long* time. Are you sure you would like to continue? [Y/y]: '
    )
    if response.lower() != 'y':
        raise RuntimeError(
            'If you have access to the images, please specify '
            'the path to the folder via the `--yfcc-path` '
            'command line argument.'
        )
    image_prefix = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images'
    hashes = []
    dts = ['train', 'val', 'test']
    if task == 'image_chat':
        dts[1] = 'valid'
    for dt in dts:
        with open(os.path.join(dpath, '{}.json'.format(dt))) as f:
            data = json.load(f)
            hashes += [d['image_hash'] for d in data]
    os.makedirs(image_path, exist_ok=True)

    print('[downloading images to {}]'.format(image_path))
    image_urls = [
        f"{image_prefix}/{p_hash[:3]}/{p_hash[3:6]}/{p_hash}.jpg" for p_hash in hashes
    ]
    download_multiprocess(
        image_urls, image_path, dest_filenames=[f"{h}.jpg" for h in hashes]
    )
    build_data.mark_done(image_path, version)


if __name__ == '__main__':
    parser = ParlaiParser()
    download_images(parser.parse_args())
