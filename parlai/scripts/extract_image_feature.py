#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic example which iterates through the tasks specified and load/extract the image
features.

For more options, check ``parlai.core.image_featurizers``

Examples
--------

To extract the image feature of COCO images:

.. code-block:: shell

  python examples/extract_image_feature.py -t vqa_v1 -im resnet152
"""
import importlib
import h5py
import copy
import os
import json
import datetime
import tqdm

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task


# TODO: this may not be adequately updated after deleting pytorch data teacher


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, False, 'Load/extract image features')
    arg_group = parser.add_argument_group('Image Extraction')
    arg_group.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Pytorch Dataset; if specified, will save \
                           the images in one hdf5 file according to how \
                           they are returned by the specified dataset',
    )
    arg_group.add_argument(
        '-at',
        '--attention',
        action='store_true',
        help='Whether to extract image features with attention \
                           (Note - this is specifically for the mlb_vqa model)',
    )
    arg_group.add_argument(
        '--use-hdf5-extraction',
        type='bool',
        default=False,
        help='Whether to extract images into an hdf5 dataset',
    )

    return parser


def get_dataset_class(opt):
    """
    To use a custom Pytorch Dataset, specify it on the command line: ``--dataset
    parlai.tasks.vqa_v1.agents:VQADataset``

    Note that if the dataset is named ``DefaultDataset``, then you do not need to
    specify its name following the colon; e.g., it would just be: ``--dataset
    parlai.tasks.vqa_v1.agents``
    """
    dataset_name = opt.get('pytorch_teacher_dataset')
    sp = dataset_name.strip().split(':')
    module_name = sp[0]
    if len(sp) > 1:
        dataset = sp[1]
    else:
        dataset = 'DefaultDataset'
    my_module = importlib.import_module(module_name)
    return getattr(my_module, dataset)


def extract_feats(opt):
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: extract_feats should be passed opt not Parser ]')
        opt = opt.parse_args()
    # Get command line arguments
    opt = copy.deepcopy(opt)
    dt = opt['datatype'].split(':')[0] + ':ordered'
    opt['datatype'] = dt
    bsz = opt.get('batchsize', 1)
    opt['no_cuda'] = False
    opt['gpu'] = 0
    opt['num_epochs'] = 1
    opt['use_hdf5'] = False
    opt['num_load_threads'] = 20
    print("[ Loading Images ]")
    # create repeat label agent and assign it to the specified task
    if opt.get('pytorch_teacher_dataset') is None:
        agent = RepeatLabelAgent(opt)
        world = create_task(opt, agent)

        total_exs = world.num_examples()
        pbar = tqdm.tqdm(unit='ex', total=total_exs)
        while not world.epoch_done():
            world.parley()
            pbar.update()
        pbar.close()
    elif opt.get('use_hdf5_extraction', False):
        # TODO Deprecate
        """
        One can specify a Pytorch Dataset for custom image loading.
        """
        nw = opt.get('numworkers', 1)
        im = opt.get('image_mode', 'raw')
        opt['batchsize'] = 1
        opt['extract_image'] = True
        bsz = 1
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError('Need to install Pytorch: go to pytorch.org')

        dataset = get_dataset_class(opt)(opt)
        pre_image_path, _ = os.path.split(dataset.image_path)
        image_path = os.path.join(pre_image_path, opt.get('image_mode'))
        images_built_file = image_path + '.built'

        if not os.path.exists(image_path) or not os.path.isfile(images_built_file):
            """
            Image features have not been computed yet.
            """
            opt['num_load_threads'] = 20
            agent = RepeatLabelAgent(opt)
            if opt['task'] == 'pytorch_teacher':
                if opt.get('pytorch_teacher_task'):
                    opt['task'] = opt['pytorch_teacher_task']
                else:
                    opt['task'] = opt['pytorch_teacher_dataset']
            world = create_task(opt, agent)
            exs_seen = 0
            total_exs = world.num_examples()
            pbar = tqdm.tqdm(unit='ex', total=total_exs)
            print('[ Computing and Saving Image Features ]')
            while exs_seen < total_exs:
                world.parley()
                exs_seen += bsz
                pbar.update(bsz)
            pbar.close()
            print('[ Feature Computation Done ]')
            with open(images_built_file, 'w') as write:
                write.write(str(datetime.datetime.today()))

        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=False,
            num_workers=nw,
            collate_fn=lambda batch: batch[0],
        )

        dataset_shape = None
        image_id_to_index = {}
        num_images = dataset.num_images()
        attention = opt.get('attention', False)
        if attention:
            hdf5_path = '{}mode_{}.hdf5'.format(dataset.image_path, im)
        else:
            hdf5_path = '{}mode_{}_noatt.hdf5'.format(dataset.image_path, im)
        image_id_to_idx_path = '{}mode_{}_id_to_idx.txt'.format(dataset.image_path, im)
        hdf5_built_file = hdf5_path + '.built'
        if os.path.isfile(hdf5_path) and os.path.isfile(hdf5_built_file):
            print('[ Images already extracted at: {} ]'.format(hdf5_path))
            return

        print("[ Beginning image extraction for {} images ]".format(dt.split(':')[0]))
        hdf5_file = h5py.File(hdf5_path, 'w')
        idx = 0
        iterator = tqdm.tqdm(
            dataloader, unit='batch', unit_scale=True, total=total_exs // bsz
        )
        for ex in iterator:
            if ex['image_id'] in image_id_to_index:
                continue
            else:
                image_id_to_index[ex['image_id']] = idx

            img = ex['image']
            if isinstance(img, torch.autograd.Variable):
                img = img.cpu().data

            if not attention:
                nb_regions = img.size(2) * img.size(3)
                img = img.sum(3).sum(2).div(nb_regions).view(-1, 2048)

            if dataset_shape is None:
                if attention:
                    dataset_shape = (num_images, img.size(1), img.size(2), img.size(3))
                else:
                    dataset_shape = (num_images, img.size(1))
                hdf5_dataset = hdf5_file.create_dataset(
                    'images', dataset_shape, dtype='f'
                )

            hdf5_dataset[idx] = img
            idx += 1

        hdf5_file.close()
        if not os.path.exists(image_id_to_idx_path):
            with open(image_id_to_idx_path, 'w') as f:
                json.dump(image_id_to_index, f)
        with open(hdf5_built_file, 'w') as write:
            write.write(str(datetime.datetime.today()))

    print("[ Finished extracting images ]")


if __name__ == '__main__':
    extract_feats(setup_args().parse_args())
