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
import importlib
import h5py
import copy
import os
import json

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.utils import ProgressLogger

def get_dataset_class(opt):
    """ To use a custom Pytorch Dataset, specify it on the command line:
        ``--dataset parlai.tasks.vqa_v1.agents:VQADataset``

        Note that if the dataset is named ``DefaultDataset``, then you do
        not need to specify its name following the colon; e.g., it
        would just be:
        ``--dataset parlai.tasks.vqa_v1.agents``
    """
    dataset_name = opt.get('dataset')
    sp = dataset_name.strip().split(':')
    module_name = sp[0]
    if len(sp) > 1:
        dataset = sp[1]
    else:
        dataset = 'DefaultDataset'
    my_module = importlib.import_module(module_name)
    return getattr(my_module, dataset)


def main(opt):
    # Get command line arguments
    opt = copy.deepcopy(opt)
    opt['datatype'] = 'train:ordered'
    bsz = opt.get('batchsize', 1)
    opt['no_cuda'] = False
    opt['gpu'] = 0
    opt['num_epochs'] = 1
    opt['no_hdf5'] = True
    logger = ProgressLogger(should_humanize=False, throttle=0.1)
    print("\n---Beginning image extraction---\n")

    # create repeat label agent and assign it to the specified task
    if opt.get('dataset') is None:
        agent = RepeatLabelAgent(opt)
        world = create_task(opt, agent)

        exs_seen = 0
        total_exs = world.num_examples()
        while not world.epoch_done():
            world.parley()
            exs_seen += bsz
            logger.log(exs_seen, total_exs)
    else:
        '''One can specify a Pytorch Dataset for custom image loading'''
        nw = opt.get('numworkers', 1)
        im = opt.get('image_mode', 'raw')
        opt['batchsize'] = 1
        opt['extract_image'] = True
        bsz = 1
        try:
            import torch
        except Exception as e:
            raise ModuleNotFoundError('Need to install Pytorch: go to pytorch.org')
        from torch.utils.data import Dataset, DataLoader, sampler

        dataset = get_dataset_class(opt)(opt)

        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=False,
            num_workers=nw,
            collate_fn=lambda batch: batch[0]
        )

        dataset_shape = None
        image_id_to_index = {}
        num_images = dataset.num_images()
        attention = opt.get('attention', False)
        if attention:
            hdf5_path = dataset.image_path + 'mode_{}.hdf5'.format(im)
        else:
            hdf5_path = dataset.image_path + 'mode_{}_noatt.hdf5'.format(im)
        image_id_to_idx_path = dataset.image_path + 'mode_{}_id_to_idx.txt'.format(im)
        if os.path.isfile(hdf5_path):
            print('[ Images already extracted at: {} ]'.format(hdf5_path))
            return

        hdf5_file = h5py.File(hdf5_path, 'w')
        idx = 0
        for ex in iter(dataloader):
            if ex['image_id'] in image_id_to_index:
                continue
            else:
                image_id_to_index[ex['image_id']] = idx

            img = ex['image']
            if not attention:
                nb_regions = img.size(2) * img.size(3)
                img = img.sum(3).sum(2).div(nb_regions).view(-1, 2048)

            if dataset_shape is None:
                if attention:
                    dataset_shape = (num_images, img.size(1), img.size(2), img.size(3))
                else:
                    dataset_shape = (num_images, img.size(1))
                hdf5_dataset = hdf5_file.create_dataset(
                    'images',
                    dataset_shape,
                    dtype='f')

            hdf5_dataset[idx] = img
            logger.log(idx, num_images)
            idx+=1

        hdf5_file.close()
        if not os.path.exists(image_id_to_idx_path):
            with open(image_id_to_idx_path, 'w') as f:
                json.dump(image_id_to_index, f)

    print("\n---Finished extracting images---\n")


if __name__ == '__main__':
    parser = ParlaiParser(True, False)
    arg_group = parser.add_argument_group('Image Extraction')
    arg_group.add_argument('--dataset', type=str, default=None,
                           help='Pytorch Dataset')
    opt = parser.parse_args()
    main(opt)
