# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.scripts.extract_image_feature import extract_feats
from .build_2014 import build as build_2014
from .build_2014 import buildImage as buildImage_2014
from .build_2017 import build as build_2017
from .build_2017 import buildImage as buildImage_2017
try:
    import torch
except Exception as e:
    raise ImportError('Need to install Pytorch: go to pytorch.org')
from torch.utils.data import Dataset
from parlai.core.dict import DictionaryAgent

import os
import json

# There is no real dialog in this task, so for the purposes of display_data, we
# include a generic question that applies to all images.
QUESTION = "Describe the above picture in a sentence."


def _path(opt, version):
    if version == '2014':
        build_2014(opt)
        buildImage_2014(opt)
    elif version == '2017':
        build_2017(opt)
        buildImage_2017(opt)
    else:
        raise Exception('Unknown version for COCO Captions: %s' % version)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        annotation_suffix = 'train{}'.format(version)
        img_suffix = os.path.join(
                  'train{}'.format(version),
                  'COCO_train{}_'.format(version) if version == '2014' else '')
    elif dt == 'valid':
        annotation_suffix = 'val{}'.format(version)
        img_suffix = os.path.join(
                  'val{}'.format(version),
                  'COCO_val{}_'.format(version) if version == '2014' else '')
    elif dt == 'test':
        annotation_suffix = 'None'
        img_suffix = os.path.join(
                  'test{}'.format(version),
                  'COCO_test{}_'.format(version) if version == '2014' else '')
    else:
        raise RuntimeError('Not valid datatype.')

    test_info_path = os.path.join(opt['datapath'],
                                  'COCO_{}_Caption'.format(version),
                                  'annotations',
                                  'image_info_test{}.json'.format(version))

    annotation_path = os.path.join(opt['datapath'],
                                   'COCO_{}_Caption'.format(version),
                                   'annotations',
                                   'captions_' + annotation_suffix + '.json')

    image_path = os.path.join(opt['datapath'], 'COCO-IMG-{}'.format(version),
                              img_suffix)

    return test_info_path, annotation_path, image_path


class DefaultDataset(Dataset):
    """A Pytorch Dataset utilizing streaming."""

    def __init__(self, opt, version='2014'):
        self.opt = opt
        self.use_hdf5 = opt.get('use_hdf5', False)
        self.datatype = self.opt.get('datatype')
        self.training = self.datatype.startswith('train')
        self.num_epochs = self.opt.get('num_epochs', 0)
        self.image_loader = ImageLoader(opt)
        test_info_path, annotation_path, self.image_path = _path(opt, version)
        self._setup_data(test_info_path, annotation_path, opt.get('unittest', False))
        if self.use_hdf5:
            try:
                import h5py
                self.h5py = h5py
            except ImportError:
                raise ImportError('Need to install h5py - `pip install h5py`')
            self._setup_image_data()
        self.dict_agent = DictionaryAgent(opt)

    def __getitem__(self, index):
        index %= self.num_episodes()
        image_id = None
        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][index]
            image_id = anno['image_id']
        else:
            image_id = self.test_info['images'][index]['id']
        ep = {
            'text': self.dict_agent.txt2vec(QUESTION),
            'image': self.get_image(image_id),
            'episode_done': True,
        }
        if self.opt.get('extract_image', False):
            ep['image_id'] = image_id
            return ep
        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][index]
            ep['labels'] = [anno['caption']]
            ep['valid'] = True
        else:
            ep['valid'] = True
        ep['use_hdf5'] = self.use_hdf5
        return (index, ep)

    def __len__(self):
        num_epochs = self.num_epochs if self.num_epochs > 0 else 100
        num_iters = num_epochs if self.training else 1
        return int(num_iters * self.num_episodes())

    def _load_lens(self):
        with open(self.length_datafile) as length:
            lengths = json.load(length)
            self.num_eps = lengths['num_eps']
            self.num_exs = lengths['num_exs']

    def _setup_data(self, test_info_path, annotation_path, unittest):
        if not self.datatype.startswith('test'):
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)
        else:
            with open(test_info_path) as data_file:
                self.test_info = json.load(data_file)

        if unittest:
            if not self.datatype.startswith('test'):
                self.annotation['annotations'] = self.annotation['annotations'][:10]
            else:
                self.test_info['images'] = self.test_info['images'][:10]

        self.image_paths = set()
        # Depending on whether we are using the train/val/test set, we need to
        # find the image IDs in annotations or test image info
        if not self.datatype.startswith('test'):
            for anno in self.annotation['annotations']:
                self.image_paths.add(self.image_path + '%012d.jpg' % (anno['image_id']))
        else:
            for info in self.test_info['images']:
                self.image_paths.add(self.image_path + '%012d.jpg' % (info['id']))


    def _setup_image_data(self):
        '''hdf5 image dataset'''
        extract_feats(self.opt)
        im = self.opt.get('image_mode')
        hdf5_path = os.path.join(self.image_path, 'mode_{}_noatt.hdf5'.format(im))
        hdf5_file = self.h5py.File(hdf5_path, 'r')
        self.image_dataset = hdf5_file['images']

        image_id_to_idx_path = os.path.join(self.image_path, 'mode_{}_id_to_idx.txt'.format(im))
        with open(image_id_to_idx_path, 'r') as f:
            self.image_id_to_idx = json.load(f)

    def get_image(self, image_id):
        if not self.use_hdf5:
            im_path = os.path.join(self.image_path, '%012d.jpg' % (image_id))
            return self.image_loader.load(im_path)
        else:
            img_idx = self.image_id_to_idx[str(image_id)]
            return torch.Tensor(self.image_dataset[img_idx])

    def num_examples(self):
        if not self.datatype.startswith('test'):
            return len(self.annotation['annotations'])
        else:
            return len(self.test_info['images'])

    def num_episodes(self):
        return self.num_examples()

    def num_images(self):
        if not hasattr(self, 'num_imgs'):
            return self.num_examples()
        return self.num_imgs


class V2014Dataset(DefaultDataset):
    def __init__(self, opt):
        super(V2014Dataset, self).__init__(opt, '2014')


class V2017Dataset(DefaultDataset):
    def __init__(self, opt):
        super(V2017Dataset, self).__init__(opt, '2017')


class DefaultTeacher(FixedDialogTeacher):
    """
    COCO default teacher that expects open-ended descriptions of images
    """
    def __init__(self, opt, shared=None, version='2017'):
        super().__init__(opt, shared)
        self.image_mode = opt.get('image_mode', 'none')

        if shared:
            # another instance was set up already, just reference its data
            if 'annotation' in shared:
                self.annotation = shared['annotation']
            self.image_loader = shared['image_loader']
            self.image_path = shared['image_path']
        else:
            # need to set up data from scratch
            test_info_path, annotation_path, self.image_path = _path(opt, version)
            self._setup_data(test_info_path, annotation_path)
            self.image_loader = ImageLoader(opt)

        self.reset()

    def reset(self):
        super().reset()  # call parent reset so other fields can be set up
        self.example = None  # set up caching fields
        self.imageEpochDone = False

    def num_examples(self):
        # We only have annotations for the train and val sets, so for the test
        # set we need to determine how many images we have.
        if not self.datatype.startswith('test'):
            return len(self.annotation['annotations'])
        else:
            return len(self.test_info['images'])

    def num_episodes(self):
        return self.num_examples()

    def submit_load_request(self, image_id):
        img_path = self.image_path + '%012d.jpg' % (image_id)
        self.data_loader.request_load(self.receive_data, self.image_loader.load, (img_path,))

    def get(self, episode_idx, entry_idx=0):
        action = {
            'text': "",
            'episode_done': True
        }

        if not self.datatype.startswith('test'):
            # test set annotations are not available for this dataset
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [anno['caption']]
            action['image_id'] = anno['image_id']
        else:
            action['image_id'] = self.test_info['images'][episode_idx]['id']

        return action

    def next_example(self):
        """Returns the next example from this dataset after starting to queue
        up the next example.
        """
        ready = None
        # pull up the currently queued example
        if self.example is not None:
            if self.image_mode != 'none' and 'image_id' in self.example:
                # move the image we loaded in the background into the example
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.imageEpochDone)
        # get the next base example: super().next_example() calls self.get()
        self.example, self.imageEpochDone = super().next_example()
        if self.image_mode != 'none' and 'image_id' in self.example:
            # load the next image in the background
            image_id = self.example['image_id']
            self.submit_load_request(image_id)
        # Try to return the previously cached example
        if ready is None:
            return self.next_example()
        else:
            return ready

    def share(self):
        shared = super().share()
        if hasattr(self, 'annotation'):
            shared['annotation'] = self.annotation
        shared['image_loader'] = self.image_loader
        shared['image_path'] = self.image_path
        return shared

    def _setup_data(self, test_info_path, annotation_path):
        if not self.datatype.startswith('test'):
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)
        else:
            print('loading: ' + test_info_path)
            with open(test_info_path) as data_file:
                self.test_info = json.load(data_file)


class V2014Teacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super(V2014Teacher, self).__init__(opt, shared, '2014')


class V2017Teacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super(V2017Teacher, self).__init__(opt, shared, '2017')
