# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.scripts.extract_image_feature import extract_feats
from .build import build
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


def _path(opt):
    build(opt)

    caption_path = os.path.join(opt['datapath'], 'Flickr30k',
                                'results_20130124.token')
    image_path = os.path.join(opt['datapath'], 'Flickr30k', 'flickr30k_images')

    return caption_path, image_path


class FlickrDataset(Dataset):
    """A Pytorch Dataset utilizing streaming"""
    def __init__(self, opt, shared=None):
        self.opt = opt
        self.use_hdf5 = opt.get('use_hdf5', False)
        self.datatype = self.opt.get('datatype')
        self.training = self.datatype.startswith('train')
        self.num_epochs = self.opt.get('num_epochs', 0)
        self.image_loader = ImageLoader(opt)
        caption_path, self.image_path = _path(opt)
        self._setup_data(caption_path, opt.get('unittest', False))
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
        cap = self.caption[index]
        ep = {
            'text': self.dict_agent.txt2vec(QUESTION),
            'image': self.get_image(cap['image_id']),
            'episode_done': True,
        }
        if self.opt.get('extract_image', False):
            ep['image_id'] = cap['image_id']
            return ep

        ep['labels'] = cap['captions']
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

    def _setup_data(self, caption_path, unittest):
        with open(caption_path) as data_file:
            self.caption = []
            prev_img_id = None
            for line in data_file:
                img_id = line.split('#')[0][:-4]
                caption = line.split('\t')[1]
                if img_id != prev_img_id:
                    prev_img_id = img_id
                    to_add = {}
                    to_add['image_id'] = int(img_id)
                    to_add['captions'] = [caption]
                    self.caption.append(to_add)
                else:
                    self.caption[-1]['captions'].append(caption)
        if unittest:
            self.caption = self.caption[:10]
        self.image_paths = set()
        for cap in self.caption:
            self.image_paths.add(os.path.join(self.image_path,
                                              '%d.jpg' % (cap['image_id'])))

    def _setup_image_data(self):
        '''hdf5 image dataset'''
        extract_feats(self.opt)
        im = self.opt.get('image_mode')
        hdf5_path = self.image_path + 'mode_{}_noatt.hdf5'.format(im)
        hdf5_file = self.h5py.File(hdf5_path, 'r')
        self.image_dataset = hdf5_file['images']

        image_id_to_idx_path = self.image_path + 'mode_{}_id_to_idx.txt'.format(im)
        with open(image_id_to_idx_path, 'r') as f:
            self.image_id_to_idx = json.load(f)

    def get_image(self, image_id):
        if not self.use_hdf5:
            im_path = os.path.join(self.image_path, '%d.jpg' % (image_id))
            return self.image_loader.load(im_path)
        else:
            img_idx = self.image_id_to_idx[str(image_id)]
            return torch.Tensor(self.image_dataset[img_idx])

    def num_episodes(self):
        return len(self.caption)

    def num_examples(self):
        return self.num_episodes()

    def num_images(self):
        return self.num_episodes()


class DefaultDataset(FlickrDataset):
    pass


class DefaultTeacher(FixedDialogTeacher):
    """
    Flickr default teacher that expects open-ended descriptions of images
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.image_mode = opt.get('image_mode', 'none')

        if shared:
            # another instance was set up already, just reference its data
            self.caption = shared['caption']
            self.image_loader = shared['image_loader']
            self.image_path = shared['image_path']
        else:
            # need to set up data from scratch
            caption_path, self.image_path = _path(opt)
            self._setup_data(caption_path)
            self.image_loader = ImageLoader(opt)

        self.reset()

    def reset(self):
        super().reset()  # call parent reset so other fields can be set up
        self.example = None  # set up caching fields
        self.imageEpochDone = False

    def num_examples(self):
        return len(self.caption)

    def num_episodes(self):
        return self.num_examples()

    def submit_load_request(self, image_id):
        img_path = os.path.join(self.image_path, '%d.jpg' % (image_id))
        self.data_loader.request_load(self.receive_data,
                                      self.image_loader.load,
                                      (img_path,))

    def get(self, episode_idx, entry_idx=0):
        cap = self.caption[episode_idx]

        action = {
            'text': "",
            'image_id': cap['image_id'],
            'episode_done': True,
            'labels': cap['captions']
        }

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
        shared['caption'] = self.caption
        shared['image_loader'] = self.image_loader
        shared['image_path'] = self.image_path
        return shared

    def _setup_data(self, caption_path):
        print('loading: ' + caption_path)
        with open(caption_path) as data_file:
            self.caption = []
            prev_img_id = None
            for line in data_file:
                img_id = line.split('#')[0][:-4]
                caption = line.split('\t')[1]
                if img_id != prev_img_id:
                    prev_img_id = img_id
                    to_add = {}
                    to_add['image_id'] = int(img_id)
                    to_add['captions'] = [caption]
                    self.caption.append(to_add)
                else:
                    self.caption[-1]['captions'].append(caption)
