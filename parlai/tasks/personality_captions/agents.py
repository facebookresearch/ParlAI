# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
    Images and Comments from Personality-Captions dataset

    200k images + comments, with different personalities.

    An example is given as follows:
        obs = {'text': <personality>,
               'image': <image features if specified else image>,
               'label': <comment>,
              }

'''
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from torch.utils.data import Dataset
from .build import build

import json
import os


def _path(opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt in ['train', 'valid', 'test']:
        data_path = os.path.join(opt['datapath'],
                                 'personality_captions/{}.json'.format(dt))

    personalities_data_path = os.path.join(opt['datapath'],
                                           'personality_captions/personalities.json')
    image_path = ''
    if opt.get('yfcc_path'):
        image_path = opt['yfcc_path']
    else:
        image_path = os.path.join(opt['datapath'], 'yfcc_images')

    return data_path, personalities_data_path, image_path


class DefaultDataset(Dataset):
    """A Pytorch Dataset"""
    def __init__(self, opt):
        self.opt = opt
        opt['image_load_task'] = 'personality_captions'
        self.image_mode = opt.get('image_mode', 'none')
        self.datatype = self.opt.get('datatype')
        self.training = self.datatype.startswith('train')
        self.include_image = opt.get('include_image')
        self.include_personality = opt.get('include_personality')
        self.num_test_labels = opt.get('num_test_labels', 1)
        data_path, personalities_data_path, self.image_path = _path(opt)
        self.image_loader = ImageLoader(opt)
        self._setup_data(data_path, personalities_data_path)

    @staticmethod
    def add_cmdline_args(argparser):
        PersonalityCaptionsTeacher.add_cmdline_args(argparser)

    def _setup_data(self, data_path, personalities_data_path):
        print('loading: ' + data_path)
        with open(data_path) as f:
            self.data = json.load(f)
        with open(personalities_data_path) as f:
            self.personalities = json.load(f)

    def __getitem__(self, index):
        data = self.data[index]
        image = self.get_image(data['image_hash'])

        ep = {
            'text': data['personality'] if self.include_personality else '',
            'episode_done': True,
            'image': image if self.include_image else None,
        }

        if self.opt.get('extract_image', False):
            ep['image_id'] = data['image_hash']
            return ep
        ep['labels'] = [data['comment']]
        if self.num_test_labels == 5 and 'test' in self.datatype:
            ep['labels'] += data['additional_comments']
        if not self.training:
            if self.num_test_labels == 5 and 'test' in self.datatype:
                ep['label_candidates'] = data['500_candidates']
            else:
                ep['label_candidates'] = data['candidates']

        return (index, ep)

    def __len__(self):
        return self.num_episodes()

    def get_image(self, image_id):
        im_path = os.path.join(self.image_path, '{}.jpg'.format(image_id))
        return self.image_loader.load(im_path)

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return self.num_episodes()

    def num_images(self):
        if not hasattr(self, 'num_imgs'):
            self.num_imgs = len({d['image_num'] for d in self.data})
        return self.num_imgs


class PersonalityCaptionsTeacher(FixedDialogTeacher):
    """
        Provides the personality in the `text` field, and
        the captions in the `labels` field

        To specify your own path to the YFCC100m images, please use the
        `--yfcc-path` command line argument.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.image_mode = opt.get('image_mode', 'none')
        self.data_path, personalities_data_path, self.image_path = _path(opt)
        self.datatype = opt.get('datatype').split(':')[0]
        self.include_personality = opt.get('include_personality')
        self.include_image = opt.get('include_image')
        self.num_test_labels = opt.get('num_test_labels')
        if shared and 'data' in shared:
            self.data = shared['data']
            self.image_loader = shared['image_loader']
        else:
            self.image_loader = ImageLoader(opt)
            self._setup_data(self.data_path, personalities_data_path)
        self.reset()

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Personality-Captions arguments')
        agent.add_argument('--include-personality', type='bool',
                           default=True,
                           help='Whether to provide personality to agent')
        agent.add_argument('--include-image', type='bool',
                           default=True,
                           help='Whether to provide image to agent')
        agent.add_argument('--num-test-labels', type=int, default=1,
                           choices=[1, 5],
                           help='Provide model with either 1 or 5 possible '
                           'labels for each test example. The number of label '
                           'candidates for each case is 100 and 500 '
                           'respectively.')
        agent.add_argument('--yfcc-path', type=str, default=None,
                           help='Path to yfcc images (if not downloaded '
                                'via the provided download script)')

    def _setup_data(self, data_path, personalities_data_path):
        print('loading: ' + data_path)
        with open(data_path) as f:
            self.data = json.load(f)
        with open(personalities_data_path) as f:
            self.personalities = json.load(f)

    def reset(self):
        super().reset()
        self.example = None

    def num_episodes(self):
        return self.num_examples()

    def num_examples(self):
        return len(self.data)

    def submit_load_request(self, image_id):
        img_path = os.path.join(self.image_path, '{}.jpg'.format(image_id))
        self.data_loader.request_load(self.receive_data,
                                      self.image_loader.load,
                                      (img_path,))

    def get(self, episode_idx, entry_idx=0):
        data = self.data[episode_idx]

        action = {
            'text': data['personality'] if self.include_personality else '',
            'image_id': data['image_hash'],
            'episode_done': True,
            'labels': [data['comment']],
        }
        if self.num_test_labels == 5 and 'test' in self.datatype:
            action['labels'] += data['additional_comments']

        if 'candidates' in data:
            if self.num_test_labels == 5 and 'test' in self.datatype:
                action['label_candidates'] = data['500_candidates']
            else:
                action['label_candidates'] = data['candidates']

        return action

    def next_example(self):
        """Returns the next example from this dataset after starting to queue
        up the next example.
        """
        ready = None
        load_image = self.image_mode != 'none' and self.include_image
        # pull up the currently queued example
        if self.example is not None:
            # if self.image_mode != 'none' and 'image_id' in self.example:
            if load_image and 'image_id' in self.example:
                # move the image we loaded in the background into the example
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.imageEpochDone)
        # get the next base example: super().next_example() calls self.get()
        self.example, self.imageEpochDone = super().next_example()
        # if self.image_mode != 'none' and 'image_id' in self.example:
        if load_image and 'image_id' in self.example:
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
        shared['data'] = self.data
        shared['image_loader'] = self.image_loader
        return shared


class DefaultTeacher(PersonalityCaptionsTeacher):
    pass
