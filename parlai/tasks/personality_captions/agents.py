#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Images and Comments from Personality-Captions dataset.

200k images + comments, with different personalities.

An example is given as follows:
    obs = {'text': <personality>,
           'image': <image features if specified else image>,
           'label': <comment>,
          }
"""
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.utils.io import PathManager
from .build import build

import json
import os


def _path(opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt in ['train', 'valid', 'test']:
        data_path = os.path.join(
            opt['datapath'], 'personality_captions/{}.json'.format(dt)
        )

    personalities_data_path = os.path.join(
        opt['datapath'], 'personality_captions/personalities.json'
    )
    image_path = ''
    if opt.get('yfcc_path'):
        image_path = opt['yfcc_path']
    else:
        image_path = os.path.join(opt['datapath'], 'yfcc_images')

    return data_path, personalities_data_path, image_path


class PersonalityCaptionsTeacher(FixedDialogTeacher):
    """
    Provide the personality in the `text` field, and the captions in the `labels` field.

    To specify your own path to the YFCC100m images, please use the `--yfcc-path`
    command line argument.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.image_mode = opt.get('image_mode', 'no_image_model')
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
        """
        Add command line args.
        """
        agent = argparser.add_argument_group('Personality-Captions arguments')
        agent.add_argument(
            '--include-personality',
            type='bool',
            default=True,
            help='Whether to provide personality to agent',
        )
        agent.add_argument(
            '--include-image',
            type='bool',
            default=True,
            help='Whether to provide image to agent',
        )
        agent.add_argument(
            '--num-test-labels',
            type=int,
            default=1,
            choices=[1, 5],
            help='Provide model with either 1 or 5 possible '
            'labels for each test example. The number of label '
            'candidates for each case is 100 and 500 '
            'respectively.',
        )
        agent.add_argument(
            '--yfcc-path',
            type=str,
            default=None,
            help='Path to yfcc images (if not downloaded '
            'via the provided download script)',
        )

    def _setup_data(self, data_path, personalities_data_path):
        print('loading: ' + data_path)
        with PathManager.open(data_path) as f:
            self.data = json.load(f)
        with PathManager.open(personalities_data_path) as f:
            self.personalities = json.load(f)

    def reset(self):
        """
        Reset teacher.
        """
        super().reset()
        self.example = None

    def num_episodes(self):
        """
        Return number of episodes.
        """
        return self.num_examples()

    def num_examples(self):
        """
        Return number of examples.
        """
        return len(self.data)

    def submit_load_request(self, image_id):
        """
        Submit a load request to the image loader.

        :param image_id:
            id of image to load
        """
        img_path = os.path.join(self.image_path, '{}.jpg'.format(image_id))
        self.data_loader.request_load(
            self.receive_data, self.image_loader.load, (img_path,)
        )

    def get(self, episode_idx, entry_idx=0):
        """
        Get an example.

        :param episode_idx:
            index of episode in self.data
        :param entry_idx:
            optional, which entry in the episode to get

        :return:
            an example
        """
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
        """
        Return the next example from this dataset.

        Queues next example.
        """
        ready = None
        load_image = self.image_mode != 'no_image_model' and self.include_image
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
        """
        Share appropriate attributes.
        """
        shared = super().share()
        shared['data'] = self.data
        shared['image_loader'] = self.image_loader
        return shared


class PersonalityCaptionsTestTeacher(PersonalityCaptionsTeacher):
    """
    Test PersonalityCaptions teacher for ensuring pretrained model does not break.
    """

    def _setup_data(self, data_path, personalities_data_path):
        super()._setup_data(data_path, personalities_data_path)
        from parlai.zoo.personality_captions.transresnet import download

        download(self.opt['datapath'])
        image_features_path = os.path.join(
            self.opt['datapath'],
            'models/personality_captions/transresnet/test_image_feats',
        )
        import torch

        with PathManager.open(image_features_path, 'rb') as f:
            self.image_features = torch.load(f)

    def reset(self):
        """
        Reset teacher.
        """
        super().reset()
        self.example = None

    def num_episodes(self):
        """
        Return number of episodes.
        """
        return len(self.image_features)

    def num_examples(self):
        """
        Return number of examples.
        """
        return len(self.image_features)

    def get(self, episode_idx, entry_idx=0):
        """
        Get an example.

        :param episode_idx:
            index of episode in self.data
        :param entry_idx:
            optional, which entry in the episode to get

        :return:
            an example
        """
        data = self.data[episode_idx]

        action = {
            'text': data['personality'] if self.include_personality else '',
            'image': self.image_features[data['image_hash']],
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


class DefaultTeacher(PersonalityCaptionsTeacher):
    """
    Default teacher.
    """

    pass
