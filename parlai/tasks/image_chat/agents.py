# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
    Images and Dialogues from Image-Chat dataset

    202k images, 401k utterances, over 215 different personalities.

    An example is given as follows:
        obs = {'text': <personality>,
               'image': <image features if specified else image>,
               'label': <comment/response>,
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
                                 'image_chat/{}.json'.format(dt))

    personalities_data_path = os.path.join(opt['datapath'],
                                           'image_chat/personalities.json')
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
        opt['image_load_task'] = 'image_chat'
        self.image_mode = opt.get('image_mode', 'none')
        self.datatype = self.opt.get('datatype')
        self.training = self.datatype.startswith('train')
        self.include_image = opt.get('include_image')
        self.include_personality = opt.get('include_personality')
        self.num_cands = opt.get('num_cands')
        data_path, personalities_data_path, self.image_path = _path(opt)
        self.image_loader = ImageLoader(opt)
        self._setup_data(data_path, personalities_data_path)

    @staticmethod
    def add_cmdline_args(argparser):
        ImageChatTeacher.add_cmdline_args(argparser)

    def _setup_data(self, data_path, personalities_data_path):
        print('loading: ' + data_path)
        with open(data_path) as f:
            self.data = json.load(f)
        with open(personalities_data_path) as f:
            self.personalities = json.load(f)

    def __getitem__(self, index):
        data = self.data[index]
        dialog = data['dialog']
        personality = dialog[-1][0]
        text = ''
        if len(dialog) > 1:
            text = '\n'.join((dialog[0][1], dialog[1][1]))
        if self.include_personality:
            text += personality
        label = dialog[-1][1]
        image = self.get_image(data['image_hash'])

        ep = {
            'text': text,
            'episode_done': True,
            'image': image if self.include_image else None,
        }

        if self.opt.get('extract_image', False):
            ep['image_id'] = data['image_hash']
            return ep
        if not self.opt['datatype'].startswith('test'):
            ep['labels'] = [label]
        if not self.training:
            ep['label_candidates'] = data['candidates'][-1][self.num_cands]

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


class ImageChatTeacher(FixedDialogTeacher):
    """
        Provides the personality in the `text` field, and
        response in the `labels` field

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
        self.num_cands = opt.get('num_cands')
        if shared and 'data' in shared:
            self.data = shared['data']
            self.personalities = shared['personalities']
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
        agent.add_argument('--yfcc-path', type=str, default=None,
                           help='Path to yfcc images (if not downloaded '
                                'via the provided download script)')
        agent.add_argument('--num-cands', type=str, default='100',
                           choices=['100', '1000'],
                           help='how many candidates to provide agent')

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
        return len(self.data)

    def num_examples(self):
        return sum(len(d['dialog']) for d in self.data)

    def submit_load_request(self, image_id):
        img_path = os.path.join(self.image_path, '{}.jpg'.format(image_id))
        self.data_loader.request_load(self.receive_data,
                                      self.image_loader.load,
                                      (img_path,))

    def get(self, episode_idx, entry_idx=0):
        data = self.data[episode_idx]
        personality, text = data['dialog'][entry_idx]
        episode_done = entry_idx == len(data['dialog']) - 1

        action = {
            'text': personality if self.include_personality else '',
            'image_id': data['image_hash'],
            'episode_done': episode_done,
            'labels': [text],
        }

        if 'candidates' in data:
            action['label_candidates'] = data['candidates'][entry_idx][self.num_cands]

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
        shared['personalities'] = self.personalities
        return shared


class DefaultTeacher(ImageChatTeacher):
    pass
