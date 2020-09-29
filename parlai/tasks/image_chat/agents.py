#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Images and Dialogues from Image-Chat dataset.

202k images, 401k utterances, over 215 different personalities.
"""
import json
import os
import random
from typing import Tuple, Dict, List

from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.utils.typing import TShared
from parlai.utils.io import PathManager
from .build import build


def _path(opt: Opt) -> Tuple[str, str, str]:
    """
    Return appropriate datapaths.

    :param opt:
        options

    :return (data path, personalities path, image_path):
        path to data, personalities, and images
    """
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt in ['train', 'valid', 'test']:
        data_path = os.path.join(opt['datapath'], 'image_chat/{}.json'.format(dt))

    personalities_data_path = os.path.join(
        opt['datapath'], 'image_chat/personalities.json'
    )
    image_path = ''
    if opt.get('yfcc_path'):
        image_path = opt['yfcc_path']
    else:
        image_path = os.path.join(opt['datapath'], 'yfcc_images')

    return data_path, personalities_data_path, image_path


class ImageChatTeacher(FixedDialogTeacher):
    """
    Provides the personality in the `text` field, and response in the `labels` field.

    To specify your own path to the YFCC100m images, please use the `--yfcc-path`
    command line argument.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        super().__init__(opt, shared)
        self.opt = opt
        self.image_mode = opt.get('image_mode', 'no_image_model')
        self.data_path, personalities_data_path, self.image_path = _path(opt)
        self.datatype = opt['datatype'].split(':')[0]
        self.include_personality = opt.get('include_personality')
        self.include_image = opt.get('include_image') and opt.get('load_images')
        self.num_cands = opt.get('num_cands')
        if shared and 'data' in shared:
            self.data = shared['data']
            self.personalities = shared['personalities']
            self.image_loader = shared['image_loader']
        else:
            self.image_loader = ImageLoader(opt)
            self._setup_data(self.data_path, personalities_data_path)
        self.num_exs = sum(len(d['dialog']) for d in self.data)
        self.reset()

    @staticmethod
    def add_cmdline_args(argparser):
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
            '--yfcc-path',
            type=str,
            default=None,
            help='Path to yfcc images (if not downloaded '
            'via the provided download script)',
        )
        agent.add_argument(
            '--load-images',
            type='bool',
            default=True,
            help='Specify whether to load images',
        )
        agent.add_argument(
            '--num-cands',
            type=str,
            default='100',
            choices=['100', '1000'],
            help='how many candidates to provide agent',
        )

    def _setup_data(self, data_path: str, personalities_data_path: str):
        """
        Load the data.
        """
        print('loading: ' + data_path)
        with PathManager.open(data_path) as f:
            self.data = json.load(f)
        with PathManager.open(personalities_data_path) as f:
            self.personalities = json.load(f)

    def reset(self):
        """
        Override to Reset self.example.
        """
        super().reset()
        self.example = None

    def num_episodes(self) -> int:
        return len(self.data)

    def num_examples(self) -> int:
        return self.num_exs

    def submit_load_request(self, image_id: str):  # type: ignore
        img_path = os.path.join(self.image_path, '{}.jpg'.format(image_id))
        self.data_loader.request_load(
            self.receive_data, self.image_loader.load, (img_path,)
        )

    def get(self, episode_idx: int, entry_idx: int = 0):
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

    def next_example(self) -> Tuple[Message, bool]:
        """
        Returns the next example from this dataset after starting to queue up the next
        example.

        :return (example, epoch done):
            returns the next example as well as whether the epoch is done.
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
            ready = (self.example, self.imageEpochDone)  # type: ignore
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

    def share(self) -> TShared:
        shared = super().share()
        shared['data'] = self.data
        shared['image_loader'] = self.image_loader
        shared['personalities'] = self.personalities
        return shared


class GenerationTeacher(ImageChatTeacher):
    """
    GenerationTeacher - dialogues are split into two episodes, one from
    each individual person's point of view.

    Used in the #dodecaDialogue task. (see https://parl.ai/projects/dodecadialogue/)
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        if not shared:
            self.idx_to_ep = {}
        else:
            self.idx_to_ep = shared['idx_to_ep']
        self.prepend_personality = opt.get('prepend_personality', True)
        self.include_dialogue_history = opt.get('include_dialogue_history', True)
        self.category_frac = opt.get('category_frac', 0.0)
        super().__init__(opt, shared)
        self.num_eps = len(self.data) + len(
            [d for d in self.data if len(d['dialog']) > 1]
        )

        # Replace personalities with polarity categories ("positive/neutral" or
        # "negative"), with probability self.category_frac
        if not shared:
            category_map = get_category_map(self.personalities)
            for i, d in enumerate(self.data):
                use_category_rand = random.random()
                if use_category_rand < self.category_frac:
                    self.data[i]['dialog'] = [
                        [category_map[personality], label]
                        for personality, label in d['dialog']
                    ]

    @staticmethod
    def add_cmdline_args(argparser):
        ImageChatTeacher.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('generation teacher arguments')
        agent.add_argument(
            '--prepend-personality',
            type='bool',
            default=True,
            help='if true, always prepend first turn text with the personality',
        )
        agent.add_argument(
            '--include-dialogue-history',
            type='bool',
            default=True,
            help='if false, remove the dialogue history',
        )
        agent.add_argument(
            '--category-frac',
            type=float,
            default=0.0,
            help='Fraction of the time to replace the personality with its polarity category ("positive/neutral" or "negative")',
        )

    def num_episodes(self) -> int:
        return self.num_eps

    def get(self, episode_idx: int, entry_idx: int = 0):
        entry_idx *= 2
        first_turn = entry_idx == 0
        if episode_idx >= len(self.data):
            # Second time through dataset
            data = self.data[self.idx_to_ep[episode_idx]]
            entry_idx += 1
        else:
            data = self.data[episode_idx]

        personality, label = data['dialog'][entry_idx]
        if not self.include_personality:
            personality = ''

        if entry_idx > 0:
            _, text = data['dialog'][entry_idx - 1]
            if not self.include_dialogue_history:
                text = ''
            if first_turn and self.prepend_personality and self.include_personality:
                text = '\n'.join([personality, text])
        elif self.prepend_personality and self.include_personality:
            text = personality
        else:
            text = ''

        episode_done = entry_idx >= len(data['dialog']) - 2

        action = {
            'text': text,
            'personality': personality,
            'image_id': data['image_hash'],
            'episode_done': episode_done,
            'labels': [label],
        }

        if "candidates" in data:
            action['label_candidates'] = data['candidates'][entry_idx][self.num_cands]

        return action

    def _setup_data(self, data_path: str, personalities_data_path: str):
        super()._setup_data(data_path, personalities_data_path)
        ep_idx = len(self.data)
        for i, d in enumerate(self.data):
            if len(d['dialog']) > 1:
                self.idx_to_ep[ep_idx] = i
                ep_idx += 1

    def share(self):
        shared = super().share()
        shared['idx_to_ep'] = self.idx_to_ep
        return shared


class ImageChatTestTeacher(ImageChatTeacher):
    """
    Test ImageChat teacher for ensuring pretrained model does not break.
    """

    def _setup_data(self, data_path, personalities_data_path):
        super()._setup_data(data_path, personalities_data_path)
        from parlai.zoo.image_chat.transresnet_multimodal import download

        download(self.opt['datapath'])
        image_features_path = os.path.join(
            self.opt['datapath'],
            'models/image_chat/transresnet_multimodal/test_image_feats',
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
        personality, text = data['dialog'][entry_idx]
        episode_done = entry_idx == len(data['dialog']) - 1

        action = {
            'text': personality if self.include_personality else '',
            'image': self.image_features[data['image_hash']],
            'episode_done': episode_done,
            'labels': [text],
        }

        if 'candidates' in data:
            action['label_candidates'] = data['candidates'][entry_idx][self.num_cands]

        return action


class DefaultTeacher(ImageChatTeacher):
    pass


def get_category_map(personalities: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Map personalities to polarity categories: "positive/neutral" and "negative".

    Given a dictionary mapping Image-Chat categories (positive/neutral/negative) to
    personalities, return a dictionary mapping each personality to its category.
    Categories are merged into only two buckets: "positive/neutral", for personalities
    that are more likely to be safe to use, and "negative". Add in rare personalities.
    """

    category_map = {
        personality: _get_final_category(category)
        for category, personalities in personalities.items()
        for personality in personalities
    }
    category_map['Crude'] = _get_final_category('negative')
    category_map['Earnest'] = _get_final_category('positive')
    # These personalities occasionally appear but are not in personalities
    return category_map


def _get_final_category(category: str) -> str:
    """
    Given the input raw category label, return the final one.
    """
    if category in ['positive', 'neutral']:
        return 'positive/neutral'
    elif category == 'negative':
        return 'negative'
    else:
        raise ValueError(f'Category "{category}" unrecognized!')
