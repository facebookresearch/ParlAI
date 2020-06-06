#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Image Grounded Conversations (IGC) Task.

See https://www.aclweb.org/anthology/I17-1047/ for more details. One must download
the data from https://www.microsoft.com/en-us/download/details.aspx?id=55324
prior to using this teacher.

The images are then downloaded from the urls specified in the data. Unfortunately,
not all links are live, and thus some examples do not have valid images.

As there is no training set, we manually split 90% of the validation set
into train.
"""
import csv
import os

from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Dict, Any

from parlai.core.build_data import download_multiprocess
from parlai.core.params import Opt
from parlai.core.teachers import AbstractImageTeacher

import parlai.utils.typing as PT


class IGCTeacher(AbstractImageTeacher):
    """
    Teacher for Image Grounded Conversations (IGC) Task.

    See https://arxiv.org/abs/1701.08251 for more details
    """

    def __init__(self, opt: Opt, shared: PT.TShared = None):
        self.blank_image_id = '0000'
        super().__init__(opt, shared)
        if shared is not None:
            self.valid_image_ids = shared['valid_image_ids']
        if self.image_features_dict is not None:
            self.image_features_dict[self.blank_image_id] = self.blank_image_features
        self.multi_ref = opt.get('igc_multi_ref', False)

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Include arg.

        for multi-reference labels.
        """
        super().add_cmdline_args(argparser)
        agent = argparser.add_argument_group('IGC Arguments')
        agent.add_argument(
            '--igc-multi-ref',
            type='bool',
            default=False,
            help='specify to evaluate on multi-reference labels',
        )

    def image_id_to_image_path(self, image_id: str) -> str:
        """
        Return image path given image id.

        As this function is used in building the image features, and some of the

        :param image_id:
            image_id key, for IGC this is a str

        :return:
            the image path associated with the given image key
        """
        if image_id not in self.valid_image_ids:
            image_id = self.blank_image_id
        return os.path.join(self.get_image_path(self.opt), image_id)

    def get_data_path(self, opt: Opt) -> str:
        """
        Determines path to the data file.

        :param opt:
            Opt with all options

        :return:
            the path to the dataset
        """
        data_path = os.path.join(opt['datapath'], 'igc')
        return data_path

    def get_image_features_path(self, task, image_model_name, dt):
        """
        Override so that subclasses can see same image features.
        """
        # In default implementation, self.data_path already has task name added
        image_features_path = os.path.join(self.data_path, 'image_features')

        if not os.path.isdir(image_features_path):
            os.makedirs(image_features_path)

        return os.path.join(
            image_features_path, f'{image_model_name}_{dt}_features_dict'
        )

    def num_episodes(self) -> int:
        """
        Number of episodes.

        Iterate through each episode twice, playing each side of the conversation once.
        """
        return 2 * len(self.data)

    def num_examples(self) -> int:
        """
        Number of examples.

        There are three turns of dialogue in the IGC task -
        Context, Question, Response.

        Thus, return 3 * number of data examples.
        """
        return 3 * len(self.data)

    def get(self, episode_idx: int, entry_idx: int = 0) -> dict:
        """
        Override to handle corrupt images and multi-reference labels.
        """
        entry_idx *= 2

        if episode_idx >= len(self.data):
            data = self.data[episode_idx % len(self.data)]
            entry_idx += 1
        else:
            data = self.data[episode_idx]
        image_id = data[self.image_id_key]
        if data[self.image_id_key] not in self.valid_image_ids:
            data[self.image_id_key] = self.blank_image_id
        image_features = self.get_image_features(data)

        conversation = [data['context'], data['question'], data['response']]
        labels = [conversation[entry_idx]]
        if self.multi_ref and entry_idx != 0:
            key = 'questions' if entry_idx == 1 else 'responses'
            labels = data[f'multiref_{key}'].split('***')
        text = '' if entry_idx == 0 else conversation[entry_idx - 1]
        episode_done = entry_idx >= len(conversation) - 2

        action = {
            'text': text,
            'image_id': image_id,
            'episode_done': episode_done,
            'image': image_features,
            'labels': labels,
        }

        return action

    def load_data(self, data_path: str, opt: Opt) -> List[Dict[str, Any]]:
        """
        Override to load CSV files.
        """

        dt = opt['datatype'].split(':')[0]
        dt_str = 'test' if dt == 'test' else 'val'
        dp = os.path.join(self.get_data_path(opt), f'IGC_crowd_{dt_str}.csv')
        if not os.path.exists(dp):
            raise RuntimeError(
                'Please download the IGC Dataset from '
                'https://www.microsoft.com/en-us/download/details.aspx?id=55324. '
                'Then, make sure to put the two .csv files in {}'.format(
                    self.get_data_path(opt)
                )
            )
        if (
            not os.path.exists(self.get_image_path(opt))
            or len(os.listdir(self.get_image_path(opt))) <= 1
        ):
            self._download_images(opt)

        self.data = []
        with open(dp, newline='\n') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            fields = []
            for i, row in enumerate(reader):
                if i == 0:
                    fields = row
                else:
                    ep = dict(zip(fields, row))
                    ep['image_id'] = f'{ep["id"]}'
                    self.data.append(ep)

        if dt == 'train':
            # Take first 90% of valid set as train
            self.data = self.data[: int(len(self.data) * 0.9)]
        elif dt == 'valid':
            self.data = self.data[int(len(self.data) * 0.9) :]

        self.valid_image_ids = []
        for d in self.data:
            img_path = os.path.join(self.get_image_path(opt), d['image_id'])
            if os.path.isfile(img_path):
                self.valid_image_ids.append(d['image_id'])

        self.valid_image_ids = set(self.valid_image_ids)
        return self.data

    def _download_images(self, opt: Opt):
        """
        Download available IGC images.
        """
        urls = []
        ids = []
        for dt in ['test', 'val']:
            df = os.path.join(self.get_data_path(opt), f'IGC_crowd_{dt}.csv')
            with open(df, newline='\n') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                fields = []
                for i, row in enumerate(reader):
                    if i == 0:
                        fields = row
                    else:
                        data = dict(zip(fields, row))
                        urls.append(data['url'])
                        ids.append(data['id'])
        os.makedirs(self.get_image_path(opt), exist_ok=True)
        # Make one blank image
        image = Image.new('RGB', (100, 100), color=0)
        image.save(os.path.join(self.get_image_path(opt), self.blank_image_id), 'JPEG')
        # Download the rest
        download_multiprocess(urls, self.get_image_path(opt), dest_filenames=ids)

        # Remove bad images
        for fp in os.listdir(self.get_image_path(opt)):
            img_path = os.path.join(self.get_image_path(opt), fp)
            if os.path.isfile(img_path):
                try:
                    Image.open(img_path).convert('RGB')
                except OSError:
                    os.remove(img_path)

    def share(self) -> PT.TShared:
        shared = super().share()
        shared['valid_image_ids'] = self.valid_image_ids
        return shared


class IGCOneSideTeacher(ABC, IGCTeacher):
    """
    Override to only return one side of the conversation.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        super().add_cmdline_args(argparser)
        agent = argparser.add_argument_group('IGCResponseOnly Arguments')
        agent.add_argument(
            '--igc-multi-ref',
            type='bool',
            default=False,
            help='specify true to evaluate on multi-reference labels',
        )

    def num_episodes(self) -> int:
        return len(self.data)

    def num_examples(self) -> int:
        return len(self.data)

    @abstractmethod
    def get_label_key(self) -> str:
        """
        Return key into data dictionary for the label.
        """
        pass

    @abstractmethod
    def get_text(self, data) -> str:
        """
        Return text for an example.
        """
        pass

    def get(self, episode_idx: int, entry_idx: int = 0) -> Dict[str, Any]:
        """
        Override to handle one-sided conversation.
        """
        data = self.data[episode_idx]

        image_id = data[self.image_id_key]
        if data[self.image_id_key] not in self.valid_image_ids:
            data[self.image_id_key] = self.blank_image_id
        image_features = self.get_image_features(data)

        labels = [data[self.get_label_key()]]
        if self.multi_ref:
            labels = data[f'multiref_{self.get_label_key()}s'].split('***')

        text = self.get_text(data)

        action = {
            'text': text,
            'image_id': image_id,
            'episode_done': True,
            'image': image_features,
            'labels': labels,
        }

        return action


class ResponseOnlyTeacher(IGCOneSideTeacher):
    """
    Responses Only.
    """

    def get_label_key(self) -> str:
        return 'response'

    def get_text(self, data) -> str:
        return '\n'.join([data['context'], data['question']])


class QuestionOnlyTeacher(IGCOneSideTeacher):
    """
    Questions Only.
    """

    def get_label_key(self) -> str:
        return 'question'

    def get_text(self, data) -> str:
        return data['context']


class DefaultTeacher(IGCTeacher):
    pass
