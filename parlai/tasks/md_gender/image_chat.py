#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.md_gender.build import build
from parlai.tasks.image_chat.agents import ImageChatTeacher as OrigImageTeacher
import parlai.tasks.md_gender.utils as gend_utils


from copy import deepcopy
import json
import os
import random


class ImageChatTeacher(FixedDialogTeacher):
    """
    Image Chat gender.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        argparser = gend_utils.add_common_args(argparser)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.is_train = 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']
        self.is_valid = 'valid' in opt['datatype']
        self.add_unknown_classes = opt['add_unknown_classes'] and self.is_train
        if shared is None:
            # set map
            self.data = self._setup_data(opt)
            if (self.is_train and opt['balance']) or (
                self.is_valid and opt['balance_valid']
            ):
                to_exclude = gend_utils.PARTNER_CANDS + gend_utils.SELF_CANDS
                self.data = gend_utils.balance_data(
                    self.data, key=1, exclude_labels=to_exclude
                )
        else:
            self.data = shared['data']
        super().__init__(opt, shared)

        self.label_candidates = gend_utils.ALL_CANDS

        self.reset()

    def _load_gender_data(self, opt):
        """
        Get data file.
        """
        build(opt)
        dt = opt['datatype'].split(':')[0]
        folder = os.path.join(
            opt['datapath'], 'md_gender', 'data_to_release', 'image_chat'
        )
        fle = 'engaging_imagechat_gender_captions_hashed.{}.jsonl'.format(dt)
        return os.path.join(folder, fle)

    def _setup_data(self, opt):
        # Load map from image ID to gender
        counts = {gend_utils.FEM: 0, gend_utils.MASC: 0, gend_utils.NEUTRAL: 0}
        self.image_id_to_gender = {}
        gender_data_fle = self._load_gender_data(opt)
        with open(gender_data_fle, 'r') as f:
            for line in f:
                gender_dct = json.loads(line)
                image_id = gender_dct['id']
                male = gender_dct['male']
                female = gender_dct['female']
                if not male and not female:
                    gender = gend_utils.NEUTRAL
                elif male and not female:
                    gender = gend_utils.MASC
                elif female and not male:
                    gender = gend_utils.FEM
                else:
                    # detected both a man and a woman in the photo
                    gender = gend_utils.NEUTRAL
                self.image_id_to_gender[image_id] = gender

        # Now load the captions
        image_teacher_opt = deepcopy(opt)
        image_teacher_opt['include_image'] = False
        orig_data = OrigImageTeacher(image_teacher_opt).data
        data = []
        missing_cnt = 0
        extra_data = []
        for ex in orig_data:
            image_num = ex['image_hash']
            gender = self.image_id_to_gender.get(image_num)
            if gender is None:
                missing_cnt += 1
                continue
            for dialog in ex['dialog']:
                text = dialog[1]
                label = f'ABOUT:{gender}'
                data.append((text, label, image_num, 'about'))
                counts[gender] += 1

                if self.add_unknown_classes:
                    extra_data.append(
                        (text, f'SELF:{gend_utils.NEUTRAL}', image_num, 'self')
                    )
                    extra_data.append(
                        (text, f'PARTNER:{gend_utils.NEUTRAL}', image_num, 'partner')
                    )

        # possibly sample the added unknown class data
        if len(extra_data) > 0:
            sample_rate = self.opt['unknown_temp']
            if sample_rate < 1.0:
                to_samp = int(sample_rate * len(extra_data))
                sampled = random.sample(extra_data, to_samp)
                data += sampled
            else:
                data += extra_data

        demarc = '=' * 50
        print('\n\n' + demarc)
        print('Gender breakdown:')
        tot = sum(counts.values())
        for k, v in counts.items():
            print(f'{k}: {v} ({v / tot})')
        print(f'Total: {tot}')
        print(demarc + '\n\n')

        if self.is_train:
            random.shuffle(data)

        return data

    def get(self, episode_idx, entry_idx=0):
        text, label, image_num, class_type = self.data[episode_idx]
        if class_type == 'self' or class_type == 'partner':
            # not TRUE neutral, sample all labels
            labels = gend_utils.UNKNOWN_LABELS[class_type]
        else:
            labels = [label]
        return Message(
            {
                'text': text,
                'labels': labels,
                'label_candidates': self.label_candidates[class_type],
                'episode_done': True,
                'id': 'ImageChat Gender',
                'image_id': image_num,
            }
        )

    def num_examples(self):
        return len(self.data)

    def num_episodes(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared
