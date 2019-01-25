#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.scripts.extract_image_feature import extract_feats
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage as buildImage_2014
from parlai.tasks.coco_caption.build_2015 import buildImage as buildImage_2015
try:
    import torch
except Exception as e:
    raise ImportError('Need to install Pytorch: go to pytorch.org')
from torch.utils.data import Dataset
from parlai.agents.mlb_vqa.mlb_vqa import VqaDictionaryAgent

import json
import os


def _path(opt):
    build(opt)
    buildImage_2014(opt)
    buildImage_2015(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        ques_suffix = 'MultipleChoice_mscoco_train2014'
        annotation_suffix = 'mscoco_train2014'
        img_suffix = os.path.join('train2014', 'COCO_train2014_')
        img_version = '2014'
    elif dt == 'valid':
        ques_suffix = 'MultipleChoice_mscoco_val2014'
        annotation_suffix = 'mscoco_val2014'
        img_suffix = os.path.join('val2014', 'COCO_val2014_')
        img_version = '2014'
    elif dt == 'test':
        ques_suffix = 'MultipleChoice_mscoco_test2015'
        annotation_suffix = 'None'
        img_suffix = os.path.join('test2015', 'COCO_test2015_')
        img_version = '2015'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'VQA-v1',
                             ques_suffix + '_questions.json')

    annotation_path = os.path.join(opt['datapath'], 'VQA-v1',
                                   annotation_suffix + '_annotations.json')

    image_path = os.path.join(opt['datapath'],
                              'COCO-IMG-{}'.format(img_version), img_suffix)

    return data_path, annotation_path, image_path


class VQADataset(Dataset):
    """A Pytorch Dataset utilizing streaming"""
    def __init__(self, opt):
        self.opt = opt
        self.use_att = opt.get('attention', False)
        self.use_hdf5 = opt.get('use_hdf5', False)
        self.opt['use_hdf5_extraction'] = self.use_hdf5
        self.datatype = self.opt.get('datatype')
        self.training = self.datatype.startswith('train')
        self.num_epochs = self.opt.get('num_epochs', 0)
        self.image_loader = ImageLoader(opt)
        data_path, annotation_path, self.image_path = _path(opt)
        self._setup_data(data_path, annotation_path, opt.get('unittest', False))
        if self.use_hdf5:
            try:
                import h5py
                self.h5py = h5py
            except ImportError:
                raise ImportError('Need to install h5py - `pip install h5py`')
            self._setup_image_data()
        self.dict_agent = VqaDictionaryAgent(opt)

    def __getitem__(self, index):
        index %= self.num_episodes()
        qa = self.ques['questions'][index]
        ep = {
            'text': qa['question'],
            'image': self.get_image(qa['image_id']),
            'episode_done': True,
        }
        if self.opt.get('extract_image', False):
            ep['image_id'] = qa['image_id']
            return ep
        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][index]
            labels = [ans['answer'] for ans in anno['answers']]
            ep['labels'] = [ans['answer'] for ans in anno['answers']]
            ep['valid'] = True
            if 'mc_label' in ep:
                if not ep['mc_label'][0] in self.dict_agent.ans2ind:
                    ep['valid'] = False
            ep = self.dict_agent.encode_question([ep], self.training)
            ep = self.dict_agent.encode_answer(ep)
            ep[0]['labels'] = labels
        else:
            ep['valid'] = True
            ep = self.dict_agent.encode_question([ep], False)
        ep[0]['use_att'] = self.use_att
        ep[0]['use_hdf5'] = self.use_hdf5
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

    def _setup_data(self, data_path, annotation_path, unittest):
        with open(data_path) as data_file:
            self.ques = json.load(data_file)
        if not self.datatype.startswith('test'):
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)
        if unittest:
            self.ques['questions'] = self.ques['questions'][:10]
            if not self.datatype.startswith('test'):
                self.annotation['annotations'] = self.annotation['annotations'][:10]
        self.image_paths = set()
        for qa in self.ques['questions']:
            self.image_paths.add(self.image_path + '%012d.jpg' % (qa['image_id']))

    def _setup_image_data(self):
        '''hdf5 image dataset'''
        extract_feats(self.opt)
        im = self.opt.get('image_mode')
        if self.opt.get('attention', False):
            hdf5_path = self.image_path + 'mode_{}.hdf5'.format(im)
        else:
            hdf5_path = self.image_path + 'mode_{}_noatt.hdf5'.format(im)
        hdf5_file = self.h5py.File(hdf5_path, 'r')
        self.image_dataset = hdf5_file['images']

        image_id_to_idx_path = self.image_path + 'mode_{}_id_to_idx.txt'.format(im)
        with open(image_id_to_idx_path, 'r') as f:
            self.image_id_to_idx = json.load(f)

    def get_image(self, image_id):
        if not self.use_hdf5:
            im_path = self.image_path + '%012d.jpg' % (image_id)
            return self.image_loader.load(im_path)
        else:
            img_idx = self.image_id_to_idx[str(image_id)]
            return torch.Tensor(self.image_dataset[img_idx])

    def num_episodes(self):
        return len(self.ques['questions'])

    def num_examples(self):
        return self.num_episodes()

    def num_images(self):
        if not hasattr(self, 'num_imgs'):
            self.num_imgs = len({q['image_id'] for q in self.ques['questions']})
        return self.num_imgs


class DefaultDataset(VQADataset):
    pass


class OeTeacher(FixedDialogTeacher):
    """
    VQA Open-Ended teacher, which loads the json vqa data and implements its
    own `act` method for interacting with student agent.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        data_path, annotation_path, self.image_path = _path(opt)
        self.datafile = data_path
        self.image_mode = opt.get('image_mode', 'none')

        if shared and 'ques' in shared:
            self.ques = shared['ques']
            if 'annotation' in shared:
                self.annotation = shared['annotation']
            self.image_loader = shared['image_loader']
        else:
            self._setup_data(data_path, annotation_path)
            self.image_loader = ImageLoader(opt)
        self.reset()

    def reset(self):
        super().reset()
        self.example = None

    def num_examples(self):
        """Number of examples in VQA-v1."""
        return len(self.ques['questions'])

    def num_episodes(self):
        # same as number of examples since all episodes are of length one
        return self.num_examples()

    def submit_load_request(self, image_id):
        img_path = self.image_path + '%012d.jpg' % (image_id)
        self.data_loader.request_load(
            self.receive_data, self.image_loader.load, (img_path,)
        )

    def get(self, episode_idx, entry_idx=0):
        # queue up the next one
        qa = self.ques['questions'][episode_idx]
        question = qa['question']

        action = {
            'text': question,
            'image_id': qa['image_id'],
            'episode_done': True
        }

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [ans['answer'] for ans in anno['answers']]

        return action

    def next_example(self):
        # save the currently queued example
        ready = None
        if self.example is not None:
            if self.image_mode != 'none':
                image = self.data_queue.get()
                self.example['image'] = image
            ready = (self.example, self.epochDone)
        # queue up the next example
        self.example, self.epochDone = super().next_example()
        if self.image_mode != 'none' and 'image_id' in self.example:
            image_id = self.example['image_id']
            self.submit_load_request(image_id)
        # Try to return the previously cached example
        if ready is None:
            return self.next_example()
        else:
            return ready

    def share(self):
        shared = super().share()
        shared['ques'] = self.ques
        if hasattr(self, 'annotation'):
            shared['annotation'] = self.annotation
        shared['image_loader'] = self.image_loader
        return shared

    def _setup_data(self, data_path, annotation_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.ques = json.load(data_file)

        if not self.datatype.startswith('test'):
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)


class McTeacher(OeTeacher):
    """
    VQA Multiple-Choice teacher, which inherits from OeTeacher but overrides
    the label and label_candidates fields with multiple choice data.
    """

    def get(self, episode_idx, entry_idx=0):
        action = super().get(episode_idx, entry_idx)
        qa = self.ques['questions'][episode_idx]
        multiple_choices = qa['multiple_choices']
        action['label_candidates'] = multiple_choices

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [anno['multiple_choice_answer']]

        return action


class AllTeacher(OeTeacher):
    """
    VQA Teacher, which inherits from OeTeacher and gives access to
    the multiple choices and the multiple choice answer.
    """

    def act(self):
        # parent class increments episode_idx after getting ex, so need to
        # cache the episode_idx first
        episode_idx = self.episode_idx
        action = super().act()

        qa = self.ques['questions'][episode_idx]
        multiple_choices = qa['multiple_choices']

        action['label_candidates'] = multiple_choices

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            self.mclabel = [anno['multiple_choice_answer']]

        if self.datatype.startswith('train'):
            action['mc_label'] = self.mclabel

        return action


class DefaultTeacher(McTeacher):
    # default to Multiple-Choice Teacher
    pass
