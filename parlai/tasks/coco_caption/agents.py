#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build_2014 import build as build_2014
from .build_2014 import buildImage as buildImage_2014
from .build_2017 import build as build_2017
from .build_2017 import buildImage as buildImage_2017
try:
    import torch  # noqa: F401
except ImportError:
    raise ImportError('Need to install Pytorch: go to pytorch.org')
from torch.utils.data import Dataset

import os
import json
import random
"""
    Agents for MSCOCO Image Captioning Task

    There are two versions of the task - one comprising MSCOCO 2014 splits
    (from the 2015 task competition), and one comprising MSCOCO 2017 splits

    For the 2014 splits, we use the train, val, and test split of Karpathy et.
    al, "Deep visual-semantic alignments for generating image descriptions"
    (splits from here: https://cs.stanford.edu/people/karpathy/deepimagesent/).
    This split has ~82k train images, 5k validation images, and 5k test images.
    The val and test images are taken from the original validation set of ~40k.

    For 2017, we use the splits from the official MSCOCO Image Captioning 2017
    task.

"""
# There is no real dialog in this task, so for the purposes of display_data, we
# include a generic question that applies to all images.
QUESTION = "Describe the above picture in a sentence."


def load_candidates(datapath, datatype, version):
    if not datatype.startswith('train'):
        suffix = 'captions_{}{}.json'
        suffix_val = suffix.format('val', version)

        val_path = os.path.join(datapath,
                                'COCO_{}_Caption'.format(version),
                                'annotations',
                                suffix_val)
        val = json.load(open(val_path))['annotations']
        val_caps = [x['caption'] for x in val]
        if datatype.startswith('test'):
            suffix_train = suffix.format('train', version)
            train_path = os.path.join(datapath,
                                      'COCO_{}_Caption'.format(version),
                                      'annotations',
                                      suffix_train)

            train = json.load(open(train_path))['annotations']

            train_caps = [x['caption'] for x in train]
            test_caps = train_caps + val_caps
            return test_caps
        else:
            return val_caps
    else:
        return None


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
            'COCO_train{}_'.format(version) if version == '2014' else ''
        )
    elif dt == 'valid' or (dt == 'test' and version == '2014'):
        annotation_suffix = 'val{}'.format(version)
        img_suffix = os.path.join(
            'val{}'.format(version),
            'COCO_val{}_'.format(version) if version == '2014' else ''
        )
    elif dt == 'test':
        annotation_suffix = 'None'
        img_suffix = os.path.join(
            'test{}'.format(version),
            'COCO_test{}_'.format(version) if version == '2014' else ''
        )
    else:
        raise RuntimeError('Not valid datatype.')

    if version == '2017':
        test_info_path = os.path.join(opt['datapath'],
                                      'COCO_2017_Caption',
                                      'annotations',
                                      'image_info_test2017.json')

        annotation_path = os.path.join(opt['datapath'],
                                       'COCO_2017_Caption',
                                       'annotations',
                                       'captions_' + annotation_suffix + '.json')
    else:
        test_info_path = None
        annotation_path = os.path.join(opt['datapath'],
                                       'COCO_2014_Caption',
                                       'dataset_coco.json')

    image_path = os.path.join(opt['datapath'], 'COCO-IMG-{}'.format(version),
                              img_suffix)

    return test_info_path, annotation_path, image_path


class DefaultDataset(Dataset):
    """A Pytorch Dataset utilizing streaming."""

    def __init__(self, opt, version='2017'):
        self.opt = opt
        self.version = version
        self.use_intro = opt.get('use_intro', False)
        self.num_cands = opt.get('num_cands', -1)
        self.datatype = self.opt.get('datatype')
        self.include_rest_val = opt.get('include_rest_val', True)
        self.image_loader = ImageLoader(opt)
        test_info_path, annotation_path, self.image_path = _path(opt, version)
        self._setup_data(test_info_path, annotation_path, opt)

    @staticmethod
    def add_cmdline_args(argparser):
        DefaultTeacher.add_cmdline_args(argparser)

    def __getitem__(self, index):
        ep = {
            'episode_done': True
        }
        if self.use_intro:
            ep['text'] = QUESTION

        if hasattr(self, 'annotation'):
            anno = self.annotation[index]
        else:
            anno = self.test_info['images'][index]

        if self.version == '2014':
            ep['labels'] = [s['raw'] for s in anno['sentences']]
            ep['image_id'] = anno['cocoid']
            ep['split'] = anno['split']
        elif not self.datatype.startswith('test'):
            ep['image_id'] = anno['image_id']
            ep['labels'] = [anno['caption']]
        else:
            ep['image_id'] = anno['id']

        ep['image']: self.get_image(ep['image_id'], anno.get('split', None))

        if self.opt.get('extract_image', False):
            return ep

        # Add Label Cands
        if not self.datatype.startswith('train'):
            if self.num_cands == -1:
                ep['label_candidates'] = self.cands
            else:
                candidates = random.Random(index).choices(self.cands,
                                                          k=self.num_cands)
                label = random.choice(ep.get('labels', ['']))
                if not (label == '' or label in candidates):
                    candidates.pop(0)
                    candidates.append(label)
                    random.shuffle(candidates)
                ep['label_candidates'] = candidates

        return (index, ep)

    def __len__(self):
        return self.num_episodes()

    def _load_lens(self):
        with open(self.length_datafile) as length:
            lengths = json.load(length)
            self.num_eps = lengths['num_eps']
            self.num_exs = lengths['num_exs']

    def _setup_data(self, test_info_path, annotation_path, opt):
        if self.version == '2014':
            with open(annotation_path) as data_file:
                raw_data = json.load(data_file)['images']
            if 'train' in self.datatype:
                self.annotation = [d for d in raw_data if d['split'] == 'train']
                if self.include_rest_val:
                    self.annotation += [d for d in raw_data if d['split'] == 'restval']
            elif 'valid' in self.datatype:
                self.annotation = [d for d in raw_data if d['split'] == 'val']
                self.cands = [
                    l for d in self.annotation
                    for l in [
                        s['raw'] for s in d['sentences']
                    ]
                ]
            else:
                self.annotation = [d for d in raw_data if d['split'] == 'test']
                self.cands = [
                    l for d in self.annotation
                    for l in [
                        s['raw'] for s in d['sentences']
                    ]
                ]
        else:
            if not self.datatype.startswith('test'):
                print('loading: ' + annotation_path)
                with open(annotation_path) as data_file:
                    self.annotation = json.load(data_file)['annotations']
            else:
                print('loading: ' + test_info_path)
                with open(test_info_path) as data_file:
                    self.test_info = json.load(data_file)
            if not self.datatype.startswith('train'):
                self.cands = load_candidates(opt['datapath'],
                                             opt['datatype'],
                                             self.version)
        if opt.get('unittest', False):
            if not self.datatype.startswith('test'):
                self.annotation = self.annotation[:10]
            else:
                self.test_info['images'] = self.test_info['images'][:10]

    def get_image(self, image_id, split):
        if split == 'restval':
            im_path = self.image_path.replace('train', 'val')
        else:
            im_path = self.image_path
        im_path = os.path.join(im_path, '%012d.jpg' % (image_id))
        return self.image_loader.load(im_path)

    def num_examples(self):
        if self.version == '2014' or not self.datatype.startswith('test'):
            return len(self.annotation)
        else:
            # For 2017, we only have annotations for the train and val sets,
            # so for the test set we need to determine how many images we have.
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
        self.version = version
        self.image_mode = opt.get('image_mode', 'none')
        self.use_intro = opt.get('use_intro', False)
        self.num_cands = opt.get('num_cands', -1)
        self.include_rest_val = opt.get('include_rest_val', False)
        test_info_path, annotation_path, self.image_path = _path(opt, version)
        self.test_split = opt['test_split']

        if shared:
            # another instance was set up already, just reference its data
            if 'annotation' in shared:
                self.annotation = shared['annotation']
            self.image_loader = shared['image_loader']
            if 'cands' in shared:
                self.cands = shared['cands']
        else:
            # need to set up data from scratch
            self._setup_data(test_info_path, annotation_path, opt)
            self.image_loader = ImageLoader(opt)
        self.reset()

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('COCO Caption arguments')
        agent.add_argument('--use_intro', type='bool',
                           default=False,
                           help='Include an intro question with each image \
                                for readability (e.g. for coco_caption, \
                                Describe the above picture in a sentence.)')
        agent.add_argument('--num_cands', type=int,
                           default=150,
                           help='Number of candidates to use during \
                                evaluation, setting to -1 uses all.')
        agent.add_argument('--include_rest_val', type='bool',
                           default=False,
                           help='Include unused validation images in training')
        agent.add_argument('--test-split', type=int, default=-1,
                           choices=[-1, 0, 1, 2, 3, 4],
                           help='Which 1k image split of dataset to use for candidates'
                           'if -1, use all 5k test images')

    def reset(self):
        super().reset()  # call parent reset so other fields can be set up
        self.example = None  # set up caching fields
        self.imageEpochDone = False

    def num_examples(self):
        if self.version == '2014' or not self.datatype.startswith('test'):
            return len(self.annotation)
        else:
            # For 2017, we only have annotations for the train and val sets,
            # so for the test set we need to determine how many images we have.
            return len(self.test_info['images'])

    def num_episodes(self):
        return self.num_examples()

    def submit_load_request(self, image_id, split=None):
        if split == 'restval':
            img_path = self.image_path.replace('train', 'val')
        else:
            img_path = self.image_path
        img_path += '%012d.jpg' % (image_id)
        self.data_loader.request_load(self.receive_data,
                                      self.image_loader.load,
                                      (img_path,))

    def get(self, episode_idx, entry_idx=0):
        action = {
            'episode_done': True
        }

        if self.use_intro:
            action['text'] = QUESTION

        if self.version == '2014':
            ep = self.annotation[episode_idx]
            action['labels'] = [s['raw'] for s in ep['sentences']]
            action['image_id'] = ep['cocoid']
            action['split'] = ep['split']
            if not self.datatype.startswith('train'):
                if self.num_cands > 0:
                    labels = action['labels']
                    cands_to_sample = [c for c in self.cands if c not in labels]
                    cands = (
                        random.Random(episode_idx)
                              .sample(cands_to_sample, self.num_cands)
                    ) + labels
                    random.shuffle(cands)
                    action['label_candidates'] = cands
                else:
                    action['label_candidates'] = self.cands
        else:
            if not self.datatype.startswith('test'):
                # test set annotations are not available for this dataset
                anno = self.annotation[episode_idx]
                action['labels'] = [anno['caption']]
                action['image_id'] = anno['image_id']
                if not self.datatype.startswith('train'):
                    if self.num_cands == -1:
                        candidates = self.cands
                    else:
                        # Can only randomly select from validation set
                        candidates = random.Random(
                            episode_idx).choices(self.cands, k=self.num_cands)
                    if anno['caption'] not in candidates:
                        candidates.pop(0)
                    else:
                        candidates.remove(anno['caption'])

                    candidate_labels = [anno['caption']]
                    candidate_labels += candidates
                    action['label_candidates'] = candidate_labels
            else:
                if self.num_cands == -1:
                    candidates = self.cands
                else:
                    # Can only randomly select from validation set
                    candidates = random.Random(
                        episode_idx).choices(self.cands, k=self.num_cands)
                action['label_candidates'] = candidates
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
            split = self.example.get('split', None)
            self.submit_load_request(image_id, split)
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
        if hasattr(self, 'cands'):
            shared['cands'] = self.cands
        return shared

    def _setup_data(self, test_info_path, annotation_path, opt):
        if self.version == '2014':
            with open(annotation_path) as data_file:
                raw_data = json.load(data_file)['images']
            if 'train' in self.datatype:
                self.annotation = [d for d in raw_data if d['split'] == 'train']
                if self.include_rest_val:
                    self.annotation += [d for d in raw_data if d['split'] == 'restval']
            elif 'valid' in self.datatype:
                self.annotation = [d for d in raw_data if d['split'] == 'val']
                self.cands = [
                    l for d in self.annotation
                    for l in [
                        s['raw'] for s in d['sentences']
                    ]
                ]
            else:
                self.annotation = [d for d in raw_data if d['split'] == 'test']
                if self.test_split != -1:
                    start = self.test_split * 1000
                    end = (self.test_split + 1) * 1000
                    self.annotation = self.annotation[start:end]
                self.cands = [
                    l for d in self.annotation
                    for l in [
                        s['raw'] for s in d['sentences']
                    ]
                ]
        else:
            if not self.datatype.startswith('test'):
                print('loading: ' + annotation_path)
                with open(annotation_path) as data_file:
                    self.annotation = json.load(data_file)['annotations']
            else:
                print('loading: ' + test_info_path)
                with open(test_info_path) as data_file:
                    self.test_info = json.load(data_file)
            if not self.datatype.startswith('train'):
                self.cands = load_candidates(opt['datapath'],
                                             opt['datatype'],
                                             self.version)


class V2014Teacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super(V2014Teacher, self).__init__(opt, shared, '2014')


class V2017Teacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super(V2017Teacher, self).__init__(opt, shared, '2017')
