#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
import random
from abc import abstractmethod

import numpy as np

from parlai.core.teachers import FixedDialogTeacher


class AbstractMixedCandidatesTeacher(FixedDialogTeacher):
    """
    This teacher can do mixed candidate evaluation of a model where examples come from a
    specified task (self.mc_task) but N more inline candidates are added from each of
    the other datasets specified in the map self.teachers_map which is created by
    abstractmethod in subclass: create_dataset_teachers_map.
    """

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.datapath = os.path.join(opt['datapath'], 'blended_skill_talk')
        self.mc_task = opt.get('mc_task', None)
        self.debug = False

        if not self.mc_task:
            raise Exception('Must specify which task to evaluate.')

        if self.opt.get('candidates') != 'inline':
            raise Exception(
                f'Mixed candidate mode must be used with \"inline\" but was {self.opt.get("candidates")}'
            )

        if not shared:
            # is primary
            print(f'[AbstractMixedCandidatesTeacher] task: {self.mc_task}')
            self.build(opt)
            self.teachers_map = self.create_dataset_teachers_map(opt)
            self.data = self._setup_mixed_candidates()
        else:
            self.data = shared['data']
            self.teachers_map = shared['teachers_map']

        super().__init__(opt, shared)
        self.reset()

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument(
            '--mc-task',
            type=str,
            default=None,
            help='Task to evaluate on. Must match a key in map returned by create_dataset_teachers_map',
        )

    @abstractmethod
    def create_dataset_teachers_map(self, opt):
        """
        Implemented by subclass.

        :return: Must return a map of dataset label to an instantiated teacher
            (e.g. {'__WIZARD__': <instance of Wizard teacher>, ...})
        """
        raise NotImplementedError('Subclass must implement this!')

    def build(self, opt):
        """
        Subclass can do something if it wants.
        """
        return

    def num_episodes(self):
        return self.teachers_map[self.mc_task].num_episodes()

    def num_examples(self):
        return self.teachers_map[self.mc_task].num_examples()

    def epoch_done(self):
        # return super().epoch_done()
        return self.teachers_map[self.mc_task].epoch_done()

    def reset(self):
        super().reset()
        self.example = None

    def get_mixed_candidates_path(self, opt):
        return os.path.join(
            opt['datapath'],
            f'mixed_candidates_{self.__class__.__name__}_{self.mc_task}_{self.opt["datatype"]}.cands',
        )

    def _setup_mixed_candidates(self):
        mixed_candidates_file = self.get_mixed_candidates_path(self.opt)
        print(f'mixed_candidates_file: {mixed_candidates_file}')

        if os.path.isfile(mixed_candidates_file):
            print(
                f'Mixed candidates file: {mixed_candidates_file} exists for task {self.mc_task}.'
            )
            with open(mixed_candidates_file) as f:
                data = json.load(f)
                print(f'Loaded and returned mixed candidates for {len(data)} episodes.')
            return data

        dataset_teacher = self.teachers_map[self.mc_task]
        outside_datasets = [l for l in self.teachers_map.keys() if l != self.mc_task]
        outside_dataset_labels = {}
        for dataset_label in outside_datasets:
            outside_dataset_labels[dataset_label] = self._get_all_labels_for_dataset(
                dataset_label
            )
        dataset_episodes = dataset_teacher.num_episodes()
        all_modified_episodes = []
        for i in range(0, dataset_episodes):
            j = 0
            modified_episode = []
            while True:
                example = dataset_teacher.get(i, entry_idx=j)
                n_cands = len(example['label_candidates'])
                additional_inline_candidates = []
                for dataset in outside_datasets:
                    potential_labels = outside_dataset_labels[dataset]
                    dset_labels_count = len(potential_labels)
                    for _ in range(0, n_cands):
                        random_int = np.random.randint(0, dset_labels_count)
                        additional_inline_candidates.append(
                            potential_labels[random_int]
                        )
                all_inline_candidates = (
                    list(example['label_candidates']) + additional_inline_candidates
                )
                random.shuffle(all_inline_candidates)
                modified_example = copy.deepcopy(example)
                modified_example['label_candidates'] = all_inline_candidates
                modified_episode.append(modified_example)
                # TODO: if a duplicate utterance has different inline
                # candidates, then this won't have the same N candidates
                # from the dataset for duplicate (small thing)
                # print(f'Example in mixed candidates mode will have {len(all_inline_candidates)} candidates for {self.mc_task}.')
                # mixed_candidates[example['text']] = all_inline_candidates
                if example['episode_done']:
                    break
                j += 1
            all_modified_episodes.append(modified_episode)
        with open(mixed_candidates_file, 'w+') as f:
            print(
                f'Writing mixed candidates for {len(all_modified_episodes)} episodes to file {mixed_candidates_file} for task {self.mc_task}.'
            )
            f.write(json.dumps(all_modified_episodes))

        return all_modified_episodes

    def _get_all_labels_for_dataset(self, dataset_label):
        """
        Get all the possible labels from all the examples/episodes for a given dataset.
        """
        print(
            f'[AbstractMixedCandidateTeacher] Getting all labels for {dataset_label}...'
        )
        dataset_teacher = self.teachers_map[dataset_label]
        all_labels = []
        dataset_episodes = dataset_teacher.num_episodes()
        for i in range(0, dataset_episodes):
            j = 0
            while True:
                try:
                    example = dataset_teacher.get(i, entry_idx=j)
                except Exception as exc:
                    # Hopefully this only occurs while building the
                    # dictionary, which we don't need (?)
                    # Breaks due to date type of self.data call inside
                    # core/teachers.py line 512
                    print(f'Exception {exc} while getting next entry.')
                all_labels.append(example['labels'][0])
                if example['episode_done']:
                    break
                j += 1
        return all_labels

    def get(self, episode_idx, entry_idx=0, preprocessing=False):
        # when you do bs > 1, it goes 0 to 99, 0 to 99, until certain episodes
        # end. then it starts interleaving 100, 101, etc.
        example = self.data[episode_idx][entry_idx]

        if not example["labels"][0] in example["label_candidates"]:
            print(
                f'WARNING: Label was not in label_candidates {example["labels"][0]}. No chance to get this right.'
            )
        example['ground_truth_dataset'] = self.mc_task
        return example

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['teachers_map'] = self.teachers_map
        return shared
