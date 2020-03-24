#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from abc import abstractmethod

import numpy as np

from parlai.core.teachers import FixedDialogTeacher, create_task_agent_from_taskname


class DatasetClassificationTeacherBase(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        self.opt = opt
        self.datapath = os.path.join(opt['datapath'], 'blended_skill_talk')

        # Would likely be True for training the classifier
        # And False for getting utterances to cache classifier predictions
        # during Two Stage Model eval (b/c want all the utterances in that case)
        # See preprocess_utterance() function below
        self.do_process_utterances = (
            opt['process_utterances'] if 'process_utterances' in opt else True
        )

        if not shared:
            # is primary
            self.build(opt)
            self.teachers_map = self.create_dataset_teachers_map(opt)
            self.data = self._setup_for_classifier()
            self._num_examples = len(self.data)
            self._num_episodes = len(self.data)
        else:
            self.data = shared['data']
            self.teachers_map = shared['teachers_map']
            self._num_episodes = shared['_num_episodes']
            self._num_examples = shared['_num_examples']

        super().__init__(opt, shared)
        self.reset()

    @abstractmethod
    def create_dataset_teachers_map(self, opt):
        """
        Implemented by subclass.

        :return: Must return a map of dataset label to an instantiated teacher
            (e.g. {'__WIZARD__': <instance of Wizard teacher>, ...})
        """
        raise NotImplementedError('Subclass must implement this method!')

    def build(self, opt):
        """
        subclass can do something if it wants.
        """
        return

    @abstractmethod
    def preprocess_utterance(self, utterance):
        """
        Do something to utterance before including For example: we have removed lines
        with \n (topics for Wizard or personas for persona chat)
        """
        raise NotImplementedError('Subclass must implement!')

    def _setup_for_classifier(self):
        """
        Collect the utterances to be used to train/eval the classifier.
        """
        # map of utterance to boolean across all datasets - to track dups
        utterances_map_for_duplicates = {}
        # map of dataset label to array of non-duplicate observation objects
        utterances_map = {}
        for dataset_label, teacher in self.teachers_map.items():
            print(f'Setting up classifier utterances for {dataset_label}')
            utterances_per_dataset = []
            for i in range(0, teacher.num_episodes()):
                j = 0
                while True:
                    try:
                        example = teacher.get(i, j)
                    except Exception:
                        # Will get here in ConvAI2 during dictionary build I
                        # think with some error saying .get() takes no arguments
                        break

                    # For both wizard and persona we want to remove the part of
                    # any utterance which occurs before a newline character
                    if self.do_process_utterances:
                        processed_utterance = self.preprocess_utterance(example['text'])
                    else:
                        processed_utterance = example['text']
                    if processed_utterance:
                        # For classifier don't add duplicates
                        if processed_utterance not in utterances_map_for_duplicates:
                            copied_example = copy.deepcopy(example)
                            copied_example.force_set('episode_done', True)
                            copied_example.force_set('entry_idx', 0)
                            copied_example.force_set('labels', dataset_label)
                            copied_example.force_set('text', processed_utterance)
                            utterances_map_for_duplicates[processed_utterance] = [
                                dataset_label
                            ]
                            utterances_per_dataset.append(copied_example)
                        else:
                            # print(f'Skipping duplicate {processed_utterance},
                            # dataset: {dataset_label}.')
                            utterances_map_for_duplicates[processed_utterance].append(
                                dataset_label
                            )
                    j += 1
                    if example['episode_done']:
                        break
            utterances_map[dataset_label] = utterances_per_dataset

        # Upsample the observations so all datasets are equally represented
        max_dataset_len = max([len(d) for d in list(utterances_map.values())])
        final_utterances = []
        for _dataset_label, dataset_observations in utterances_map.items():
            dataset_len = len(dataset_observations)
            upsample_num = max_dataset_len - dataset_len
            if dataset_len == 0:
                # Hopefully this only happens during dict building for ConvAI2
                # where there's some weird get() error as per above
                print('Warning! Dataset was length 0. Skipping upsampling')
                continue
            for _ in range(0, upsample_num):
                random_int = np.random.randint(0, dataset_len)
                dataset_observations.append(dataset_observations[random_int])
            final_utterances.extend(dataset_observations)
        print(f'Returning {len(final_utterances)} utterances for classifier.')

        # Summarize how many utterances are in more than one dataset
        # (Actually skipping the duplicates is handled above)
        duplicates_summary = ''
        for dataset_label in self.teachers_map.keys():
            duplicates_for_dataset_count = 0
            for _utt, dataset_list in utterances_map_for_duplicates.items():
                if dataset_label in dataset_list and len(dataset_list) > 1:
                    duplicates_for_dataset_count += 1
            duplicates_summary += f' {dataset_label}: {duplicates_for_dataset_count}'
        print(
            f'Number of utterances found in more than one dataset: {duplicates_summary}'
        )

        # Summarize how many examples are utterances that are in > 1 dataset
        for dataset_label, teacher in self.teachers_map.items():
            utterances_per_dataset = []
            examples_with_duplicate_utterance_count = 0
            for i in range(0, teacher.num_episodes()):
                j = 0
                while True:
                    try:
                        example = teacher.get(i, j)
                        utt = self._process_utterance_for_classifier(example['text'])
                        if len(utterances_map_for_duplicates[utt]) > 1:
                            examples_with_duplicate_utterance_count += 1
                    except Exception:
                        # Will get here in ConvAI2 during dictionary build I
                        # think with some error saying .get() takes no arguments
                        # print(f'{dataset_label} had Exception: {exc}')
                        break
                    if example['episode_done']:
                        break
                    j += 1
            print(
                f'{dataset_label} had {examples_with_duplicate_utterance_count} '
                f'examples with a duplicated utterance.'
            )
        return final_utterances

    def get(self, episode_idx, entry_idx=0, preprocessing=False):
        # when you do bs > 1, it goes 0 to 99, 0 to 99, until certain episodes
        # end. then it starts interleaving 100, 101, etc.
        return self.data[episode_idx]

    def num_episodes(self):
        return self._num_episodes

    def num_examples(self):
        return self._num_examples

    def epoch_done(self):
        return super().epoch_done()

    def reset(self):
        super().reset()
        self.example = None

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['teachers_map'] = self.teachers_map
        shared['_num_examples'] = self._num_examples
        shared['_num_episodes'] = self._num_episodes
        return shared


class SingleDatasetClassificationTeacher(DatasetClassificationTeacherBase):
    @staticmethod
    def add_cmdline_args(parser):
        # A single teacher to pull examples from and to separate them into their
        # episodes.
        # NOTE 1: this wouldn't be used for training b/c there's only
        # one dataset to classify! Only used for eval of an existing pretrained
        # classifier on a dataset
        parser.add_argument(
            '--classifier-task',
            type=str,
            default=None,
            help='Task to pull utterances from and to separate into single episodes.',
        )
        parser.add_argument(
            '--classifier-task-label-name',
            type=str,
            default=None,
            help='Label of classifier task. Must correspond to one of the labels that the classifier was originally trained on.',
        )

    def create_dataset_teachers_map(self, opt):
        """
        Implement the superclass' abstract method.

        :return: map of {<dataset label>: <teacher>}
        """
        new_opt = copy.deepcopy(opt)
        new_opt['task'] = opt.get('classifier_task')
        teacher = create_task_agent_from_taskname(new_opt)[0]
        return {self.opt['classifier_task_label_name']: teacher}

    def preprocess_utterance(self, utterance):
        """
        Implement superclass' abstract method.
        """
        return utterance
