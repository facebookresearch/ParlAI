#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
import json
import os
import random

import numpy as np
import parlai.core.build_data as build_data
import parlai.utils.logging as logging
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/dialogue_safety/single_turn_safety.json',
        'single_turn_safety.json',
        'f3a46265aa639cfa4b55d2be4dca4be1c596acb5e8f94d7e0041e1a54cedd4cd',
        zipped=False,
    ),
]
# The size of safety mix
DATA_SIZE = {'train': 200, 'valid': 24, 'test': 24}

TROLL_TYPES = {
    'troll',
    'master_troll',
    'lazy_troll',
    'safe_troll',
    'unsafe_troll',
    'gaslight_troll',
}

FILE_TYPE_EXTENSIONS = {
    'train': '_train.jsonl',
    'valid': '_valid.jsonl',
    'test': '_test.jsonl',
}


# Dataset analysis
# check how many samples from standard vs adversarial
def analysis_dataset(dataset):
    standard_sample_cnt = 0
    adversarial_sample_cnt = 0
    flipped_label_cnt = 0
    duplicated_samples = {}
    for each_sample in dataset:
        if each_sample['source'] == 'standard':
            standard_sample_cnt += 1
        else:
            adversarial_sample_cnt += 1
        if each_sample['gold_labels'][0] != each_sample['labels'][0]:
            flipped_label_cnt += 1
        if each_sample['text'] in duplicated_samples:
            duplicated_samples[each_sample['text']] += 1
        else:
            duplicated_samples[each_sample['text']] = 1
    duplicated_samples = {
        k: v for k, v in sorted(duplicated_samples.items(), key=lambda item: item[1])
    }
    duplicated_samples = {k: v for k, v in duplicated_samples.items() if int(v) >= 2}

    all_values = list(duplicated_samples.values())
    max_value = 0 if len(all_values) == 0 else max(all_values)
    mean_value = 0 if len(all_values) == 0 else np.mean(all_values)
    logging.info(
        f'Real noise level {flipped_label_cnt / (standard_sample_cnt + adversarial_sample_cnt)}'
    )
    logging.info(
        f'Standard samples: {standard_sample_cnt} adversarial samples: {adversarial_sample_cnt}'
    )
    logging.info(
        f'It has {len(duplicated_samples)} duplicated samples, the the maximum appearance {max_value}, in average {mean_value}'
    )


def flip_labels(samples, N):
    flip_cnt = 0
    for sample in samples:
        text = sample['text']
        flip = hashlib.md5(text.encode('utf8')).digest()[0] < N * 256
        labels = sample['labels']
        # deterministic flipping for each example
        if flip:
            flip_cnt += 1
            if labels[0] == '__ok__':
                new_labels = ['__notok__']
            elif labels[0] == '__notok__':
                new_labels = ['__ok__']
            else:
                raise ValueError('labels are not in the right format.')
        if flip:
            sample['labels'] = new_labels
    logging.debug(f'{flip_cnt} labels flipped.')
    return samples


def sample_from_single_dataset(
    dataset_for_sample, source, utterance_num, user_id_to_start, random_int_for_sample
):
    utterance_cnt = 0
    user_id = user_id_to_start
    output_data = []
    while utterance_cnt < utterance_num:
        #     draw a number for user id
        current_user_utterance_num = int(random.choice(random_int_for_sample))
        if current_user_utterance_num + utterance_cnt > utterance_num:
            current_user_utterance_num = utterance_num - utterance_cnt
        utterance_cnt += current_user_utterance_num
        #     sample with replacement
        samples = copy.deepcopy(
            random.choices(dataset_for_sample, k=current_user_utterance_num)
        )
        for each_sample in samples:
            #         adding user id
            each_sample['user_id'] = user_id
            each_sample['source'] = source
            each_sample['gold_labels'] = copy.deepcopy(each_sample['labels'])
            output_data.append(each_sample)
        user_id += 1
    assert len(output_data) == utterance_num
    return output_data, user_id


def sample_from_both_datasets(
    standard_dataset_for_sample,
    adversarial_dataset_for_sample,
    utterance_num,
    user_id_to_start,
    random_int_for_sample,
):
    utterance_cnt = 0
    user_id = user_id_to_start
    output_data = []
    while utterance_cnt < utterance_num:
        #     draw a number for user id
        current_user_utterance_num = int(random.choice(random_int_for_sample))
        if current_user_utterance_num + utterance_cnt > utterance_num:
            current_user_utterance_num = utterance_num - utterance_cnt
        utterance_cnt += current_user_utterance_num
        #     sample with replacement
        #     flip a coin
        samples = []
        for _ in range(current_user_utterance_num):
            if random.random() > 0.5:
                sample = copy.deepcopy(random.choice(standard_dataset_for_sample))
                sample['source'] = 'standard'
            else:
                sample = copy.deepcopy(random.choice(adversarial_dataset_for_sample))
                sample['source'] = 'adversarial'
            samples.append(sample)
        for each_sample in samples:
            #         adding user id
            each_sample['user_id'] = user_id
            each_sample['gold_labels'] = copy.deepcopy(each_sample['labels'])
            output_data.append(each_sample)
        user_id += 1
    assert len(output_data) == utterance_num
    return output_data, user_id


def set_labels(samples, label):
    for sample in samples:
        sample['labels'] = [label]
    return samples


def set_user_type(samples, user_type):
    for sample in samples:
        sample['user_type'] = user_type
    return samples


def set_random_labels(samples, N):
    random_cnt = 0
    for sample in samples:
        text = sample['text']
        change_label = hashlib.md5(text.encode('utf8')).digest()[0] < N * 256
        # deterministic flipping for each example
        if change_label:
            random_cnt += 1
        if random.random() > 0.5:
            sample['labels'] = ['__ok__']
        else:
            sample['labels'] = ['__notok__']
    logging.debug(f'{random_cnt} labels randomized.')
    return samples


def generate_user_based_dataset(
    N, generation_fold, safety_dataset, user_ratios, random_int_for_sample
):
    total_number_of_samples = DATA_SIZE[generation_fold]
    # Take the first 800 samples from safety set to make it balanced between good and bad.
    standard_safe = safety_dataset['standard'][generation_fold]['1']['good'][:800]
    standard_unsafe = safety_dataset['standard'][generation_fold]['1']['bad'][:800]
    adversarial_safe = safety_dataset['adversarial'][generation_fold]['1']['good'][:800]
    adversarial_unsafe = safety_dataset['adversarial'][generation_fold]['1']['bad'][
        :800
    ]
    standard_dataset_for_sample = random.sample(
        standard_safe + standard_unsafe, len(standard_safe + standard_unsafe)
    )
    adversarial_dataset_for_sample = random.sample(
        adversarial_safe + adversarial_unsafe,
        len(adversarial_safe + adversarial_unsafe),
    )
    user_id = 0

    # for helpers
    helper_utterance_num = int(total_number_of_samples * user_ratios['helper'])
    helper_samples, user_id = sample_from_single_dataset(
        standard_dataset_for_sample,
        'standard',
        helper_utterance_num,
        user_id,
        random_int_for_sample,
    )
    helper_samples = set_user_type(helper_samples, 'helper')

    # for master helper
    master_helper_utterance_num = int(
        total_number_of_samples * user_ratios['master_helper']
    )
    master_helper_samples, user_id = sample_from_single_dataset(
        adversarial_dataset_for_sample,
        'adversarial',
        master_helper_utterance_num,
        user_id,
        random_int_for_sample,
    )
    master_helper_samples = set_user_type(master_helper_samples, 'master_helper')

    # for trolls
    troll_utterance_num = int(total_number_of_samples * user_ratios['troll'])
    troll_samples, user_id = sample_from_single_dataset(
        standard_dataset_for_sample,
        'standard',
        troll_utterance_num,
        user_id,
        random_int_for_sample,
    )
    # flip samples for trolls
    troll_samples = flip_labels(troll_samples, N)
    troll_samples = set_user_type(troll_samples, 'troll')

    # sample master trolls
    master_troll_utterance_num = int(
        total_number_of_samples * user_ratios['master_troll']
    )
    master_troll_samples, user_id = sample_from_single_dataset(
        adversarial_dataset_for_sample,
        'adversarial',
        master_troll_utterance_num,
        user_id,
        random_int_for_sample,
    )
    # flip samples for master trolls
    master_troll_samples = flip_labels(master_troll_samples, N)
    master_troll_samples = set_user_type(master_troll_samples, 'master_troll')

    # sample safe trolls
    safe_troll_utterance_num = int(total_number_of_samples * user_ratios['safe_troll'])
    safe_troll_samples, user_id = sample_from_both_datasets(
        standard_dataset_for_sample,
        adversarial_dataset_for_sample,
        safe_troll_utterance_num,
        user_id,
        random_int_for_sample,
    )
    safe_troll_samples = set_labels(safe_troll_samples, '__ok__')
    safe_troll_samples = set_user_type(safe_troll_samples, 'safe_troll')

    # sample unsafe trolls
    unsafe_troll_utterance_num = int(
        total_number_of_samples * user_ratios['unsafe_troll']
    )
    unsafe_troll_samples, user_id = sample_from_both_datasets(
        standard_dataset_for_sample,
        adversarial_dataset_for_sample,
        unsafe_troll_utterance_num,
        user_id,
        random_int_for_sample,
    )
    unsafe_troll_samples = set_labels(unsafe_troll_samples, '__notok__')
    unsafe_troll_samples = set_user_type(unsafe_troll_samples, 'unsafe_troll')

    # happy helper
    happy_helper_utterance_num = int(
        total_number_of_samples * user_ratios['happy_helper']
    )
    happy_helper_samples, user_id = sample_from_single_dataset(
        random.sample(standard_safe, len(standard_safe)),
        'standard',
        happy_helper_utterance_num,
        user_id,
        random_int_for_sample,
    )
    happy_helper_samples = set_user_type(happy_helper_samples, 'happy_helper')

    # gaslight trolls
    gaslight_troll_utterance_num = int(
        total_number_of_samples * user_ratios['gaslight_troll']
    )
    gaslight_troll_samples, user_id = sample_from_single_dataset(
        random.sample(adversarial_unsafe, len(adversarial_unsafe)),
        'adversarial',
        gaslight_troll_utterance_num,
        user_id,
        random_int_for_sample,
    )
    gaslight_troll_samples = set_labels(gaslight_troll_samples, '__ok__')
    gaslight_troll_samples = set_user_type(gaslight_troll_samples, 'gaslight_troll')

    # lazy trolls
    lazy_troll_utterance_num = int(total_number_of_samples * user_ratios['lazy_troll'])
    lazy_troll_samples, user_id = sample_from_both_datasets(
        standard_dataset_for_sample,
        adversarial_dataset_for_sample,
        lazy_troll_utterance_num,
        user_id,
        random_int_for_sample,
    )
    lazy_troll_samples = set_random_labels(lazy_troll_samples, N)
    lazy_troll_samples = set_user_type(lazy_troll_samples, 'lazy_troll')

    dataset = (
        helper_samples
        + master_helper_samples
        + troll_samples
        + master_troll_samples
        + safe_troll_samples
        + unsafe_troll_samples
        + happy_helper_samples
        + gaslight_troll_samples
        + lazy_troll_samples
    )
    return dataset


def save_dataset_for_two_class(dataset, troll_type, output_data_path, generation_fold):
    # split dataset into postive and negative for balanced training
    output_file_name = os.path.join(
        output_data_path, troll_type + FILE_TYPE_EXTENSIONS[generation_fold]
    )
    with open(output_file_name, 'w') as outfile:
        for json_blob in dataset:
            outfile.write(json.dumps(json_blob))
            outfile.write("\n")

    dataset_pos = []
    dataset_neg = []
    for each_sample in dataset:
        if each_sample['labels'][0] == '__ok__':
            dataset_pos.append(each_sample)
        else:
            dataset_neg.append(each_sample)
    output_file_name = os.path.join(
        output_data_path, 'pos_' + troll_type + FILE_TYPE_EXTENSIONS[generation_fold]
    )
    with open(output_file_name, 'w') as outfile:
        for json_blob in dataset_pos:
            outfile.write(json.dumps(json_blob))
            outfile.write("\n")
    output_file_name = os.path.join(
        output_data_path, 'neg_' + troll_type + FILE_TYPE_EXTENSIONS[generation_fold]
    )
    with open(output_file_name, 'w') as outfile:
        for json_blob in dataset_neg:
            outfile.write(json.dumps(json_blob))
            outfile.write("\n")


def build(datapath):
    version = 'v1.0'
    dpath = os.path.join(datapath, 'safety_mix')

    if not build_data.built(dpath, version):
        logging.info('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)
        random.seed(42)
        np.random.seed(42)
        # The number of user utterances for sampling
        random_nums = np.random.normal(loc=10, scale=2, size=100000)
        random_int_for_sample = np.round(random_nums)
        # load safety dataset
        with open(os.path.join(dpath, 'single_turn_safety.json')) as safety_file:
            safety_data = json.load(safety_file)
        for troll_type in TROLL_TYPES:
            # by default we do a 50% 50% mix with helpers and trolls
            ratios = {
                'helper': 0.5,
                'troll': 0.0,
                'master_helper': 0.0,
                'master_troll': 0.0,
                'lazy_troll': 0.0,
                'safe_troll': 0.0,
                'unsafe_troll': 0.0,
                'gaslight_troll': 0.0,
                'happy_helper': 0.0,
            }
            # Set the troll type 50% accordingly
            ratios[troll_type] = 0.5
            # Set the noise ratio N for some type of trolls
            N = 0.8
            for generation_fold in ['train', 'valid', 'test']:
                # for deterministic data generation
                random.seed(42)
                np.random.seed(42)
                dataset = generate_user_based_dataset(
                    N, generation_fold, safety_data, ratios, random_int_for_sample
                )
                if generation_fold == 'train':
                    logging.info(f'Stats for troll type {troll_type}')
                    analysis_dataset(dataset)
                #         save_dataset(dataset)
                save_dataset_for_two_class(dataset, troll_type, dpath, generation_fold)
                # Mark the data as built.
                build_data.mark_done(dpath, version)
