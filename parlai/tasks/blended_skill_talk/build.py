#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import json
import os

from parlai.core import build_data


RESOURCES = [
    build_data.DownloadableFile(
        'http://parl.ai/downloads/blended_skill_talk/blended_skill_talk.tar.gz',
        'blended_skill_talk.tar.gz',
        '5fbed0068ee89e2d43b93c3ecb341e784617033efa5e8e911a219d4eda6134a6',
    ),
    build_data.DownloadableFile(
        'http://parl.ai/downloads/blended_skill_talk/personas_list.txt',
        'persona_list.txt',
        '59a51adedc78e806a380f16477de3740cefe3494d20f8a2a733841bedaaa3ee5',
        zipped=False,
    ),
    build_data.DownloadableFile(
        'http://parl.ai/downloads/blended_skill_talk/topic_to_persona_list.txt',
        'topic_to_persona_list.txt',
        '47cdb6cbee0516ca7400be35fa07761339b86c6c026425bf5dba00e5534e8182',
        zipped=False,
    ),
    build_data.DownloadableFile(
        'http://parl.ai/downloads/blended_skill_talk/ed_persona_topicifier__train__both_sides.json',
        'ed_persona_topicifier__train__both_sides.json',
        'ff2ea7c5fcb0449890d57a629cc3e8794ab95ac6db1057bf58d540c2b576e4cc',
        zipped=False,
    ),
    build_data.DownloadableFile(
        'http://parl.ai/downloads/blended_skill_talk/ed_persona_topicifier__train__experiencer_only.json',
        'ed_persona_topicifier__train__experiencer_only.json',
        '751f0ba2f421a11eee2bfc896d60ab70d669093c3a5f6cb30e8d202133a90ec7',
        zipped=False,
    ),
    build_data.DownloadableFile(
        'http://parl.ai/downloads/blended_skill_talk/ed_persona_topicifier__valid__experiencer_only.json',
        'ed_persona_topicifier__valid__experiencer_only.json',
        '15d5412f5990a8a9c892305009d8597a737322aafe878b03ec71143703b25ba0',
        zipped=False,
    ),
    build_data.DownloadableFile(
        'http://parl.ai/downloads/blended_skill_talk/ed_persona_topicifier__test__experiencer_only.json',
        'ed_persona_topicifier__test__experiencer_only.json',
        '2604e977787be0b5edc54561f7ce8a54c40758d235a3fee262fe20fe36b8cd15',
        zipped=False,
    ),
    build_data.DownloadableFile(
        'http://parl.ai/downloads/blended_skill_talk/safe_personas_2.txt',
        'safe_personas.txt',
        '2ee292aa0006ea002e9b23d4f7326fe9e17514ce5793d31fd8d679035d4366a7',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'blended_skill_talk')
    version = 'v1.4'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Format it for use with ParlAIDialogTeacher
        _create_parlai_format(dpath)

        # Mark the data as built
        build_data.mark_done(dpath, version_string=version)


def _create_parlai_format(dpath: str):
    """
    Copy data into the format read by ParlAIDialogTeacher.

    'text' will be from the free Turker, who speaks first, and 'label' will be from the
    guided Turker.
    """

    datatypes = ['train', 'valid', 'test']
    for datatype in datatypes:

        load_path = os.path.join(dpath, f'{datatype}.json')
        save_path = os.path.join(dpath, f'{datatype}.txt')

        print(f'Loading {load_path}.')
        with open(load_path, 'r', encoding='utf8') as f_read:
            data = json.load(f_read)

        print(f'Saving to {save_path}')
        with open(save_path, 'w', encoding='utf8') as f_write:
            for episode in data:
                assert (
                    len(episode['dialog'])
                    == len(episode['suggestions'])
                    == len(episode['chosen_suggestions'])
                )
                num_entries = len(episode['dialog']) // 2
                for entry_idx in range(num_entries):
                    line = _get_line(
                        episode=episode, num_entries=num_entries, entry_idx=entry_idx
                    )
                    f_write.write(f'{line} \n')


def _get_line(episode: dict, num_entries: int, entry_idx: int) -> str:
    """
    Return the line to print in the reformatted file.
    """
    episode_done = entry_idx == num_entries - 1

    # Compile original context
    if entry_idx == 0:
        # Add those pieces of context that appear in the datasets that this one was
        # based on. Specifically:
        # - Your persona, but not your partner's persona (from ConvAI2)
        # - Topic (from Wizard of Wikipedia)
        # - **Not** the situation (from EmpatheticDialogues)
        persona_pieces = [
            f"your persona: {episode['personas'][1][0]}",
            f"your persona: {episode['personas'][1][1]}",
        ]
        if episode['context_dataset'] == 'wizard_of_wikipedia':
            additional_context_pieces = [episode['additional_context']]
        else:
            additional_context_pieces = []
        previous_utterance_pieces = [
            episode['free_turker_utterance'],
            episode['guided_turker_utterance'],
        ]
        original_context = (
            '\n'.join(
                persona_pieces + additional_context_pieces + previous_utterance_pieces
            )
            + '\n'
        )
    else:
        original_context = ''

    # Gather messages and suggestions
    free_message = episode['dialog'][2 * entry_idx][1]
    guided_message = episode['dialog'][2 * entry_idx + 1][1]
    single_task_suggestions = {
        task: episode['suggestions'][2 * entry_idx + 1][task]
        for task in ['convai2', 'empathetic_dialogues', 'wizard_of_wikipedia']
    }
    guided_chosen_suggestion = episode['chosen_suggestions'][2 * entry_idx + 1]

    # Compile into text string
    parts = {
        'text': original_context + free_message,
        'labels': guided_message,
        'context_dataset': episode['context_dataset'],
        'free_message': free_message,
        **single_task_suggestions,
        'guided_chosen_suggestion': guided_chosen_suggestion,
    }
    assert all([isinstance(part, str) for part in parts.values()])
    line = '\t'.join([f'{key}:{_escape(value)}' for key, value in parts.items()])

    # Add episode_done
    if episode_done:
        line += '\tepisode_done:True'

    # Add label_candidates
    if 'label_candidates' in episode:
        label_candidates = episode['label_candidates'][entry_idx]
        # Note that episode['dialog'] is indexed by utterance (from either Turker) and
        # episode['label_candidates'] is indexed by guided Turker response
        assert all([isinstance(cand, str) for cand in label_candidates])
        escaped_label_candidates = [_escape(cand) for cand in label_candidates]
        line += '\tlabel_candidates:' + '|'.join(escaped_label_candidates)
    return line


def _escape(value: str) -> str:
    return value.replace('\t', '\\t').replace('\n', '\\n').replace('|', '__PIPE__')
