#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities + Functions.
"""
from collections import deque
import random
from typing import Dict

# DEFAULT_DELIM = '  '
DEFAULT_DELIM = '\n'

BEGIN_SPEAKER = '__speaker__'
END_SPEAKER = '__end_speaker__'
WHO_AM_I = '__whoami__'
WHO_ARE_YOU = '__whoareyou__'
WHO_IS_THIS = '__whoisthis__'
SELF = '__self__'
PARTNER = '__partner__'
END_CONTEXT = '__end_context__'

CONTEXT_KEYS = [
    '_setting_name',
    '_setting_desc',
    '_partner_name',
    '_self_name',
    '_self_persona',
    '_other_persona',
]


def flatten(episode, context_length, include_labels=True, delimiter='\n'):
    context = deque(maxlen=context_length if context_length > 0 else None)
    new_episode = []

    for ex in episode:
        context.append(ex.get('text', ''))
        # add context
        if len(context) > 1:
            ex.force_set('text', delimiter.join(context))
        # set episode_done to be True
        ex.force_set('episode_done', True)
        labels = ex.get('labels', ex.get('eval_labels', None))
        if labels is not None and include_labels:
            context.append(random.choice(labels))

        new_episode.append(ex)

    return new_episode


def extract_characters(context: str) -> Dict[str, str]:
    """
    Extract characters from context.

    :param context:
        context string

    :return characters:
        return the two characters within the conversation; in dict form,
        mapping _self_name --> self character; _partner_name --> partner character
    """
    name_lines = [
        l for l in context.split('\n') if '_partner_name' in l or '_self_name' in l
    ]
    characters = {n.split()[0]: ' '.join(n.split()[1:]) for n in name_lines}
    return characters


def maybe_annotate(
    character: str,
    text: str,
    annotate_speaker: bool,
    speaker_separator: bool,
    speaker_annotation_position: str,
) -> str:
    """
    Annotate text.

    Depending on setting of opt arg, either prepend or append.

    :param text:
        text to augment
    :param character:
        character to add
    :param annotate_speaker:
        whether to annotate the speaker
    :param speaker_separator:
        whether to incude speaker separator tokens
    :param speaker_annotation_position:
        where to annotate the speaker in the utterance

    :return text:
        return augmented text
    """
    if not annotate_speaker:
        return text
    speaker_text = character
    if speaker_separator:
        speaker_text = f'{BEGIN_SPEAKER} {character} {END_SPEAKER}'
    if speaker_annotation_position == 'prefix':
        return f'{speaker_text} {text}'
    else:
        return f'{text} {speaker_text}'
