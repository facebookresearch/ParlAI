#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy

from parlai.core.teachers import ParlAIDialogTeacher, FbDeprecatedDialogTeacher


class Parley(object):
    """A single example for training
    Args:
        context: (str) the dialog context in which the response was given;
            - at minimum, this is the immediately previous utterance
            - at maximum, this is the entire conversation up to that point
            - external knowledge/memories should be in memories
        response: (str) response
        reward: (int) reward
        candidates: (list) of strings
        memories: (list) of strings in order (if it matters)
    """

    def __init__(
        self,
        context,
        response='',
        reward=0,
        candidates=None,
        memories=None,
        episode_done=False,
        **kwargs,
    ):
        if candidates is None:
            candidates = []
        if memories is None:
            memories = []
        self.context = context
        self.response = response if response is not None else ''
        self.reward = reward
        self.candidates = candidates
        self.memories = memories
        self.episode_done = bool(episode_done)
        # NOTE: currently tossing any extra kwargs passed in

    def __repr__(self):
        return f"Parley({self.to_dict()})"

    def to_dict(self, include_empty=False):
        if include_empty:
            return {
                'context': self.context,
                'response': self.response,
                'reward': self.reward,
                'candidates': self.candidates,
                'memories': self.memories,
                'episode_done': self.episode_done,
            }
        else:
            pdict = {'context': self.context, 'response': self.response}
            if self.reward:
                pdict['reward'] = self.reward
            if self.candidates:
                pdict['candidates'] = self.candidates
            if self.memories:
                pdict['memories'] = self.memories
            if self.episode_done:
                pdict['episode_done'] = self.episode_done
            return pdict

    def to_parlai(self):
        string = f"context:{self.context}"
        string += f"\tresponse:{self.response}" if self.response else ''
        string += f"\treward:{self.reward}" if self.reward else ''
        string += f"\tcandidates:{'|'.join(self.candidates)}" if self.candidates else ''
        string += f"\tmemories:{'|'.join(self.candidates)}" if self.candidates else ''
        string += f"\tepisode_done:{self.episode_done}" if self.episode_done else ''
        return string.strip()

    def to_fb(self):
        pieces = [
            self.context,
            self.response,
            str(self.reward),
            '|'.join(self.candidates),
            '|'.join(self.memories),
        ]
        return '\t'.join(pieces).strip()


def sanitize_parley(parley):
    """
    Separate memories from context, pull out response, split context/memories lists.
    """
    if '\n' in parley.context:
        snippets = parley.context.split('\n')
        text = snippets[-1]
        mems = snippets[:-1]
        parley.context = text
        parley.memories = mems
    parley.response = parley.response[0]
    assert isinstance(parley.candidates, list)
    assert isinstance(parley.memories, list)
    return parley


def add_person_tokens(responses, first_speaker=None, last_speaker=1):
    """Converts a list of responses into a single tag-separated string
    Args:
        responses: list of responses (strings)
        first_speaker: either 1 or 2; the owner of the first response
        last_speaker: either 1 or 2; the owner of the last response
            NOTE: if first_speaker is provided, it overrides last_speaker
    Output:
        text: the concatenated text

    e.g.,
    responses = ["How are you?", "I'm doing fine!", "I'm glad to hear it!"]
    result = add_person_tokens(responses)
    result: "__p1__ How are you? __p2__ I'm doing fine! __p1__ I'm glad to
        hear it!"
    """
    if first_speaker is None:
        first_speaker = (last_speaker + len(responses)) % 2 + 1
    speaker = first_speaker
    text = ''
    for response in responses:
        tag = f"__p{speaker}__"
        text += ' ' + tag + ' ' + response
        speaker = 1 if speaker == 2 else 2
    return text.strip()


def extract_fb_episodes(datafile):
    opt = {'datatype': 'train', 'datafile': datafile}
    episode = None
    for parley in FbDeprecatedDialogTeacher(opt).setup_data(datafile):
        fields, is_new_episode = parley
        if is_new_episode:
            if episode is not None:
                yield episode
            episode = []
        raw_parley = Parley(*fields)
        parley = sanitize_parley(raw_parley)
        episode.append(parley)
    yield episode


def extract_parlai_episodes(datafile):
    opt = {
        'datatype': 'train',
        'datafile': datafile,  # is this necessary?
        'parlaidialogteacher_datafile': datafile,
    }
    episode = None
    for episode in ParlAIDialogTeacher(opt).episodes:
        episode = [Parley(**parley_dict) for parley_dict in episode]
        yield episode


def episode_to_examples(episode, histsz):
    """
    Converts an episode (list of Parleys) into self-feeding compatible examples.

    WARNING: we no longer require a histz when making a self-feeding file. Shortening of
    the history is typically done in the teacher file or in interactive mode.
    """
    examples = []
    history = []
    for parley in episode:
        # Update memories and history
        # memories.extend(parley.memories)
        history.append(parley.context)

        # Concatenate history and add speaker tokens as necessary
        # if history_size == 1, the bot (p2) only sees the immediately
        # preceding utterance (the prompt from the human, p1).
        if histsz < 0:
            utterances = history
            context = add_person_tokens(utterances, last_speaker=1)
        elif histsz == 0:
            context = '__null__'
        else:
            utterances = history[-histsz:]
            context = add_person_tokens(utterances, last_speaker=1)

        example = Parley(
            context,
            parley.response,
            parley.reward,
            copy.deepcopy(parley.candidates),
            # copy.deepcopy(memories),
        )
        examples.append(example)

        # Add current turn's response to the history
        history.append(parley.response)
    return examples
