#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import os
from typing import Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
import json

from parlai.core.message import Message
from parlai.core.metrics import F1Metric
from parlai.core.mutators import register_mutator, MessageMutator

from parlai.tasks.wizard_of_wikipedia.agents import GeneratorTeacher
from parlai.tasks.wizard_of_wikipedia.agents import (
    TOKEN_KNOWLEDGE,
    TOKEN_END_KNOWLEDGE,
)


class WizardOfWikipediaGeneratorTeacher(GeneratorTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        # Check if probabilistic checked sentence to input mutator is used.
        if (
            opt['mutators']
            and 'add_probabilistic_checked_sentence_to_input_wow' in opt['mutators']
        ):
            distractor_knowledge_fname = 'data/k2r/wow/distractor.txt'
            if not os.path.exists(distractor_knowledge_fname):
                # Collect all knowledge sentences.
                checked_sentences = defaultdict(set)
                for episode_idx in tqdm(
                    range(self.num_episodes()),
                    desc='Loading distractor knowledge',
                ):
                    entry_idx = 0
                    while True:
                        entry = self.get(episode_idx, entry_idx)
                        entry_idx += 1
                        checked_sentence = entry.get('checked_sentence', '')
                        chosen_topic = entry.get('chosen_topic', '')
                        if not checked_sentence or not chosen_topic:
                            continue
                        checked_sentences[chosen_topic].add(checked_sentence)
                        if entry['episode_done']:
                            break
                # Save it to file.
                checked_sentences = {k: list(v) for k, v in checked_sentences.items()}
                with open(distractor_knowledge_fname, 'w') as f:
                    json.dump(checked_sentences, f)
                print(
                    f'Saved distractor knowledge sentences to "{distractor_knowledge_fname}".'
                )

            # Add file path to opt.
            self.opt['distractor_knowledge_fname'] = distractor_knowledge_fname

            # Update mutator.
            self.mutators = [
                AddCheckedSentence(self.opt)
                if isinstance(mutator, AddCheckedSentence)
                else mutator
                for mutator in self.mutators
            ]

    def getID(self):
        name = super().getID()
        if 'mutators' in self.opt and self.opt['mutators']:
            return name + '__' + self.opt['mutators']
        return name

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ):
        super().custom_evaluation(teacher_action, labels, model_response)
        if 'knowledge_response' in model_response:
            self.metrics.add(
                'predicted_knowledge_f1',
                F1Metric.compute(
                    model_response['knowledge_response'],
                    [model_response['text']],
                ),
            )
            self.metrics.add(
                'knowledge_response_f1',
                F1Metric.compute(
                    model_response['knowledge_response'],
                    [teacher_action['checked_sentence']],
                ),
            )


@register_mutator("add_probabilistic_checked_sentence_to_input_wow")
class AddCheckedSentence(MessageMutator):
    """
    Adds the checked sentence to the end of the text.

    But with probability p, it picks a wrong one. It adds the round(p*10) to the input
    as conditioning.
    """

    def __init__(self, opt):
        super().__init__(opt)
        if 'distractor_knowledge_fname' in opt:
            # Load distractor checked sentences.
            with open(opt['distractor_knowledge_fname'], 'r') as f:
                self.distractor_knowledge_sentences = json.load(f)

    @property
    def checked_sentence_kword(self):
        return 'checked_sentence'

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        if 'text' not in message:
            return message
        text = new_message.pop('text')
        checked_sentence = new_message.get(self.checked_sentence_kword, '')
        if isinstance(checked_sentence, list):
            checked_sentence = ' '.join(checked_sentence)
        chosen_topic = message['chosen_topic']

        # Get probability of adding wrong knowledge.
        p = random.random()
        if chosen_topic not in self.distractor_knowledge_sentences:
            print(f'Chosen topic "{chosen_topic}" does not have distractor sentences.')
            p = 1.0
        if random.random() > p:
            # Replace the knowledge with incorrect one.
            distractors = list(
                set(self.distractor_knowledge_sentences[chosen_topic])
                - set([checked_sentence])
            )
            if distractors:
                checked_sentence = random.choice(distractors)
            else:
                print(
                    f'Chosen topic "{chosen_topic}" does not have distractor sentences.'
                )
                p = 1.0

        confidence = round(p * 10)
        text += f'\n{TOKEN_KNOWLEDGE} {confidence}: {checked_sentence} {TOKEN_END_KNOWLEDGE}'
        new_message['text'] = text

        return new_message
