#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import nltk
import os
import random
import copy
from typing import List, Optional, Tuple

from parlai.core.agents import create_agent_from_model_file, Agent
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.core.metrics import F1Metric
from parlai.core.mutators import register_mutator, MessageMutator, ManyEpisodeMutator
from parlai.core.opt import Opt
import parlai.core.tod.tod_core as tod
from parlai.tasks.blended_skill_talk.build import build as build_bst
import parlai.tasks.wizard_of_internet.constants as CONST

from parlai.utils.strings import normalize_reply

from projects.seeker.utils import calc_f1_msc, IS_SEARCH_REQUIRED
from projects.seeker.tasks.mutators import (
    PromptSearchQueryMutator,
    ExtractEntityResponse,
)

import projects.bb3.constants as BB3_CONST
import projects.bb3.prompts as PROMPT


###################
# Prompt Mutators #
###################


@register_mutator('prompt_knowledge_mutator')
class PromptKnowledgeMutator(PromptSearchQueryMutator):
    """
    Add a __knowledge__ prompt to the end of the context, to inform the model.

    Assumes flattened data.
    """

    PROMPT = BB3_CONST.GENERATE_KNOWLEDGE


@register_mutator('prompt_memory_mutator')
class PromptMemoryMutator(PromptSearchQueryMutator):
    """
    Add a __gnerate-memory__ prompt to the end of the context, to inform the model.

    Assumes flattened data.
    """

    PROMPT = BB3_CONST.GENERATE_MEMORY


@register_mutator('prompt_extract_entity_mutator')
class PromptEntityExtractionMutator(PromptSearchQueryMutator):
    """
    Add a __extract-entity__ prompt to the end of the context, to inform the model.

    Assumes flattened data.
    """

    PROMPT = BB3_CONST.EXTRACT_ENTITY


@register_mutator('prompt_memory_decision_mutator')
class PromptMemoryDecisionMutator(PromptSearchQueryMutator):
    """
    Add a __is-memory-required__ prompt to the end of the context, to inform the model.

    Assumes flattened data.
    """

    PROMPT = BB3_CONST.IS_MEMORY_REQUIRED


#############################
# Memory Knowledge Mutators #
#############################


SUMMARIZER = None


ALL_PERSONAS = None


def build_summarizer(opt: Opt) -> Agent:
    """
    Build the Persona Summarizer.
    """
    return create_agent_from_model_file(
        modelzoo_path(opt['datapath', 'zoo:bb3/persona_summarizer/model']),
        opt_overrides={
            'skip_generation': False,
            'inference': 'beam',
            'beam_size': 3,
            'beam_block_ngram': 3,
            'beam_min_length': 10,
            'datatype': 'valid',
        },
    )


def build_all_personas(opt: Opt) -> List[str]:
    """
    Build the personas list from which we sample memories for MDM.
    """
    personas_path = os.path.join(
        opt['datapath'], 'blended_skill_talk', 'persona_list.txt'
    )
    if not os.path.exists(personas_path):
        new_opt = copy.deepcopy(opt)
        new_opt['task'] = 'blended_skill_talk'
        build_bst(new_opt)
    with open(personas_path) as f:
        all_personas = [l.replace('||', '').replace('\n', '') for l in f.readlines()][
            :-1
        ]
    return all_personas


def merge_personas(personas: List[str], p_to_merge: List[List[str]]) -> List[str]:
    """
    Merge two groups of personas, based on word overlap.

    :param personas:
        list of original personas.
    :param p_to_merge:
        list of personas to merge.
        first element is partner personas, second is your personas.

    :return merged:
        return list of merged personas
    """
    new_personas = []
    split_personas = [
        [p.replace('\n', ' ') for p in personas if p.startswith('partner')],
        [p.replace('\n', ' ') for p in personas if p.startswith('your')],
    ]
    for i in range(2):
        prefix = "partner's persona: " if i == 0 else "your persona: "
        personas_i = split_personas[i]
        for ip in p_to_merge[i]:
            found = False
            for j, op in enumerate(personas_i):
                if F1Metric.compute(ip, [op[op.index(':') + 2 :]]) > 0.5:
                    personas_i[j] = ' '.join([op, ip])
                    found = True
                    break
            if not found:
                personas_i.append(f"{prefix}{ip}")
        new_personas.append(personas_i)
    return new_personas[0] + new_personas[1]


def get_overlap_sentence(
    full_text: str, label: str, docs: List[str], find_speaker: bool = True
) -> Tuple[str, int, Optional[int]]:
    """
    Get the sentence that most overlaps with a sentence in the context.

    :param full_text:
        full context
    :param label:
        target to find overlap with
    :param docs:
        list of candidate strings for overlap
    :param find_speaker:
        whether to determine who said the sentence in the full text

    :return (best_sentence, best_f1, best_idx):
        if we reach the F1 threshold, return the
        corresponding sentence, F1 score, and index into the docs
    """
    best_f1 = 0
    best_sentence = ''
    best_idx = None
    try:
        gold_parts = nltk.word_tokenize(label)
    except IndexError:
        return best_sentence, best_f1, best_idx
    for i, d in enumerate(docs):
        ds = d.split('\n')
        for s in ds:
            f1 = calc_f1_msc(s, gold_parts)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = s
                best_idx = i
    if best_sentence != '' and find_speaker:
        # find which speaker it is
        z = full_text.split('\n')
        z.reverse()
        ind = z.index(best_sentence)
        if (ind % 2) == 0:
            speaker = '__them__'
        else:
            speaker = '__you__'
        if 'your persona:' in best_sentence:
            speaker = '__you__'

        best_sentence = f"{best_sentence} {speaker}"

    return best_sentence, best_f1, best_idx


@register_mutator("personas_as_docs")
class PersonasAsDocsMutator(ManyEpisodeMutator):
    """
    This mutator does the following:

    1. Computes memories for context lines with the persona summarizer
    2. Determines which memory has highest overlap with target label
    3. Depending on target, does the following:
        - if knowledge, we set the target as that persona memory
        - if memory_decision, set the target as access_memory if enough overlap, otherwise don't
    """

    THRESHOLD = 0.3
    TARGET = 'knowledge'

    def __init__(self, opt: Opt) -> None:
        super().__init__(opt)
        assert 'flatten' not in opt['mutators'], 'cannot use flatten with this mutator.'
        global SUMMARIZER
        if SUMMARIZER is None:
            SUMMARIZER = build_summarizer(opt)
        global ALL_PERSONAS
        if ALL_PERSONAS is None:
            ALL_PERSONAS = build_all_personas(opt)

    def compute_context_memories(
        self, episode: List[Message]
    ) -> Tuple[List[str], List[str]]:
        """
        Compute the memories from the context.

        :param episode:
            list of messages

        :return (self_memories, partner_memories):
            return list of memories for self and partner.
        """
        partner_texts = [m['text'] for m in episode]
        first_text = [
            t for t in partner_texts[0].split('\n') if not t.startswith('your persona:')
        ]
        if not first_text:
            partner_texts[0] = BB3_CONST.DUMMY_TEXT
        else:
            partner_texts[0] = first_text[0]
        self_texts = [m.get('labels', m.get('eval_labels'))[0] for m in episode]

        assert SUMMARIZER is not None
        partner_memories = SUMMARIZER.batch_respond(
            [
                Message({'text': p.split('\n')[-1], 'episode_done': True})
                for p in partner_texts
            ]
        )
        self_memories = SUMMARIZER.batch_respond(
            [Message({'text': s, 'episode_done': True}) for s in self_texts]
        )

        return self_memories, partner_memories

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        assert ALL_PERSONAS is not None
        global NUM_DONE

        def _maybe_add_persona(text: str, is_partner: bool):
            prefix = "partner's persona: " if is_partner else "your persona: "
            if text != BB3_CONST.NOPERSONA:
                raw_personas.append(text)
                all_personas.append(f"{prefix}{text}")

        # compute the personas
        self_memories, partner_memories = self.compute_context_memories(episode)

        # find best overlaps
        all_personas = []
        raw_personas = []
        new_episode = []
        all_text = []
        for i, msg in enumerate(episode):
            text = msg['text']
            label = msg['labels'][0]
            if i == 0:
                if 'persona:' in text:
                    # Convai2; session 1
                    personas = [
                        t for t in text.split('\n') if t.startswith('your persona:')
                    ]
                    text = [
                        t for t in text.split('\n') if not t.startswith('your persona:')
                    ]
                    if not text:
                        text = BB3_CONST.DUMMY_TEXT
                        msg.force_set('text', text)
                    else:
                        text = text[0]
                    all_personas += personas
                    raw_personas += [
                        p.replace('your persona: ', '').replace(
                            "partner's persona: ", ''
                        )
                        for p in all_personas
                    ]
                elif 'personas' in msg:
                    # MSC
                    all_personas += msg['personas'].split('\n')
                    init_personas = msg['init_personas']
                    all_personas = merge_personas(all_personas, init_personas)
                    raw_personas = [
                        p.replace('your persona: ', '').replace(
                            "partner's persona: ", ''
                        )
                        for p in all_personas
                    ]
            assert (
                'persona:' not in text
            ), "cannot use this mutator with teacher that puts personas in context"
            all_text.append(text)
            best_sentence, best_f1, best_idx = get_overlap_sentence(
                '\n'.join(all_text), msg['labels'][0], raw_personas, False
            )
            if best_f1 > self.THRESHOLD:
                assert best_idx is not None
                if self.TARGET == 'knowledge':
                    msg.force_set(CONST.RETRIEVED_DOCS, all_personas)
                    msg.force_set(CONST.RETRIEVED_DOCS_URLS, [''] * len(all_personas))
                    msg.force_set(CONST.RETRIEVED_DOCS_TITLES, [''] * len(all_personas))
                    msg.force_set(CONST.SELECTED_SENTENCES, [all_personas[best_idx]])
                    msg.force_set('raw_personas', raw_personas)
                    msg.force_set('old_target', msg['labels'])
                    msg.force_set('labels', [best_sentence])
                    msg.force_set('text', '\n'.join(all_text))
                elif self.TARGET == 'memory_decision':
                    msg.pop('personas', None)
                    msg.pop('init_personas', None)
                    msg.force_set('persona_sentence', best_sentence)
                    msg.force_set('old_target', msg['labels'])
                    msg.force_set('labels', [BB3_CONST.DO_ACCESS_MEMORY])
                    # make sure to include memory in the context!
                    if not best_sentence.startswith('persona:'):
                        best_sentence = f"persona: {best_sentence}"
                    msg.force_set(
                        'text', "\n".join([best_sentence, all_text[-1].split('\n')[-1]])
                    )
                new_episode.append([msg])
            elif self.TARGET == 'memory_decision':
                msg.pop('personas', None)
                msg.pop('init_personas', None)
                msg.force_set('persona_sentence', best_sentence)
                msg.force_set('old_target', msg['labels'])
                msg.force_set('labels', [BB3_CONST.DONT_ACCESS_MEMORY])
                # include a random memory in context!
                random_pers = (
                    random.choice(ALL_PERSONAS)
                    .replace('your persona:', 'persona:')
                    .replace("partner's persona:", 'persona:')
                )
                if not random_pers.startswith('persona:'):
                    random_pers = f"persona: {random_pers}"
                msg.force_set(
                    'text', "\n".join([random_pers, all_text[-1].split('\n')[-1]])
                )
                new_episode.append([msg])
            _maybe_add_persona(partner_memories[i], True)
            _maybe_add_persona(self_memories[i], False)
            all_text.append(label)

        return new_episode


@register_mutator("convert_overlap_to_personas_as_docs")
class ConvertOverlapToPersonasAsDocs(ManyEpisodeMutator):
    """
    This mutator converts a knowledge task that originally computed word overlap between
    target and all prior sentences to choose a previous utterance as the knowledge
    sentence.

    The conversion turns all context utterances into persona memories (retrieved docs),
    and all targets into persona memories as well.
    """

    TARGET = 'knowledge'

    def __init__(self, opt: Opt) -> None:
        super().__init__(opt)
        global SUMMARIZER
        if SUMMARIZER is None:
            SUMMARIZER = build_summarizer(opt)

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        assert len(episode) == 1
        message = episode[0]
        texts = message['text'].split('\n')
        texts[-1] = texts[-1].replace(f' {BB3_CONST.GENERATE_KNOWLEDGE}', '')
        all_personas = []
        if 'persona:' in message['text']:
            # Convai2; session 1
            personas = [t for t in texts if t.startswith('your persona:')]
            new_text = [t for t in texts if not t.startswith('your persona:')]
            if not new_text:
                new_text = BB3_CONST.DUMMY_TEXT
            message.force_set('text', '\n'.join(new_text))
            all_personas += personas
        elif 'personas' in message:
            # MSC
            all_personas += message['personas'].split('\n')
        assert SUMMARIZER is not None
        required_summaries = [
            Message(
                {
                    'text': t.replace(f' {BB3_CONST.GENERATE_KNOWLEDGE}', ''),
                    'episode_done': True,
                }
            )
            for t in texts
        ] + [
            Message(
                {
                    'text': message['labels'][0]
                    .replace(f' {BB3_CONST.YOU}', '')
                    .replace(f' {BB3_CONST.THEM}', '')
                }
            )
        ]
        summaries = []
        offset = 16
        for start_idx in range(0, len(required_summaries), offset):
            summaries += SUMMARIZER.batch_respond(
                required_summaries[start_idx : start_idx + offset]
            )
        self_summaries = [
            t
            for i, t in enumerate(reversed(summaries[:-1]))
            if i % 2 == 1 and BB3_CONST.NOPERSONA not in t
        ]
        partner_summaries = [
            t
            for i, t in enumerate(reversed(summaries[:-1]))
            if i % 2 == 0 and BB3_CONST.NOPERSONA not in t
        ]
        all_personas = merge_personas(all_personas, [partner_summaries, self_summaries])
        if summaries[-1] != BB3_CONST.NOPERSONA or 'persona:' in message['labels'][0]:
            # for Convai2 we might already have this.
            new_target = (
                message['labels'][0]
                if 'persona:' in message['labels'][0]
                else summaries[-1]
            )
            if 'persona:' not in new_target:
                prefix = (
                    "your" if BB3_CONST.YOU in message['labels'][0] else "partner's"
                )
                new_target = f"{prefix} persona: {new_target}"
            message.force_set('original_target', message['labels'])
            message.force_set(CONST.RETRIEVED_DOCS, all_personas)
            message.force_set(CONST.RETRIEVED_DOCS_URLS, [''] * len(all_personas))
            message.force_set(CONST.RETRIEVED_DOCS_TITLES, [''] * len(all_personas))
            message.force_set(CONST.SELECTED_SENTENCES, [new_target])
            message.force_set('labels', [new_target])
            message.force_set(
                'text',
                message['text'].replace(
                    BB3_CONST.GENERATE_KNOWLEDGE, BB3_CONST.ACCESS_MEMORY
                ),
            )
            message.pop('skip_retrieval', None)
            return [[message]]
        else:
            return []


@register_mutator('prompt_access_memory_mutator')
class PromptAccessMemoryMutator(PromptSearchQueryMutator):
    """
    Add a __access-memory__ prompt to the end of the context, to inform the model.

    Assumes flattened data.
    """

    PROMPT = BB3_CONST.ACCESS_MEMORY


@register_mutator("match_target_to_persona_mutator")
class MatchTargetToPersonaMutator(MessageMutator):
    """
    For ad-hoc conversion; matches the knowledge label to a "your" or "partner's"
    prefix.
    """

    def message_mutation(self, message: Message) -> Message:
        personas = message[CONST.RETRIEVED_DOCS]
        target = message['labels'][0]
        p = None
        for p in personas:
            if target in p:
                break
        assert p
        assert target in p
        message.force_set('labels', [p])
        return message


@register_mutator("fix_mkm_formatting_mutator")
class FixMkmFormattingMutator(MessageMutator):
    """
    Ad-hoc mutator for the utterance overlap Mkm teachers.

    With ConvAI2 and BST, the utterance overlap mutators found persona sentences from the context.

    But those examples were not formatted correctly with `convert_overlap_to_personas_as_docs`

    So, fixing that now
    """

    def message_mutation(self, message: Message) -> Message:
        text = message['text']
        if not text.endswith(BB3_CONST.ACCESS_MEMORY):
            message.force_set('text', f"{text} {BB3_CONST.ACCESS_MEMORY}")
        for tok in [BB3_CONST.YOU, BB3_CONST.THEM]:
            if tok in message['labels'][0]:
                message.force_set(
                    'labels', [message['labels'][0].replace(f' {tok}', '')]
                )
                assert tok not in message['labels'][0]
        return message


############################
# Memory Decision Mutators #
############################
@register_mutator("memory_decision_mutator")
class MemoryDecisionMutator(PersonasAsDocsMutator):
    """
    Utilize mutator above to set the memory decision.
    """

    TARGET = 'memory_decision'


############################
# Style Grounding Mutators #
############################
@register_mutator("style_gen_to_grm")
class StyleGenToVRMMutator(MessageMutator):
    """
    Converts style gen tasks to grounded dialogue tasks.

    assumes flattened data
    """

    def message_mutation(self, message: Message) -> Message:
        text = message['text']
        style = message['personality']
        assert BB3_CONST.BEGIN_STYLE not in text
        message.force_set(
            'text', f"{text}\n{BB3_CONST.BEGIN_STYLE} {style} {BB3_CONST.END_STYLE}"
        )
        return message


############################
# Search Dialogue Mutators #
############################
@register_mutator("funpedia_to_bb3_mutator")
class FunpediaToBB3WithStyleMutator(MessageMutator):
    def message_mutation(self, message: Message) -> Message:
        context = message['text'].split('\n')
        assert len(context) == 3
        topic, style, knowledge = context
        message.force_set(
            'text',
            f"{topic}\n{BB3_CONST.TOKEN_KNOWLEDGE} {knowledge} {BB3_CONST.TOKEN_END_KNOWLEDGE}"
            f"\n{BB3_CONST.BEGIN_STYLE} {style} {BB3_CONST.END_STYLE}",
        )
        return message


@register_mutator("tod_to_srm_mutator")
class TodToSRMMutator(ManyEpisodeMutator):
    """
    Converts TOD Tasks to a Search Dialogue Response task.

    System grounding on the API responses.

    Flattens the episodes as well
    """

    def many_episode_mutation(self, episode):
        out_episodes = []
        context = []
        knowledge: List = []
        system_silence = f"{tod.STANDARD_SYSTEM_UTTERANCE}{BB3_CONST.DUMMY_TEXT}"
        user_silence = f"{tod.STANDARD_USER_UTTERANCE}{BB3_CONST.DUMMY_TEXT}"
        for m in episode:
            new_message = m.copy()
            user_utt: str = m['text']
            system_utt: str = m['labels'][0]
            system_utt_clean = system_utt.replace(
                tod.STANDARD_SYSTEM_UTTERANCE, ''
            ).replace('\n', ' ')
            if user_utt.startswith(tod.STANDARD_API_SCHEMAS):
                # start of convo; don't need this
                continue
            elif (
                user_utt.startswith(tod.STANDARD_USER_UTTERANCE)
                and user_utt != user_silence
            ):
                # add user utterances to the conversation
                context.append(user_utt.replace(tod.STANDARD_USER_UTTERANCE, ''))
            if user_utt.startswith(tod.STANDARD_RESP) and user_utt != tod.STANDARD_RESP:
                # here's the knowledge grounding
                knowledge.append(
                    user_utt.replace(tod.STANDARD_RESP, '').replace('\n', '')
                )

            if knowledge and not (
                system_utt == system_silence or system_utt.startswith(tod.STANDARD_CALL)
            ):
                str_knowledge = f"{BB3_CONST.TOKEN_KNOWLEDGE} {'. '.join(knowledge)} {BB3_CONST.TOKEN_END_KNOWLEDGE}"
                new_message.force_set('text', '\n'.join(context + [str_knowledge]))
                new_message.force_set('labels', [system_utt_clean])
                if F1Metric.compute(system_utt_clean, [str_knowledge]) > 0.0:
                    # filter out pathological examples
                    out_episodes.append([new_message])
                # reset knowledge
                knowledge = []
            if (
                not system_utt.startswith(tod.STANDARD_CALL)
                and system_utt != system_silence
            ):
                context.append(system_utt_clean)

        return out_episodes


############################
# Memory Dialogue Mutators #
############################
@register_mutator("convert_mkm_to_mrm_mutator")
class ConvertMkmToMrmMutator(MessageMutator):
    """
    Converts mkm task to mrm task.
    """

    def message_mutation(self, message: Message) -> Message:
        persona_knowledge = message['labels'][0]
        message.force_set(
            'text',
            f"{message['text'].replace(f' {BB3_CONST.ACCESS_MEMORY}', '')}\n{BB3_CONST.BEGIN_MEMORY} {persona_knowledge} {BB3_CONST.END_MEMORY}",
        )
        old_target = message['old_target']
        if not isinstance(old_target, list):
            old_target = [old_target]
        message.force_set('labels', old_target)
        return message


############################
# Entity Dailogue Mutators #
############################
@register_mutator("extract_entity_for_response_model_bb3")
class ExtractEntityResponseBB3(ExtractEntityResponse):
    BEGIN_KNOWLEDGE: str = BB3_CONST.BEGIN_ENTITY
    END_KNOWLEDGE: str = BB3_CONST.END_ENTITY


###################
# Filter Mutators #
###################


@register_mutator("filter_silence_only_mutator")
class FilterSilenceOnlyMutator(ManyEpisodeMutator):
    """
    Filters out episodes that only contain silence.
    """

    def many_episode_mutation(self, episode):
        for m in episode:
            text = m['text']
            for tok in [
                IS_SEARCH_REQUIRED,
                BB3_CONST.IS_MEMORY_REQUIRED,
                BB3_CONST.ACCESS_MEMORY,
            ]:
                text = text.replace(f' {tok}', '')
            if text.lower() == '__silence__':
                return []
        return [episode]


@register_mutator("filter_silence_only_memory_decision_mutator")
class FilterSilenceOnlyMemoryDecisionMutator(ManyEpisodeMutator):
    """
    Filters out episodes that only contain silence.
    """

    def many_episode_mutation(self, episode):
        for m in episode:
            text = m['text'].split('\n')
            if len(text) == 2 and text[1].lower() == '__silence__':
                return []
        return [episode]


@register_mutator("filter_wow_topic_only_search_decision_mutator")
class FilterWowTopicOnlySearchDecisionMutator(ManyEpisodeMutator):
    """
    Filters out episodes from the search decision teacher that are only the chosen
    topic.
    """

    def many_episode_mutation(self, episode):
        for m in episode:
            text = m['text'].replace(f' {IS_SEARCH_REQUIRED}', '')
            if 'chosen_topic' in m and text == m['chosen_topic']:
                return []
            elif len(text.split()) < 3:
                return []
        return [episode]


#######################
# Formatting Mutators #
#######################
@register_mutator("ensure_same_number_docs_and_titles_mutator")
class EnsureSameNumberDocsAndTitles(MessageMutator):
    """
    Ensures that docs and doc titles have same number of items.
    """

    def message_mutation(self, message: Message) -> Message:
        docs = message[CONST.RETRIEVED_DOCS]
        for key in [CONST.RETRIEVED_DOCS_URLS, CONST.RETRIEVED_DOCS_TITLES]:
            if len(message[key]) < len(docs):
                message.force_set(
                    key, message[key] + [''] * (len(docs) - len(message[key]))
                )
            elif len(message[key]) > len(docs):
                message.force_set(key, message[key][: len(docs)])
        return message


def _normalize_persona_line(x: str) -> str:
    """
    Normalize a persona line.
    """
    if x.startswith('your persona: '):
        # Normalize the sentence appearing after 'your persona:'
        x = x[len('your persona: ') :]
        x = normalize_reply(x)
        x = 'your persona: ' + x
    elif x.startswith("partner's persona: "):
        x = x[len("partner's persona: ") :]
        x = normalize_reply(x)
        x = "partner's persona: " + x
    return x


@register_mutator("normalize_reply_mutator")
class NormalizeReplyMutator(MessageMutator):
    """
    Uses string normalization over text and labels.

    And retrieved docs, I suppose...
    """

    def message_mutation(self, message: Message) -> Message:
        """
        Need to normalize:

        1) text 2) label 3) retrieved docs 4) best sentence.
        """
        # 1. text
        texts = message['text'].split('\n')
        your_personas = []
        partner_personas = []
        non_personas = []
        for i, x in enumerate(texts):
            if x.startswith('your persona: '):
                # Normalize the sentence appearing after 'your persona:'
                x = _normalize_persona_line(x)
                your_personas.append(x)
            elif x.startswith("partner's persona: "):
                x = _normalize_persona_line(x)
                partner_personas.append(x)
            elif i == len(texts) - 1:
                # check for memory decision, memory, generate memory etc.
                if BB3_CONST.IS_MEMORY_REQUIRED in x:
                    x = x.replace(f" {BB3_CONST.IS_MEMORY_REQUIRED}", '')
                    x = normalize_reply(x)
                    x = f"{x} {BB3_CONST.IS_MEMORY_REQUIRED}"
                    non_personas.append(x)
                elif BB3_CONST.ACCESS_MEMORY in x:
                    x = x.replace(f" {BB3_CONST.ACCESS_MEMORY}", '')
                    x = normalize_reply(x)
                    x = f"{x} {BB3_CONST.ACCESS_MEMORY}"
                    non_personas.append(x)
                elif BB3_CONST.BEGIN_MEMORY in x:
                    x = x.replace(f'{BB3_CONST.BEGIN_MEMORY} ', '').replace(
                        f' {BB3_CONST.END_MEMORY}', ''
                    )
                    if 'persona:' in x:
                        x = _normalize_persona_line(x)
                    else:
                        x = normalize_reply(x)
                    x = f"{BB3_CONST.BEGIN_MEMORY} {x} {BB3_CONST.END_MEMORY}"
                    non_personas.append(x)
                else:
                    x = normalize_reply(x)
                    non_personas.append(x)
            else:
                x = normalize_reply(x)
                non_personas.append(x)
        message.force_set(
            'text', '\n'.join(your_personas + partner_personas + non_personas)
        )
        # 2. label
        label = message['labels'][0]
        if 'persona:' in label:
            label = _normalize_persona_line(label)
        elif label not in [BB3_CONST.DONT_ACCESS_MEMORY, BB3_CONST.DO_ACCESS_MEMORY]:
            label = normalize_reply(label)
        message.force_set('labels', [label])

        # 3. retrieved docs.
        if CONST.RETRIEVED_DOCS in message:
            documents = message[CONST.RETRIEVED_DOCS]
            new_docs = []
            for x in documents:
                x = _normalize_persona_line(x)
                new_docs.append(x)
            message.force_set(CONST.RETRIEVED_DOCS, new_docs)
        # 4. best sentence
        if CONST.SELECTED_SENTENCES in message:
            selected = message[CONST.SELECTED_SENTENCES][0]
            message.force_set(
                CONST.SELECTED_SENTENCES, [_normalize_persona_line(selected)]
            )
        return message


@register_mutator("fits_pop_keys_mutator")
class FitsPopKeysMutator(MessageMutator):
    def message_mutation(self, message: Message) -> Message:
        for key in ['bot_acts', 'bot_gold', 'human_acts']:
            message.pop(key, None)
        return message


@register_mutator("fits_remove_special_toks_mutator")
class FitsRemoveSpecialToksMutator(MessageMutator):
    def message_mutation(self, message: Message) -> Message:
        from parlai.tasks.fits.mutators import GoldKnowledge

        message.force_set(
            'labels',
            [message['labels'][0].replace(f"{GoldKnowledge.TOKEN_DEC_KNOWLEDGE} ", '')],
        )
        message.force_set(
            'text', message['text'].replace(f" {GoldKnowledge.TOKEN_ENC_KNOWLEDGE}", '')
        )
        return message


########################
# OPT Teacher Mutators #
########################
@register_mutator("prefix_speakers")
class PrefixSpeakersMutator(MessageMutator):
    """
    Add control tokens to the speakers, within the context.
    """

    SELF_PREFIX = BB3_CONST.YOU_PREFIX
    PARTNER_PREFIX = BB3_CONST.PARTNER_PREFIX

    def message_mutation(self, message: Message) -> Message:
        context, utterances, post_utterances = self.get_context_and_utterances(message)
        context = [c.lstrip(' ').rstrip(' ') for c in context]
        utterances = [u.lstrip(' ').rstrip(' ') for u in utterances]
        post_utterances = [p.lstrip(' ').rstrip(' ') for p in post_utterances]

        new_utts: List[str] = []
        for i, utt in enumerate(reversed(utterances)):
            if utt.lower() != BB3_CONST.DUMMY_TEXT.lower() and not any(
                utt.startswith(tok)
                for tok in [
                    BB3_CONST.BEGIN_ENTITY,
                    BB3_CONST.BEGIN_MEMORY,
                    BB3_CONST.TOKEN_KNOWLEDGE,
                ]
            ):
                prefix = self.SELF_PREFIX if i % 2 == 1 else self.PARTNER_PREFIX
                utt = f"{prefix}{utt.lstrip(' ').rstrip(' ')}"
            new_utts.append(utt)
        for i in range(len(context)):
            # LIGHT context starts with underscores
            if 'persona:' not in context[i] and not context[i].startswith('_'):
                context[i] = f"{self.PARTNER_PREFIX}{context[i]}"
        message.force_set(
            'text', '\n'.join(context + list(reversed(new_utts)) + post_utterances)
        )
        return message

    def get_context_and_utterances(
        self, message: Message
    ) -> Tuple[List[str], List[str], List[str]]:
        context, utterances, post_utterances = [], [], []
        texts = message['text'].split('\n')
        m_id = message.get('id', '')
        # 1. Handle WizInt
        context_end = 0
        if ('Dialogue' in m_id and 'Vanilla' not in m_id) or (
            'msc_dialogue' in m_id
            and any(
                tok in message['text']
                for tok in [
                    BB3_CONST.BEGIN_MEMORY,
                    BB3_CONST.BEGIN_ENTITY,
                    BB3_CONST.TOKEN_KNOWLEDGE,
                ]
            )
        ):
            post_utterances = [texts[-1]]
            texts = texts[:-1]
        if 'Light' in m_id:
            context_end = 5
        elif 'Funpedia' in m_id:
            post_utterances.append(texts[-1])
            texts = texts[:-1]
        elif any(
            m_id.endswith(f'Woi{t}Teacher')
            for t in [
                'SearchDialogue',
                'VanillaDialogue',
                'SearchKnowledge',
                'SearchQuery',
            ]
        ):
            context_end = 2
        elif m_id.endswith('MemoryDecisionTeacher') or (
            'msc' in m_id and texts[0].startswith('persona:')
        ):
            context_end = 1
        elif any(
            m_id.endswith(f'Wow{t}Teacher')
            for t in ['SearchDialogue', 'VanillaDialogue', 'SearchKnowledge']
        ):
            context_end = 1
        elif any(
            m_id.endswith(f'BST{t}Teacher')
            for t in [
                'EntityDialogue',
                'VanillaDialogue',
                'StyleGroundingDialogue',
                'EntityKnowledge',
                'MemoryKnowledge',
                'MemoryKnowledgeUttOverlap',
            ]
        ):
            ctxt_dataset = message['context_dataset']
            if (
                any(tok in m_id for tok in ['MemoryKnowledge', 'MemoryDialogue'])
                and ctxt_dataset == 'wizard_of_wikipedia'
            ):
                context_end = 1
            elif any(
                t in m_id
                for t in [
                    'Knowledge',
                    'Dialogue',
                    'VanillaDialogue',
                    'StyleGroundingDialogue',
                ]
            ) and any('persona:' in t for t in texts):
                non_context = [i for i, t in enumerate(texts) if 'persona:' not in t]
                end_idx = 0
                if ctxt_dataset == 'wizard_of_wikipedia':
                    end_idx = 1
                context_end = (
                    non_context[end_idx] if len(non_context) > end_idx else len(texts)
                )
        elif any(
            m_id.endswith(f'Convai2{t}Teacher')
            for t in [
                'EntityDialogue',
                'EntityKnowledge',
                'VanillaDialogue',
                'StyleGroundingDialogue',
            ]
        ):
            non_context = [i for i, t in enumerate(texts) if 'persona:' not in t]
            context_end = non_context[0] if non_context else len(texts)
            if 'StyleGrounding' in m_id:
                # this has the topic as well
                context_end += 1

        context = texts[:context_end]
        utterances = texts[context_end:] if len(texts) > context_end else []
        return context, utterances, post_utterances


@register_mutator("prefix_speakers_opt")
class PrefixSpeakersOPTMutator(PrefixSpeakersMutator):
    SELF_PREFIX = f"{PROMPT.SELF_PREFIX}: "
    PARTNER_PREFIX = f"{PROMPT.PARTNER_PREFIX}: "

    def message_mutation(self, message: Message) -> Message:
        message = super().message_mutation(message)
        m_id = message.get('id', '')
        if (
            'dialogue' in m_id.lower()
            and not ('NQOpen' in m_id)
            and not (
                ('msc' in m_id and message['text'].startswith('persona:'))
                or ('msc' in m_id and BB3_CONST.EXTRACT_ENTITY in message['text'])
                or ('msc' in m_id and message['text'].endswith(BB3_CONST.ACCESS_MEMORY))
                or (
                    'msc' in m_id
                    and message['text'].endswith(BB3_CONST.IS_SEARCH_REQUIRED)
                )
                or (
                    'msc' in m_id
                    and message['text'].endswith(BB3_CONST.IS_MEMORY_REQUIRED)
                )
            )
        ):
            if message['labels'][0].startswith(PROMPT.SELF_PREFIX):
                message.force_set(
                    'labels',
                    [message['labels'][0].replace(f"{PROMPT.SELF_PREFIX}: ", '')],
                )
            if not message['text'][0].endswith(f"{PROMPT.SELF_PREFIX}: "):
                message.force_set('text', f"{message['text']}\n{PROMPT.SELF_PREFIX}: ")
        return message


@register_mutator("format_decision_tasks_for_decoder_only")
class FormatDecisionTasksForDecoderOnlyMutator(MessageMutator):
    """
    Replaces special tokens with proper prefixes.

    Before:
        (Convai2MemoryDecisionBalancedWithMemoryTeacher)
        persona: i actually like wearing suits and ties.
        Then you must be working. Or just taking some time off? __is-memory-required__
            __do-not-access-memory__

        - - - NEW EPISODE: WoiSearchDecisionTeacher - - -
        he would be so rich, I wish I had that much money! __is-search-required__
            __do-search__

    After:
        (Convai2DecoderOnlyMemoryDecisionBalancedWithMemoryTeacher)
        Personal Fact: i actually like wearing suits and ties.
        Person 1: Then you must be working. Or just taking some time off?
        Memory Decision:
            do not access memory

        - - - NEW EPISODE: WoiDecoderOnlySearchDecisionTeacher - - -
        Person 1: he would be so rich, I wish I had that much money!
        Search Decision:
            search
    """

    def message_mutation(self, message: Message) -> Message:
        context, label = message['text'], message['labels'][0]
        assert any(
            const in context
            for const in [BB3_CONST.IS_SEARCH_REQUIRED, BB3_CONST.IS_MEMORY_REQUIRED]
        ), message['id']
        context = context.replace(
            BB3_CONST.IS_SEARCH_REQUIRED, f"\n{PROMPT.SEARCH_DECISION}: "
        )
        context = context.replace(
            BB3_CONST.IS_MEMORY_REQUIRED, f"\n{PROMPT.MEMORY_DECISION}: "
        )
        context = context.replace('persona: ', f"{PROMPT.MEMORY_KNOWLEDGE_PREFIX}: ")
        assert '\n\n' not in context
        label = label.replace(BB3_CONST.DO_SEARCH, PROMPT.SEARCH)
        label = label.replace(BB3_CONST.DO_NOT_SEARCH, PROMPT.DO_NOT_SEARCH)
        label = label.replace(BB3_CONST.DO_ACCESS_MEMORY, PROMPT.ACCESS_MEMORY)
        label = label.replace(BB3_CONST.DONT_ACCESS_MEMORY, PROMPT.DO_NOT_ACCESS_MEMORY)
        message.force_set('text', context)
        message.force_set('labels', [label])
        return message


@register_mutator("format_gen_tasks_for_decoder_only")
class FormatGenTasksForDecoderOnlyMutator(MessageMutator):
    """
    For use with SGM, and MGM.

    Replaces the special tokens with proper prefixes.

    Before:
        (MSCMemoryGeneratorTeacher)
        Hi how are you doing today?
        Doing good. Not ready for the weekend to be over though.
        So true. Off to work?
        Yeah, I am an accountant. It kinda runs in the family. Ll
        My brother is employed at best buy. It does not run in the family
        Ll. That's a good place to shop. Any pets? I have a couple dogs. __generate-memory__
            I have a couple of dogs.

        ---------------
        (WoiSearchQueryTeacher)
        I live in Mesa, Arizona.
        I just bought my first home.
        a bungalow close to the city __generate-query__
            arizona

    After:
        (MSCDecoderOnlyMemoryGeneratorTeacher)
        Person 2: Hi how are you doing today?
        Person 1: Doing good. Not ready for the weekend to be over though.
        Person 2: So true. Off to work?
        Person 1: Yeah, I am an accountant. It kinda runs in the family. Ll
        Person 2: My brother is employed at best buy. It does not run in the family
        Person 1: Ll. That's a good place to shop. Any pets? I have a couple dogs.
        Memory:
            I have a couple of dogs.
        -------------------
        (WoiDecoderOnlySearchQueryTeacher)
        Person 1: I live in Mesa, Arizona.
        Person 1: I just bought my first home.
        Person 1: a bungalow close to the city
        Query:
            arizona
    """

    def message_mutation(self, message: Message) -> Message:
        context, _ = message['text'], message['labels'][0]
        assert any(
            const in context
            for const in [BB3_CONST.GENERATE_MEMORY, BB3_CONST.GENERATE_QUERY]
        )
        context = context.replace(
            BB3_CONST.GENERATE_MEMORY, f"\n{PROMPT.MEMORY_GEN_PREFIX}: "
        )
        context = context.replace(
            BB3_CONST.GENERATE_QUERY, f"\n{PROMPT.QUERY_GEN_PREFIX}: "
        )
        assert '\n\n' not in context
        message.force_set('text', context)
        return message


@register_mutator("format_knowledge_tasks_for_decoder_only")
class FormatKnowledgeForDecoderOnlyMutator(MessageMutator):
    """
    Format examples for Decoder-Only models.

    This mutator places all of the retrieved documents within the context.

    The mutator ensures that the context + label does not exceed 2048, the
    context length of OPT models.

    Before:
        - - - NEW EPISODE: Convai2MemoryKnowledgeTeacher - - -
        Hi! I work as a gourmet cook. __access-memory__
            your persona: I hate carrots.

        - - - NEW EPISODE: WoiKnowledgeTeacher - - -
        My favorite athlete is nadal.
        he will be the greatest tennis player __generate-knowledge__
            Looking for a new hobby? Or trying to decide which one will be the best use of your limited free time? We’ve got you covered.

        - - - NEW EPISODE: Convai2KnowledgeTeacher - - -
        your persona: I had a gig at local theater last night.
        your persona: I work as a stand up comedian.
        your persona: I come from a small town.
        your persona: My favorite drink is cuba libre.
        your persona: I did a few small roles in tv series.
        We all live in a yellow submarine, a yellow submarine. Morning!
        Hi! That is a great line for my next stand up.
        Lol. I am shy, anything to break the ice, and I am a beatles fan. __extract-entity__
            tv

    After:
        - - - NEW EPISODE: Convai2DecoderOnlyMemoryKnowledgeTeacher - - -
        Person 2's Persona: I'm very athletic.
        Person 2's Persona: I wear contacts.
        Person 2's Persona: I have brown hair.
        Person 2's Persona: I love bicycling.
        Person 2's Persona: I hate carrots.
        Person 1's Persona: I work as a gourmet cook. I am female.
        Person 2's Persona: I don't like carrots. I throw them away.
        Person 1's Persona: I can sing. I can sing pitch perfect.
        Person 2's Persona: I ride my bike to work. I cook.
        Person 1's Persona: I won a spelling bee. I am a student.
        Person 2's Persona: I have a good sense of humor. I can see through what you are trying to sell me.
        Person 1's Persona: I was published in the new yorker once.
        Person 2's Persona: Spelling is important to me. I am a teacher.
        Person 1's Persona: I can cook. I am not a chef.
        Person 2's Persona: My hair is brown. I am white. I have brown hair.
        Person 1's Persona: I have no hair. I am asian.
        Person 2's Persona: I like hairless asians. I like carrots.
        Person 1's Persona: I eat carrots like a horse. I love carrots.
        Person 1's Persona: I work as a gourmet cook. I have a perfect voice.
        Person 1: Hi! I work as a gourmet cook.
        Personal Fact:
            Person 2's Persona: I hate carrots.

        - - - NEW EPISODE: WoiDecoderOnlyKnowledgeTeacher - - -
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | The Ultimate List of Hobbies
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | feeling it. Eventually you�
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | lucrative.. These hobbies are rewarding
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | to do it for free by
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Teaching and Tutoring Music.
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Tutoring Children. 54.
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | you can meditate.. 66
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | a penny for your thoughts.
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Learning how to DIY. 96
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | wonderful paper art.. List of
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | pay for itself if you review
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Robot Making. 148. Pipe
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | if you have a facility near
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Pursuing one of these hobbies
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | are plenty of guided yoga courses
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | drop-shipping, running
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | friends away at weddings.. 200
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | but all age groups can enjoy
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | and birdhouses. 238.
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | start, the better you�
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | more flexible than adults, so
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Producing Electronic Music. If
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | awesome.. 290. Soap
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | games are fun alone or with
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | of 53 Hobbies for Your
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Camping. 342. Sur
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Expand your horizons and experience
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Fell Running. 389. Track
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | always wanted to forge your own
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | as you once were. Brewing
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Snowmobiling. 422.
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | sculpture racing. 454.
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | season, you get delicious homegrown
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | or just looking to expand your
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | getting started. From there,
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Looking for a new hobby?
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | views and quiet time to yourself
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | you want to be outside,
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | comes from is worth it in
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | Maybe one of the most peaceful
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | lakes, ponds; anywhere there
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | a track. Find somewhere beautiful
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | not the prettiest thing,
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | gets to know the world around
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | up, but I guarantee that
        External Knowledge: The Ultimate List of Hobbies 549 handpicked hobby ideas | for history lovers. Find something
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | Stamp Collecting: Also for anyone who
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | ways of life. Knowing how people dressed
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | intent. Even better, learn a language
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | from the next, so collect and press
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | and collect them. They are great decorations
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | games. It goes beyond the simple sport
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | and style: you may just learn a
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | better way to feel classy than by serving
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | Bread Making: A great way to customize
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | like it, without being beholden to
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | world around you. Like writing, painting
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | time you’ll soon be making
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | design. A scrapbook allows to you
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | A practical form of art, quil
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | If you enjoy adorning yourself with
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | One of the more intensive yet possibly rewarding
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | if you really have a lot of time
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | for maximum fun.. 68. Furniture
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | Build cabinets, fix plumbing, demolish
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | Dawn. Get drawn into a fantastic interactive
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | true intellectual’s game the world
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | your friends? Juggling really does help
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | help you learn about the world and its
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | safe, but if you do it right
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | for a Historical Society: Love history and
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | have. Check out these guides to the
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | can’t get your voice on
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | entertainment.. 94. Dancing: Both art
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | Nemo think so. In reality,
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | most of us have a lot to learn
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | you can dress and role play as them
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | want to prepare for Russian strategic rocket forces
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | /New Arrivals/. Taxi Car w
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | Trailer Clear Display Case. 1/25
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | Mega Hobby. /Our Family Business/.
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | at the top of our pyramid. But
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | A Truly Unimaginable List of
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | the other. They also bring different people
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | as a hobby, and is now a
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | and a true observer is the one that
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | who is known to have seen the maximum
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | that every human seeks because it gives them
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | College Dictionary.. Hobbies Related to Comput
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | of Facebook, started writing software as a
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | there are many people, who define relaxing
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | Fishing All Across Europe: Spain Has the
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | of the house, there are some who
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | card (received). Did you know?
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | for celebrities and models, or people who
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | how the 54-year-old div
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | very dear Tom Hanks, who collects
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | The world s premier supplier of radio remote
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | 16A 380W Power Supply. Red
        External Knowledge: 100 Creative Fun Hobbies to Try Ultimate List 2021 | electric aircraft.. Parts Saver ONLY $
        Person 1: My favorite athlete is nadal.
        Person 1: he will be the greatest tennis player
        Interesting Fact:
            Looking for a new hobby? Or trying to decide which one will be the best use of your limited free time? We’ve got you covered.

        - - - NEW EPISODE: Convai2DecoderOnlyKnowledgeTeacher - - -
        Person 2's Persona: I had a gig at local theater last night.
        Person 2's Persona: I work as a stand up comedian.
        Person 2's Persona: I come from a small town.
        Person 2's Persona: My favorite drink is cuba libre.
        Person 2's Persona: I did a few small roles in tv series.
        Person 1: We all live in a yellow submarine, a yellow submarine. Morning!
        Person 2: Hi! That is a great line for my next stand up.
        Person 1: Lol. I am shy, anything to break the ice, and I am a beatles fan.
        Previous Topic:
            tv
    """

    MAX_MODEL_LEN = 2048
    MAX_CONTEXT_LEN = 1024

    def __init__(self, opt):
        self.opt = opt
        self.dict = DictionaryAgent(
            {**ParlaiParser(True, True).parse_args([]), 'dict_tokenizer': 'gpt2'}
        )
        self.task = opt['task']

    def join_docs(self, docs: List[str], module: str) -> str:
        if module == 'search':
            prefix = PROMPT.EXTERNAL_KNOWLEDGE_PREFIX
            return f"{prefix}: " + f'\n{prefix}: '.join(docs)
        else:
            docs = [
                d.replace('your persona', PROMPT.SELF_MEMORY_PREFIX).replace(
                    'partner\'s persona', PROMPT.PARTNER_MEMORY_PREFIX
                )
                for d in docs
            ]
            return '\n'.join(docs)

    def reduce_doc_len(
        self,
        gen_prompt: str,
        context: str,
        joined_docs: str,
        message: Message,
        docs: List[str],
        module: str,
    ) -> str:
        prompt_toks_len = len(self.dict.txt2vec(gen_prompt))
        context_len = len(self.dict.txt2vec(context))
        docs_len = len(self.dict.txt2vec(joined_docs))
        label_len = len(self.dict.txt2vec(message['labels'][0]))
        max_doc_len = self.MAX_MODEL_LEN - context_len - label_len - prompt_toks_len
        if docs_len > max_doc_len:
            per_doc_len = max_doc_len // len(docs)
            docs = [self.dict.vec2txt(self.dict.txt2vec(d)[:per_doc_len]) for d in docs]
            joined_docs = self.join_docs(docs, module)
        return joined_docs

    def message_mutation(self, message: Message) -> Message:
        docs = message.get(CONST.RETRIEVED_DOCS)
        if (
            not docs
            or 'Entity' in message['id']
            or BB3_CONST.EXTRACT_ENTITY in message['text']
        ):
            # contextual knowledge
            text = message['text'].replace(
                f" {BB3_CONST.EXTRACT_ENTITY}",
                f"\n{PROMPT.CONTEXTUAL_KNOWLEDGE_PREFIX}: ",
            )
            # convert persona lines
            text = text.replace(
                "your persona:", f"{PROMPT.SELF_MEMORY_PREFIX}:"
            ).replace("partner's persona:", f"{PROMPT.PARTNER_MEMORY_PREFIX}")
            text = '\n'.join(
                [
                    c
                    for c in text.split('\n')
                    if BB3_CONST.DUMMY_TEXT.lower() not in c.lower()
                ]
            )
            message.force_set('text', text)
            return message
        context = (
            message['text']
            .replace(BB3_CONST.ACCESS_MEMORY, '')
            .replace(BB3_CONST.GENERATE_KNOWLEDGE, '')
            .replace(BB3_CONST.EXTRACT_ENTITY, '')
        )
        context = '\n'.join(
            [
                c
                for c in context.split('\n')
                if BB3_CONST.DUMMY_TEXT.lower() not in c.lower()
            ]
        )
        titles = message[CONST.RETRIEVED_DOCS_TITLES]
        if all(t for t in titles):
            docs = [f"{t} | {d}" for t, d in zip(titles, docs)]
            docs = [d.replace('\n', '. ') for d in docs]

        if any('persona:' in d for d in docs):
            module = 'memory'
        else:
            module = 'search'
        joined_docs = self.join_docs(docs, module)

        if module == 'search':
            gen_prompt = f"{PROMPT.SEARCH_KNOWLEDGE_PREFIX}: "
        else:
            gen_prompt = f"{PROMPT.MEMORY_KNOWLEDGE_PREFIX}: "
            label = (
                message['labels'][0]
                .replace('your persona', PROMPT.SELF_MEMORY_PREFIX)
                .replace('partner\'s persona', PROMPT.PARTNER_MEMORY_PREFIX)
            )
            message.force_set('labels', [label])

        joined_docs = self.reduce_doc_len(
            gen_prompt, context, joined_docs, message, docs, module
        )

        message.force_set('text', '\n'.join([joined_docs, context, gen_prompt]))

        return message


@register_mutator("format_knowledge_tasks_for_decoder_only_reduced_docs")
class FormatKnowledgeForDecoderOnlyReducedDocsMutator(
    FormatKnowledgeForDecoderOnlyMutator
):
    """
    Same as format_knowledge_tasks_for_decoder_only, except reduces the number of docs
    before reducing the doc length.
    """

    NUM_DOCS = 5
    CHUNK_SIZE = 500

    def message_mutation(self, message: Message) -> Message:
        from parlai.tasks.wizard_of_internet.mutators import chunk_docs_in_message

        message.force_set(
            CONST.SELECTED_SENTENCES,
            message.get(CONST.SELECTED_SENTENCES) or message['labels'],
        )
        message = chunk_docs_in_message(message, self.CHUNK_SIZE)
        docs = message.get(CONST.RETRIEVED_DOCS)
        checked_sentences = message.get(CONST.SELECTED_SENTENCES) or message['labels']
        if isinstance(checked_sentences, tuple):
            checked_sentences = [l for l in checked_sentences]
        for i in range(len(checked_sentences)):
            checked_sentences[i] = checked_sentences[i].lstrip(' ').rstrip(' ')
        assert docs
        assert checked_sentences
        context = (
            message['text']
            .replace(BB3_CONST.ACCESS_MEMORY, '')
            .replace(BB3_CONST.GENERATE_KNOWLEDGE, '')
            .replace(BB3_CONST.EXTRACT_ENTITY, '')
        )
        good_doc_indexes = [
            i for i, d in enumerate(docs) if any(c in d for c in checked_sentences)
        ]
        rest = [i for i in range(len(docs)) if i not in good_doc_indexes]
        if len(good_doc_indexes) < self.NUM_DOCS:
            remaining_left = self.NUM_DOCS - len(good_doc_indexes)
            if len(rest) >= remaining_left:
                good_doc_indexes += random.sample(rest, remaining_left)
            else:
                good_doc_indexes += rest

        docs = [d for i, d in enumerate(docs) if i in good_doc_indexes]
        titles = [
            t
            for i, t in enumerate(message[CONST.RETRIEVED_DOCS_TITLES])
            if i in good_doc_indexes
        ]
        if all(t for t in titles):
            docs = [f"{t} | {d}" for t, d in zip(titles, docs)]
            docs = [d.replace('\n', '. ') for d in docs]

        joined_docs = self.join_docs(docs, 'search')

        gen_prompt = f"{PROMPT.SEARCH_KNOWLEDGE_PREFIX}: "

        joined_docs = self.reduce_doc_len(
            gen_prompt, context, joined_docs, message, docs, 'search'
        )

        message.force_set('text', '\n'.join([joined_docs, context, gen_prompt]))

        return message


@register_mutator("format_response_tasks_for_decoder_only")
class FormatResponseTasksForDecoderOnlyMutator(MessageMutator):
    """
    Replace special tokens with appropriate prefixes.

    Before:
        - - - NEW EPISODE: WoiDialogueTeacher - - -
        My favorite game is WWE.
        fight game very interesting to played
        __knowledge__ Play WWE Games __endknowledge__
            That's really interesting! Are there are a lot WWE games out there? Why do you like them so much>

        - - - NEW EPISODE: Convai2DialogueTeacher - - -
        your persona: I had a gig at local theater last night.
        your persona: I work as a stand up comedian.
        your persona: I come from a small town.
        your persona: My favorite drink is cuba libre.
        your persona: I did a few small roles in tv series.
        We all live in a yellow submarine, a yellow submarine. Morning!
        Hi! That is a great line for my next stand up.
        Lol. I am shy, anything to break the ice, and I am a beatles fan.
        __entity__ tv __endentity__
            I can tell. I am not, you can see me in some tv shows

        - - - NEW EPISODE: Convai2DialogueFromPersonaOverlapMAMTeacher - - -
        Hi! I work as a gourmet cook.
        __memory__ your persona: I hate carrots. __endmemory__
            I don't like carrots. I throw them away.

    After:
        - - - NEW EPISODE: WoiDecoderOnlyDialogueTeacher - - -
        Person 1: My favorite game is WWE.
        Person 1: fight game very interesting to played
        Interesting Fact: Play WWE Games
            Person 2: That's really interesting! Are there are a lot WWE games out there? Why do you like them so much>

        - - - NEW EPISODE: Convai2DecoderOnlyDialogueTeacher - - -
        Person 1: We all live in a yellow submarine, a yellow submarine. Morning!
        Person 2: Hi! That is a great line for my next stand up.
        Person 1: Lol. I am shy, anything to break the ice, and I am a beatles fan.
        Previous Topic: tv
            Person 2: I can tell. I am not, you can see me in some tv shows

        - - - NEW EPISODE: Convai2DecoderOnlyDialogueFromPersonaOverlapMAMTeacher - - -
        Person 1: Hi! I work as a gourmet cook.
        Personal Fact: Person 2's Persona: I hate carrots.
            Person 2: I don't like carrots. I throw them away.
    """

    def message_mutation(self, message: Message) -> Message:
        context, _ = message['text'], message['labels'][0]
        assert any(
            const in context
            for const in [
                BB3_CONST.BEGIN_ENTITY,
                BB3_CONST.BEGIN_MEMORY,
                BB3_CONST.TOKEN_KNOWLEDGE,
            ]
        )
        context = context.split('\n')
        context = [c for c in context if BB3_CONST.DUMMY_TEXT.lower() not in c.lower()]
        if 'DialogueFrom' not in message['id']:
            context = [
                c
                for c in context
                if not ('your persona: ' in c and BB3_CONST.BEGIN_MEMORY not in c)
            ]
        knowledge = context[-2]
        if BB3_CONST.BEGIN_ENTITY in knowledge:
            begin, end = BB3_CONST.BEGIN_ENTITY, BB3_CONST.END_ENTITY
            new_prefix = PROMPT.CONTEXTUAL_KNOWLEDGE_PREFIX
        elif BB3_CONST.BEGIN_MEMORY in knowledge:
            begin, end = BB3_CONST.BEGIN_MEMORY, BB3_CONST.END_MEMORY
            new_prefix = PROMPT.MEMORY_KNOWLEDGE_PREFIX
        else:
            assert BB3_CONST.TOKEN_KNOWLEDGE in knowledge
            begin, end = BB3_CONST.TOKEN_KNOWLEDGE, BB3_CONST.TOKEN_END_KNOWLEDGE
            new_prefix = PROMPT.SEARCH_KNOWLEDGE_PREFIX
        knowledge = knowledge.replace(begin, f"{new_prefix}:")
        knowledge = knowledge.replace(f" {end}", '')
        knowledge = knowledge.replace(
            'your persona: ', f"{PROMPT.SELF_MEMORY_PREFIX}: "
        )
        knowledge = knowledge.replace(
            'partner\'s persona: ', f"{PROMPT.PARTNER_MEMORY_PREFIX}: "
        )
        message.force_set('text', '\n'.join(context[:-2] + [knowledge] + [context[-1]]))
        return message


@register_mutator("format_vanilla_dialogue_for_decoder_only")
class FormatVanillaDialogueForDecoderOnlyMutator(MessageMutator):
    def message_mutation(self, message: Message) -> Message:
        text = message['text'].split('\n')
        new_text = []
        for t in text:
            if 'your persona:' in t:
                t = t.replace('your persona', PROMPT.SELF_MEMORY_PREFIX)
            if t != BB3_CONST.DUMMY_TEXT:
                new_text.append(t)
        message.force_set('text', '\n'.join(new_text))
        return message


@register_mutator("format_light_tasks_for_decoder_only")
class FormatLIGHTForDecoderOnlyMutator(MessageMutator):
    def message_mutation(self, message: Message) -> Message:
        text = message['text']
        text = (
            text.replace('_setting_name', f"{PROMPT.LIGHT_SETTING_NAME}:")
            .replace(
                '_setting_desc',
                f"{PROMPT.LIGHT_SETTING_DESC}:",
            )
            .replace(
                '_partner_name',
                f"{PROMPT.LIGHT_PARTNER_NAME}:",
            )
            .replace(
                '_self_name',
                f"{PROMPT.LIGHT_SELF_NAME}:",
            )
            .replace(
                '_self_persona',
                f"{PROMPT.SELF_MEMORY_PREFIX}:",
            )
        )
        message.force_set('text', text)
        return message


@register_mutator("format_style_grounding_tasks_for_decoder_only")
class FormatStyleTasksForDecoderOnlyMutator(MessageMutator):
    def message_mutation(self, message: Message) -> Message:
        text = message['text']
        text = (
            text.replace(BB3_CONST.BEGIN_STYLE, f"{PROMPT.STYLE_PREFIX}:")
            .replace(f" {BB3_CONST.END_STYLE}", '')
            .replace('your persona:', f"{PROMPT.SELF_MEMORY_PREFIX}:")
        )
        message.force_set('text', text)
        return message
