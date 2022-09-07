#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from parlai.core.mutators import register_mutator, MessageMutator, ManyEpisodeMutator
from typing import Optional
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
import parlai.tasks.wizard_of_internet.constants as CONST
from parlai.tasks.wizard_of_wikipedia.mutators import (
    AddLabel as AddLabelWizWiki,
    AddLabelLM as AddLabelLMWizWiki,
    CheckedSentenceAsLabel as CheckedSentenceAsLabelWizWiki,
    AddCheckedSentence as AddCheckedSentenceWizWiki,
)


@register_mutator("woi_add_checked_sentence_to_input")
class AddCheckedSentence(AddCheckedSentenceWizWiki):
    """
    Adds the checked sentence to the end of the text.

    E.g. run with: parlai display_data -t wizard_of_internet -n 100 -dt valid --mutators
    flatten,add_checked_sentence_to_input_woi
    """

    @property
    def checked_sentence_kword(self):
        return CONST.SELECTED_SENTENCES


@register_mutator("woi_checked_sentence_as_label")
class CheckedSentenceAsLabel(CheckedSentenceAsLabelWizWiki):
    """
    Uses the checked sentence (knowledge) as label.

    E.g. run with: parlai display_data -t wizard_of_internet -n 100 -dt valid --mutators
    flatten,checked_sentence_as_label_woi
    """

    @property
    def checked_sentence_kword(self):
        return CONST.SELECTED_SENTENCES


@register_mutator("woi_add_label_to_input")
class AddLabel(AddLabelWizWiki):
    """
    Adds the dialogue sentence to the input.

    E.g. run with: parlai display_data -t wizard_of_internet -n 100 -dt valid --mutators
    flatten,checked_sentence_as_label_woi,add_label_to_input_woi
    """

    pass


@register_mutator("woi_add_label_to_input_lm")
class AddLabelLM(AddLabelLMWizWiki):
    """
    Adds the dialogue sentence to the input (language modeling version).

    Language modeling version where a random piece of the label is sampled in
    the input. The rest is placed inside special tokens.

    E.g. run with: parlai display_data -t wizard_of_internet -n 100 -dt valid --mutators
    flatten,add_label_to_input_lm_woi

    To add the checked sentence as the label, use:
        parlai display_data -t wizard_of_internet -n 100 -dt valid --mutators
        flatten,add_label_to_input_lm_woi,checked_sentence_as_label_woi
    """

    pass


@register_mutator("woi_filter_no_passage_used")
class WoiFilterNoPassageUsed(ManyEpisodeMutator):
    """
    Allows to filter any examples where no passage was selected to base the wizard reply
    on.

    This works best in flattened mode. E.g. run with: parlai display_data -t
    wizard_of_internet -n 100 -dt valid --mutators flatten+filter_no_passage_used
    """

    def many_episode_mutation(self, episode):
        out_episodes = []
        for e in episode:
            checked_sentences = e.get(CONST.SELECTED_SENTENCES)
            checked_sentences = ' '.join(checked_sentences)
            if checked_sentences == CONST.NO_SELECTED_SENTENCES_TOKEN:
                pass
            else:
                out_episodes.append([e])
        return out_episodes


@register_mutator("woi_keep_only_no_passage_used")
class WoiKeepOnlyNoPassageUsed(ManyEpisodeMutator):
    """
    Filter all examples where passages are used.
    """

    def many_episode_mutation(self, episode):
        out_episodes = []
        for e in episode:
            checked_sentences = e.get(CONST.SELECTED_SENTENCES)
            checked_sentences = ' '.join(checked_sentences)
            if checked_sentences == CONST.NO_SELECTED_SENTENCES_TOKEN:
                out_episodes.append([e])
        return out_episodes


@register_mutator("woi_filter_selected_knowledge_in_retrieved_docs")
class WoiFilterSelectedKnowledgeInRetrievedDocs(ManyEpisodeMutator):
    """
    Allows to filter any examples where '__retrieved-docs__' field doesn't contain the
    '__selected-sentences__'.
    """

    def many_episode_mutation(self, episode):
        out_episodes = []
        for e in episode:
            checked_sentences = e.get(
                CONST.SELECTED_SENTENCES,
                e.get('labels', [CONST.NO_SELECTED_SENTENCES_TOKEN]),
            )
            docs = ' '.join(e.get('__retrieved-docs__'))
            if ' '.join(checked_sentences) != CONST.NO_SELECTED_SENTENCES_TOKEN:
                found = True
                for sent in checked_sentences:
                    s = sent.lstrip(' ').rstrip(' ')
                    if s not in docs:
                        found = False
                if found:
                    out_episodes.append([e])
            else:
                pass
        return out_episodes


def chunk_docs_in_message(message, chunk_sz):
    if CONST.RETRIEVED_DOCS not in message:
        return message
    new_message = message.copy()
    docs = message[CONST.RETRIEVED_DOCS]
    titles = message.get(CONST.RETRIEVED_DOCS_TITLES)
    urls = message.get(CONST.RETRIEVED_DOCS_URLS)
    new_docs = []
    new_titles = []
    new_urls = []
    checked_sentences = list(
        message.get(
            CONST.SELECTED_SENTENCES,
            message.get('labels', [CONST.NO_SELECTED_SENTENCES_TOKEN]),
        )
    )
    for i in range(len(checked_sentences)):
        checked_sentences[i] = checked_sentences[i].lstrip(' ').rstrip(' ')
    if ' '.join(checked_sentences) == CONST.NO_SELECTED_SENTENCES_TOKEN:
        checked_sentences = []
    for ind in range(len(docs)):
        d = docs[ind]
        # Guarantees that checked sentences are not split in half (as we split by space).
        for i in range(len(checked_sentences)):
            d = d.replace(checked_sentences[i], "||CHECKED_SENTENCE_{i}||")
        while True:
            end_chunk = d.find(' ', chunk_sz)
            if end_chunk == -1:
                # last chunk
                for i in range(len(checked_sentences)):
                    d = d.replace("||CHECKED_SENTENCE_{i}||", checked_sentences[i])
                new_docs.append(d)
                new_titles.append(titles[ind])
                new_urls.append(urls[ind])
                break
            else:
                new_d = d[0:end_chunk]
                for i in range(len(checked_sentences)):
                    new_d = new_d.replace(
                        "||CHECKED_SENTENCE_{i}||", checked_sentences[i]
                    )
                new_docs.append(new_d)
                new_titles.append(titles[ind])
                new_urls.append(urls[ind])
                d = d[end_chunk + 1 : -1]
    new_message.force_set(CONST.RETRIEVED_DOCS, new_docs)
    new_message.force_set(CONST.RETRIEVED_DOCS_TITLES, new_titles)
    new_message.force_set(CONST.RETRIEVED_DOCS_URLS, new_urls)
    return new_message


@register_mutator("woi_chunk_retrieved_docs")
class WoiChunkRetrievedDocs(MessageMutator):
    """
    Chunks '__retrieved-docs__' into smaller docs (max 100 words each).
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument(
            '--woi-doc-chunk-size',
            default=500,
            type=int,
            help='Document chunk size (in characters).',
        )

    def message_mutation(self, message: Message) -> Message:
        chunk_sz = self.opt.get('woi_doc_chunk_size', 500)
        return chunk_docs_in_message(message, chunk_sz)


@register_mutator("woi_dropout_retrieved_docs")
class WoiDropoutRetrievedDocs(MessageMutator):
    """
    Drops out '__retrieved-docs__' to only keep a maximum number in each example.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument(
            '--woi-doc-max-chunks',
            default=100,
            type=int,
            help='Largest number of chunks to use, others will be dropped out at random. Chunks containing gold checked sentences will not be removed.',
        )

    def message_mutation(self, message: Message) -> Message:
        if CONST.RETRIEVED_DOCS not in message:
            return message
        new_message = message.copy()
        docs = message.get(CONST.RETRIEVED_DOCS)
        new_docs = []
        max_chunks = self.opt.get('woi_doc_max_chunks', 100)
        if max_chunks > 0:
            keep = torch.randperm(len(docs))[0:max_chunks]
        else:
            keep = torch.randperm(len(docs))
        remove = torch.ones(len(docs))
        remove[keep] = 0

        for i in range(len(docs)):
            if remove[i] == 0:
                new_docs.append(docs[i])
            else:
                # We may still keep the doc if it contains the gold checked sentence(s).
                checked_sentences = message.get(
                    CONST.SELECTED_SENTENCES,
                    message.get('labels', [CONST.NO_SELECTED_SENTENCES_TOKEN]),
                )
                d = docs[i]
                found = False
                if ' '.join(checked_sentences) != CONST.NO_SELECTED_SENTENCES_TOKEN:
                    for sent in checked_sentences:
                        s = sent.lstrip(' ').rstrip(' ')
                        if s in d:
                            found = True
                if found:
                    new_docs.append(docs[i])

        new_message.force_set(CONST.RETRIEVED_DOCS, new_docs)
        return new_message
