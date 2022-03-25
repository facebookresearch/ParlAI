#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utility Functions.
"""
from collections import Counter
from enum import Enum
import torch
from typing import Callable, List, Tuple

from parlai.core.torch_agent import Batch
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
import parlai.utils.logging as logging

try:
    from nltk.corpus import stopwords

    STOP_WORDS = stopwords.words('english')
    import nltk
except (ModuleNotFoundError, LookupError):
    logging.error("Please install NLTK: pip install nltk")
    logging.error("Then run `nltk.download('stopwords')`")
    nltk = None
    stopwords = None
    STOP_WORDS = []

try:
    import spacy
except ModuleNotFoundError:
    logging.error("Please install spacy: pip install spacy")
    spacy = None

##############
# Zoo Models #
##############
R2C2_BASE_400M = 'zoo:seeker/r2c2_base_400M/model'
R2C2_BASE_3B = 'zoo:seeker/r2c2_base_3B/model'
R2C2_BLENDERBOT_400M = 'zoo:seeker/r2c2_blenderbot_400M/model'
R2C2_BLENDERBOT_3B = 'zoo:seeker/r2c2_blenderbot_3B/model'
SEEKER_DIALOGUE_400M = 'zoo:seeker/seeker_dialogue_400M/model'
SEEKER_DIALOGUE_3B = 'zoo:seeker/seeker_dialogue_3B/model'
SEEKER_LM_DIALOGUE_3B = 'zoo:seeker/seeker_lm_dialogue_3B/model'
SEEKER_LM_MED = 'zoo:seeker/seeker_lm_med/model'
SEEKER_LM_LARGE = 'zoo:seeker/seeker_lm_large/model'
SEEKER_LM_XL = 'zoo:seeker/seeker_lm_xl/model'

##################
# Control Tokens #
##################
GENERATE_QUERY = '__generate-query__'
IS_SEARCH_REQUIRED = '__is-search-required__'
DO_SEARCH = '__do-search__'
DO_NOT_SEARCH = '__do-not-search__'

nlp = None


class SearchDecision(Enum):
    """
    Abstracting how to decide to search.
    """

    ALWAYS = 'always'
    NEVER = 'never'
    COMPUTE = 'compute'


def drm_get_batch_context(self, batch: Batch, orig_fun: Callable):
    """
    Override the get context method for the DRM.

    This override removes the knowledge response from the set of tokens being blocked in
    the dialogue response.
    """
    ctxts = orig_fun(batch)
    knowledge_start_id = self.dict.txt2vec(TOKEN_KNOWLEDGE)
    knowledge_end_id = self.dict.txt2vec(TOKEN_END_KNOWLEDGE)

    def mask_ctxttensor_between_sublists(
        ctxts: torch.Tensor, sub1: List[int], sub2: List[int]
    ) -> torch.Tensor:
        """
        Generate a mask that masks out the context between sub1 and sub2.
        """
        mask = []
        for ctxt in ctxts:
            mask_idxs = []
            should_copy = False
            idx_pointer = 0
            id_to_match = sub1
            for j, token in enumerate(ctxt.cpu().numpy()):
                if token == id_to_match[idx_pointer]:
                    idx_pointer += 1
                    if idx_pointer == 1 and id_to_match == sub1:
                        mask_idxs.append([j])
                    elif idx_pointer >= len(id_to_match):
                        should_copy = id_to_match == sub1
                        idx_pointer = 0
                        id_to_match = sub2 if (id_to_match == sub1) else sub1
                        mask_idxs[-1].append(j)
                    else:
                        mask_idxs[-1].append(j)
                elif should_copy:
                    assert isinstance(mask_idxs[-1], list)
                    mask_idxs[-1].append(j)
                elif idx_pointer > 0:
                    idx_pointer = 0
                    del mask_idxs[-1]
            mask.append(
                [
                    0 if idx in [i for sl in mask_idxs for i in sl] else 1
                    for idx in range(len(ctxt))
                ]
            )
        return torch.LongTensor(mask).to(ctxts.device)

    ctxts *= mask_ctxttensor_between_sublists(
        ctxts, knowledge_start_id, knowledge_end_id
    )
    return ctxts


def krm_get_batch_context(
    self, batch: Batch, orig_fun: Callable[[Batch], torch.Tensor]
):
    """
    Monkey-patch get_batch_context for the KRM to block on the prior knowledge
    responses.
    """
    ctxts = orig_fun(batch)
    if 'prior_knowledge_responses_vec' in batch:
        ctxts = torch.cat([ctxts, batch.prior_knowledge_responses_vec.to(ctxts)], dim=1)
    return ctxts


def krm_get_batch_context_only_knowledge(
    self, batch: Batch, orig_fun: Callable[[Batch], torch.Tensor]
):
    """
    Monkey-patch get_batch_context for the KRM to block on the prior knowledge
    responses.

    Discard conversational context.
    """
    ctxts = orig_fun(batch)
    if 'prior_knowledge_responses_vec' in batch:
        ctxts = batch.prior_knowledge_responses_vec.to(ctxts)
    return ctxts


def extract_entities(
    sentence: str,
    pos: Tuple[str] = ('PROPN', 'NOUN'),
    use_named_entities: bool = True,
    use_noun_chunks: bool = True,
) -> List[str]:
    """
    Given a sentence, extract the entities from the sentence.

    :param sentence:
        provided sentence
    :param pos:
        parts of speech to look at
    :param use_named_entities:
        whether to include named entities
    :param use_noun_chunks:
        whether to include noun chunks.

    :return entities:
        return list of entities.
    """
    global nlp
    if nlp is None:
        logging.info('Loading spacy once')
        try:
            assert spacy is not None
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            raise RuntimeError(
                'Please download: python -m spacy download en_core_web_sm'
            )
    doc = nlp(sentence)
    results = []
    if pos:
        for token in doc:
            if token.pos_ in pos:
                results.append(token)
    if use_named_entities:
        for ent in doc.ents:
            results.append(ent)
    if use_noun_chunks:
        for chunk in doc.noun_chunks:
            if chunk.text.lower() not in STOP_WORDS:
                results.append(chunk)
    results = list(set([r.text for r in results]))
    return results


def calc_f1_msmarco(pred: str, gold_items: List[str]) -> float:
    """
    Calculate F1 overlap between prediction sentence and gold labels.

    :param pred:
        prediction string
    :param gold_items:
        list of gold items

    :return f1_overlap:
    """
    try:
        assert nltk is not None
        pred_items = nltk.word_tokenize(pred)
    except IndexError:
        # malformed prediction; return 0
        return 0
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calc_f1_msc(pred, gold_items):
    """
    Calculate F1 overlap between prediction sentence and gold labels.

    :param pred:
        prediction string
    :param gold_items:
        list of gold items

    :return f1_overlap:
    """
    try:
        assert nltk is not None
        pred_items = nltk.word_tokenize(pred)
    except IndexError:
        # malformed prediction; return 0
        return 0
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)

    context_ent = extract_entities(' '.join(gold_items))
    label_ent = extract_entities(pred)
    ents = set(context_ent).intersection(label_ent)
    if len(ents) == 0:
        f1 = 0

    return f1


def remove_possible_title_from_text(
    text: str, title: str, min_title_length: int = 3, overlap_ratio: float = 0.5
) -> str:
    """
    Remove title from text document.

    :param text:
        text string
    :param title:
        title to remove
    :param min_title_length:
        minimum length of title to remove
    :param overlap_ratio:
        minimum ratio required to remove.

    :return cleaned:
        return cleaned text
    """

    def _high_intersection(s1, s2):
        return len(s1.intersection(s2)) > overlap_ratio * len(s2)

    titile_tokesn = set(title.lower().split(' '))
    if (len(titile_tokesn) < min_title_length) or (len(text) <= len(title)):
        return text
    text_beginning_tokens = set(text.lower().split(' '))
    if _high_intersection(text_beginning_tokens, titile_tokesn):
        text = text[len(title) :]
    return text
