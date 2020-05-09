#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains the main code for running CT and WD controlled models.
"""

import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora


# Interrogative words, used to control question-asking via weighted decoding
# From https://en.wikipedia.org/wiki/Interrogative_word
QN_WORDS = ['who', 'what', 'where', 'why', 'when', 'how', 'which', 'whom', 'whose', '?']


# ========================================
# LOADING NIDF MEASURES
# ========================================


class NIDFFeats(object):
    """
    An object to hold a vector containing the NIDF values for all words in the
    vocabulary.

    The vector is contstructed when first needed.
    """

    def __init__(self):
        self.NIDF_FEATS = None  # will be vector length vocab_size containing NIDF vals

    def make_feat_vec(self, dict):
        """
        Construct the NIDF feature vector for the given dict.
        """
        print("Constructing NIDF feature vector...")
        self.NIDF_FEATS = torch.zeros((len(dict)))
        num_oovs = 0
        for idx in range(len(dict)):
            word = dict[idx]
            if word in word2nidf:
                # Leave emoji (these appear in Twitter dataset) as NIDF=0
                # (so we don't encourage emoji when we set WD weight high for NIDF)
                if word[0] == '@' and word[-1] == '@':
                    continue
                nidf = word2nidf[word]  # between 0 and 1
                self.NIDF_FEATS[idx] = nidf
            else:
                # print("WARNING: word %s has no NIDF; marking it as NIDF=0" % word)
                num_oovs += 1  # If we don't have NIDF for this word, set as 0
        print(
            'Done constructing NIDF feature vector; of %i words in dict there '
            'were %i words with unknown NIDF; they were marked as NIDF=0.'
            % (len(dict), num_oovs)
        )

    def get_feat_vec(self, dict):
        """
        Return the NIDF feature vector.

        If necessary, construct it first.
        """
        if self.NIDF_FEATS is None:
            self.make_feat_vec(dict)
        return self.NIDF_FEATS


word2nidf = None
nidf_feats = None
arora_data = None
sent_embedder = None


def initialize_control_information(opt, build_task=True):
    """
    Loads information from word2count.pkl, arora.pkl in data/controllable_dialogue, and
    uses it to initialize objects for computing NIDF and response-relatedness controls.

    By default (build_task=True) we will also build the controllable_dialogue task i.e.
    download data/controllable_dialogue if necessary.
    """
    global word2nidf, nidf_feats, arora_data, sent_embedder

    if word2nidf is not None:
        # already loaded, no need to do anything
        return

    if build_task:
        build(opt)

    print("Loading up controllable features...")
    word2nidf = load_word2nidf(opt)  # get word2nidf dict
    nidf_feats = NIDFFeats()  # init the NIDFFeats object
    # load info for arora sentence embeddings
    arora_data = load_arora(opt)
    sent_embedder = SentenceEmbedder(
        arora_data['word2prob'],
        arora_data['arora_a'],
        arora_data['glove_name'],
        arora_data['glove_dim'],
        arora_data['first_sv'],
        data_path=opt['datapath'],
    )


# ========================================
# UTIL
# ========================================


def flatten(list_of_lists):
    """
    Flatten a list of lists.
    """
    return [item for sublist in list_of_lists for item in sublist]


def intrep_frac(lst):
    """
    Returns the fraction of items in the list that are repeated.
    """
    if len(lst) == 0:
        return 0
    num_rep = 0
    for idx in range(len(lst)):
        if lst[idx] in lst[:idx]:
            num_rep += 1
    return num_rep / len(lst)


def extrep_frac(lst1, lst2):
    """
    Returns the fraction of items in lst1 that are in lst2.
    """
    if len(lst1) == 0:
        return 0
    num_rep = len([x for x in lst1 if x in lst2])
    return num_rep / len(lst1)


def get_ngrams(text, n):
    """
    Returns all ngrams that are in the text.

    Inputs:
        text: string
        n: int
    Returns:
        list of strings (each is a ngram)
    """
    tokens = text.split()
    return [
        " ".join(tokens[i : i + n]) for i in range(len(tokens) - (n - 1))
    ]  # list of str


def matching_ngram_completions(comparison_seq, hypothesis, n):
    """
    Return the list of words that if appended to hypothesis, would create a n-gram that
    already exists in comparison_seq. For efficiency, this function represents words as
    integers not strings.

    Inputs:
        comparison_seq: list of integers
        hypothesis: list of integers or None
        n: integer

    Output:
        bad_words: list of integers
    """
    if hypothesis is None or len(hypothesis) < n - 1 or len(comparison_seq) < n:
        return []
    hypothesis = [int(i) for i in hypothesis]  # cast to list of ints
    comparison_seq = [int(i) for i in comparison_seq]  # cast to list of ints
    n_minus_1_gram = hypothesis[-(n - 1) :]  # list of ints length n-1
    bad_words = [
        comparison_seq[i]
        for i in range(n - 1, len(comparison_seq))
        if comparison_seq[i - (n - 1) : i] == n_minus_1_gram
    ]  # list of ints
    return bad_words


# ========================================
# WEIGHTED DECODING FEATURE FUNCTIONS
# These functions compute the decoding features for weighted decoding.
#
# Given a conversational history and a hypothesis (i.e. a partially generated response),
# these functions update the weighted decoding feature vector (of length vocab_size) by
# adding the new decoding feature, multiplied by its corresponding weight.
#
# All these functions have the following inputs and outputs:
#
# Inputs:
#   dict: parlai DictionaryAgent
#   hypothesis: a list of integers. This is the partially generated response,
#     represented via word indices.
#   history: a ConvAI2History. This represents the conversation history.
#   wt: a float. This is the weight for the weighted decoding feature.
#   feat: a vector length vocab_size. This will ultimately contain the sum of all the
#     weighted decoding features.
# Output:
#   feat: a vector length vocab_size. This is the feature vector, now with the new
#     weighted decoding feature added (multiplied by wt).
# ========================================


def intrep_word_used_before(dict, hypothesis, history, wt, feat, remove_stopwords):
    """
    Weighted decoding feature function. See explanation above. This feature is 1 for
    words that have already appeared within the hypothesis, 0 otherwise.

    Additional inputs:
      remove_stopwords: bool. If True, stopwords are not included when identifying words
        that have already appeared.
    """
    if hypothesis is not None:
        if remove_stopwords:
            hypothesis = [idx for idx in hypothesis if dict[idx] not in STOPWORDS]
        if len(hypothesis) > 0:
            feat[hypothesis] += wt
    return feat


def intrep_ngram_used_before(dict, hypothesis, history, wt, feat, n):
    """
    Weighted decoding feature function. See explanation above. This feature is 1 for
    words that, if added to the hypothesis, will create a n-gram that has already
    appeared in the hypothesis; otherwise 0.

    Additional inputs:
      n: int, the size of the n-grams considered.
    """
    if hypothesis is not None:
        bad_words = matching_ngram_completions(hypothesis, hypothesis, n)
        if len(bad_words) > 0:
            feat[bad_words] += wt
    return feat


def extrep_word_used_before(
    dict, hypothesis, history, wt, feat, remove_stopwords, person
):
    """
    Weighted decoding feature function. See explanation above. This feature is 1 for
    words that have already been used earlier in the conversation; otherwise 0.

    Additional inputs:
      remove_stopwords: bool. If True, stopwords are not included when identifying words
        that have already appeared.
      person: If 'self', identify words that have already been used by self (bot).
        If 'partner', identify words that have already been used by partner (human).
    """
    if person == 'self':
        prev_utts = history.own_utts
    elif person == 'partner':
        prev_utts = history.partner_utts
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return feat
    prev_words = [dict.txt2vec(utt) for utt in prev_utts]  # list of list of ints
    prev_words = list(set(flatten(prev_words)))  # list of ints, no duplicates
    if remove_stopwords:
        prev_words = [idx for idx in prev_words if dict[idx] not in STOPWORDS]
    if len(prev_words) > 0:
        feat[prev_words] += wt
    return feat


def extrep_ngram_used_before(dict, hypothesis, history, wt, feat, n, person):
    """
    Weighted decoding feature function. See explanation above. This feature is 1 for
    words that, if added to hypothesis, would create a n-gram that has already been used
    earlier in the conversation; otherwise 0.

    Additional inputs:
      n: int, the size of the n-grams considered.
      person: If 'self', identify n-grams that have already been used by self (bot).
        If 'partner', identify n-grams that have already been used by partner (human).
    """
    if person == 'self':
        prev_utts = history.own_utts
    elif person == 'partner':
        prev_utts = history.partner_utts
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return feat
    if hypothesis is None:
        return feat
    prev_utts_wordidx = [dict.txt2vec(utt) for utt in prev_utts]  # list of list of ints
    bad_words = [
        matching_ngram_completions(prev_utt, hypothesis, n)
        for prev_utt in prev_utts_wordidx
    ]  # list of list of ints
    bad_words = list(set(flatten(bad_words)))  # list of ints, no duplicates
    if len(bad_words) > 0:
        feat[bad_words] += wt
    return feat


def nidf(dict, hypothesis, history, wt, feat):
    """
    Weighted decoding feature function.

    See explanation above. This feature is equal to the NIDF (normalized inverse
    document frequency) score for each word. The score is always between 0 and 1.
    """
    feat += wt * nidf_feats.get_feat_vec(dict)
    return feat


def qn_words(dict, hypothesis, history, wt, feat):
    """
    Weighted decoding feature function.

    See explanation above. This feature is 1 for 'interrogative words', 0 otherwise.
    """
    qn_indices = [dict[w] for w in QN_WORDS]
    feat[qn_indices] += wt
    return feat


def lastutt_sim_arora_word(dict, hypothesis, history, wt, feat):
    """
    Weighted decoding feature function.

    See explanation above. Given a word w, this feature is equal to cos_sim(word_emb(w),
    sent_emb(l)) the cosine similarity between the GloVe vector for word w, and the
    Arora-style sentence embedding for the partner's last utterance l.
    """
    partner_utts = history.partner_utts
    if len(partner_utts) == 0:  # if bot goes first then do nothing
        return feat
    last_utt = partner_utts[-1]  # string
    if last_utt.strip().lower() == "__silence__":  # if bot goes first then do nothing
        return feat

    # Get last_utt_emb, which is a tensor shape (glove_dim)
    last_utt_emb = sent_embedder.embed_sent(dict.tokenize(last_utt))
    if last_utt_emb is None:
        return feat

    # Get cosine similarities, which is a tensor shape (vocab_size)
    sims = sent_embedder.get_word_sims(last_utt, last_utt_emb, dict)

    feat += wt * sims
    return feat


# In this dictionary, the keys are the names of the WD features, and the values are
# functions with inputs (dict, hypothesis, history, wt, feat), that update the feature
# vector feat.
WDFEATURE2UPDATEFN = {
    # Use to reduce repeated words within an utterance. Not used in paper.
    "intrep_word": (
        lambda x: intrep_word_used_before(
            x[0], x[1], x[2], x[3], x[4], remove_stopwords=False
        )
    ),
    # Use to reduce repeated non-stopwords within an utterance. intrep_unigram in paper.
    "intrep_nonstopword": (
        lambda x: intrep_word_used_before(
            x[0], x[1], x[2], x[3], x[4], remove_stopwords=True
        )
    ),
    # Use to reduce repeated 2-grams within an utterance. intrep_bigram in paper.
    "intrep_2gram": (
        lambda x: intrep_ngram_used_before(x[0], x[1], x[2], x[3], x[4], n=2)
    ),
    # Use to reduce repeated 3-grams within an utterance. Not used in paper.
    "intrep_3gram": (
        lambda x: intrep_ngram_used_before(x[0], x[1], x[2], x[3], x[4], n=3)
    ),
    # Use to reduce repeating words already used in previous bot utterances.
    # Not used in paper.
    "extrep_word": (
        lambda x: extrep_word_used_before(
            x[0], x[1], x[2], x[3], x[4], remove_stopwords=False, person='self'
        )
    ),
    # Use to reduce repeating non-stopwords already used in previous bot utterances.
    # extrep_unigram in paper.
    "extrep_nonstopword": (
        lambda x: extrep_word_used_before(
            x[0], x[1], x[2], x[3], x[4], remove_stopwords=True, person='self'
        )
    ),
    # Use to reduce repeating 2-grams already used in previous bot utterances.
    # extrep_bigram in paper.
    "extrep_2gram": (
        lambda x: extrep_ngram_used_before(
            x[0], x[1], x[2], x[3], x[4], n=2, person='self'
        )
    ),
    # Use to reduce repeating 3-grams already used in previous bot utterances.
    # Not used in paper.
    "extrep_3gram": (
        lambda x: extrep_ngram_used_before(
            x[0], x[1], x[2], x[3], x[4], n=3, person='self'
        )
    ),
    # Use to reduce repeating words already used in previous partner utterances.
    # Not used in paper.
    "partnerrep_word": (
        lambda x: extrep_word_used_before(
            x[0], x[1], x[2], x[3], x[4], remove_stopwords=False, person='partner'
        )
    ),
    # Use to reduce repeating non-stopwords already used in previous partner utterances.
    # Not used in paper.
    "partnerrep_nonstopword": (
        lambda x: extrep_word_used_before(
            x[0], x[1], x[2], x[3], x[4], remove_stopwords=True, person='partner'
        )
    ),
    # Use to reduce repeating 2-grams already used in previous partner utterances.
    # partnerrep_bigram in paper.
    "partnerrep_2gram": (
        lambda x: extrep_ngram_used_before(
            x[0], x[1], x[2], x[3], x[4], n=2, person='partner'
        )
    ),
    # Use to reduce repeating 3-grams already used in previous partner utterances.
    # Not used in paper.
    "partnerrep_3gram": (
        lambda x: extrep_ngram_used_before(
            x[0], x[1], x[2], x[3], x[4], n=3, person='partner'
        )
    ),
    # Use to increase/decrease the probability of high-specificity (i.e. rare) words.
    # This is the NIDF(w) weighted decoding feature mentioned in the paper.
    "nidf": (lambda x: nidf(x[0], x[1], x[2], x[3], x[4])),
    # Use to increase/decrease the probability of interrogative (i.e. question) words.
    # This is the is_qn_word(w) WD feature mentioned in the paper.
    "question": (lambda x: qn_words(x[0], x[1], x[2], x[3], x[4])),
    # Use to increase/decrease the probability of words with high response-relatedness
    # (i.e. similarity to the partner's last utterance).
    # This is the resp_rel(w) WD feature mentioned in the paper.
    "lastuttsim": (lambda x: lastutt_sim_arora_word(x[0], x[1], x[2], x[3], x[4])),
}


def get_wd_features(dict, hypothesis, history, wd_features, wd_weights):
    """
    Given a conversational history and a hypothesis (i.e. partially generated response),
    compute the Weighted Decoding features for all words in the vocabulary.

    Inputs:
        dict: parlai DictionaryAgent
        hypothesis: list of ints or None
        history: a ConvAI2History. This represents the conversation history.
        wd_features: list of strings; the names of the WD features we want to use
        wd_weights: list of floats; the weights corresponding to the WD features.
    Returns:
        wd_feat_vec: tensor shape (vocab_size), containing weighted sum of the feature
          functions, for each candidate continuation word
    """
    wd_feat_vec = torch.zeros((len(dict)))
    for f, w in zip(wd_features, wd_weights):
        wd_feat_vec = WDFEATURE2UPDATEFN[f]((dict, hypothesis, history, w, wd_feat_vec))
    return wd_feat_vec


# ========================================
# SENTENCE-LEVEL ATTRIBUTE FUNCTIONS
# Given an input utterance, these functions compute the value of the controllable
# attribute at the sentence level (more precisely, at the utterance level).
#
# All these functions have the following inputs and outputs:
#
# Inputs:
#   utt: a string, tokenized and lowercase
#   history: a ConvAI2History. This represents the conversation history.
# Output:
#   score: float. the value of the controllable attribute for utt.
# ========================================


def intrep_repeated_word_frac(utt, history, remove_stopwords):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns the fraction of words in utt that are repeated.
    Additional inputs:
      remove_stopwords: bool. If True, stopwords are removed before counting repetition.
    """
    assert utt.strip() != ""
    tokens = utt.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return intrep_frac(tokens)


def intrep_repeated_ngram_frac(utt, history, n):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns the fraction of n-grams in utt that are repeated.
    Additional inputs:
      n: int, the size of the n-grams considered.
    """
    assert utt.strip() != ""
    ngrams = get_ngrams(utt, n)
    return intrep_frac(ngrams)


def extrep_repeated_word_frac(utt, history, remove_stopwords, person):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns the fraction of words in utt that already appeared in a previous utterance.
    Additional inputs:
      remove_stopwords: bool. If True, stopwords are removed from utt before counting
        repetition.
      person: If 'self', identify words that have already been used by self (bot).
        If 'partner', identify words that have already been used by partner (human).
    """
    assert utt.strip() != ""
    if person == 'self':
        prev_utts = history.own_utts  # should already be tokenized
    elif person == 'partner':
        prev_utts = history.partner_utts  # should already be tokenized
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return 0
    tokens = utt.split()  # list of strings
    if remove_stopwords:  # remove stopwords from utt
        tokens = [t for t in tokens if t not in STOPWORDS]
    prev_words = [s.split() for s in prev_utts]  # list of list of ints
    prev_words = list(set(flatten(prev_words)))  # list of ints, no duplicates
    return extrep_frac(tokens, prev_words)


def extrep_repeated_ngram_frac(utt, history, n, person):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns fraction of n-grams in utt that already appeared in a previous utterance.
    Additional inputs:
      n: int, the size of the n-grams considered.
      person: If 'self', identify n-grams that have already been used by self (bot).
        If 'partner', identify n-grams that have already been used by partner (human).
    """
    assert utt.strip() != ""
    if person == 'self':
        prev_utts = history.own_utts  # should already be tokenized
    elif person == 'partner':
        prev_utts = history.partner_utts  # should already be tokenized
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return 0
    utt_ngrams = get_ngrams(utt, n)
    prev_ngrams = [get_ngrams(prev, n) for prev in prev_utts]  # list of list of strings
    prev_ngrams = list(set(flatten(prev_ngrams)))  # list of strings, no duplicates
    return extrep_frac(utt_ngrams, prev_ngrams)


def avg_nidf(utt, history):
    """
    Sentence-level attribute function.

    See explanation above. Returns the mean NIDF of the words in utt.
    """
    words = utt.split()
    problem_words = [w for w in words if w not in word2nidf]
    ok_words = [w for w in words if w in word2nidf]
    if len(ok_words) == 0:
        print(
            "WARNING: For all the words in the utterance '%s', we do not have the "
            "NIDF score. Marking as avg_nidf=1." % utt
        )
        return 1  # rarest possible sentence
    nidfs = [word2nidf[w] for w in ok_words]
    avg_nidf = sum(nidfs) / len(nidfs)
    if len(problem_words) > 0:
        print(
            "WARNING: When calculating avg_nidf for the utterance '%s', we don't "
            "know NIDF for the following words: %s" % (utt, str(problem_words))
        )
    assert avg_nidf >= 0 and avg_nidf <= 1
    return avg_nidf


def contains_qmark(utt, history):
    """
    Sentence-level attribute function.

    See explanation above. Returns 1 if utt contains a question mark, otherwise 0.
    """
    return int("?" in utt)


def lastutt_sim_arora_sent(utt, history):
    """
    Sentence-level attribute function. See explanation above.

    Returns
      cos_sim(sent_emb(last_utt), sent_emb(utt))
    the cosine similarity of the Arora-style sentence embeddings for the current
    response (utt) and the partner's last utterance (last_utt, which is in history).

    - If there is no last_utt (i.e. utt is the first utterance of the conversation),
      returns None.
    - If one or both of utt and last_utt are all-OOV; thus we can't compute sentence
      embeddings, return the string 'oov'.
    """
    partner_utts = history.partner_utts
    if len(partner_utts) == 0:
        # print('WARNING: returning lastuttsim = None because bot goes first')
        return None
    last_utt = partner_utts[-1]  # string
    if "__SILENCE__" in last_utt:
        assert last_utt.strip() == "__SILENCE__"
        # print('WARNING: returning lastuttsim = None because bot goes first')
        return None

    # Get sentence embeddings. Here we're naively splitting last_utt and utt; this is
    # fine given that we assume both utt and history are lowercase and tokenized.
    # Both last_utt_emb and response_emb are tensors length glove_dim (or None)
    last_utt_emb = sent_embedder.embed_sent(last_utt.split())
    response_emb = sent_embedder.embed_sent(utt.split())
    if last_utt_emb is None or response_emb is None:
        return 'oov'

    sim = torch.nn.functional.cosine_similarity(last_utt_emb, response_emb, dim=0)
    return sim.item()


def wordlist_frac(utt, history, word_list):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns the fraction of words in utt that are in word_list.
    Additional inputs:
      word_list: list of strings.
    """
    words = utt.split()
    num_in_list = len([w for w in words if w in word_list])
    return num_in_list / len(words)


# In this dict, the keys are the names of the sentence-level attributes, and the values
# are functions with input (utt, history), returning the attribute value measured on utt
ATTR2SENTSCOREFN = {
    # Proportion of words in utt that appear earlier in utt
    "intrep_word": (
        lambda x: intrep_repeated_word_frac(x[0], x[1], remove_stopwords=False)
    ),
    # Proportion of non-stopwords in utt that appear earlier in utt
    "intrep_nonstopword": (
        lambda x: intrep_repeated_word_frac(x[0], x[1], remove_stopwords=True)
    ),
    # Proportion of 2-grams in utt that appear earlier in utt
    "intrep_2gram": (lambda x: intrep_repeated_ngram_frac(x[0], x[1], n=2)),
    # Proportion of 3-grams in utt that appear earlier in utt
    "intrep_3gram": (lambda x: intrep_repeated_ngram_frac(x[0], x[1], n=3)),
    # Proportion of words in utt that appeared in a previous bot utterance
    "extrep_word": (
        lambda x: extrep_repeated_word_frac(
            x[0], x[1], remove_stopwords=False, person='self'
        )
    ),
    # Proportion of non-stopwords in utt that appeared in a previous bot utterance
    "extrep_nonstopword": (
        lambda x: extrep_repeated_word_frac(
            x[0], x[1], remove_stopwords=True, person='self'
        )
    ),
    # Proportion of 2-grams in utt that appeared in a previous bot utterance
    "extrep_2gram": (
        lambda x: extrep_repeated_ngram_frac(x[0], x[1], n=2, person='self')
    ),
    # Proportion of 3-grams in utt that appeared in a previous bot utterance
    "extrep_3gram": (
        lambda x: extrep_repeated_ngram_frac(x[0], x[1], n=3, person='self')
    ),
    # Proportion of words in utt that appeared in a previous partner utterance
    "partnerrep_word": (
        lambda x: extrep_repeated_word_frac(
            x[0], x[1], remove_stopwords=False, person='partner'
        )
    ),
    # Proportion of non-stopwords in utt that appeared in a previous partner utterance
    "partnerrep_nonstopword": (
        lambda x: extrep_repeated_word_frac(
            x[0], x[1], remove_stopwords=True, person='partner'
        )
    ),
    # Proportion of 2-grams in utt that appeared in a previous partner utterance
    "partnerrep_2gram": (
        lambda x: extrep_repeated_ngram_frac(x[0], x[1], n=2, person='partner')
    ),
    # Proportion of 3-grams in utt that appeared in a previous partner utterance
    "partnerrep_3gram": (
        lambda x: extrep_repeated_ngram_frac(x[0], x[1], n=3, person='partner')
    ),
    # Mean NIDF score of the words in utt
    "avg_nidf": (lambda x: avg_nidf(x[0], x[1])),
    # 1 if utt contains '?', 0 otherwise
    "question": (lambda x: contains_qmark(x[0], x[1])),
    # Proportion of words in utt that are interrogative words
    "qn_words": (lambda x: wordlist_frac(x[0], x[1], word_list=QN_WORDS)),
    # Cosine similarity of utt to partner's last utterance
    "lastuttsim": (lambda x: lastutt_sim_arora_sent(x[0], x[1])),
}


def eval_attr(utt, history, attr):
    """
    Given a conversational history and an utterance, compute the requested sentence-
    level attribute for utt.

    Inputs:
        utt: string. The utterance, tokenized and lowercase
        history: a ConvAI2History. This represents the conversation history.
        attr: string. The name of the sentence-level attribute.
    Returns:
        value: float. The value of the attribute for utt.
    """
    # Check everything is lowercased already
    assert utt == utt.lower()
    for thing in [history.persona_lines, history.partner_utts, history.own_utts]:
        for line in thing:
            if line != "__SILENCE__":
                assert line == line.lower()

    # Eval attribute
    return ATTR2SENTSCOREFN[attr]((utt, history))


# ========================================
# GETTING CONTROL VARIABLE BUCKETS
# For Conditional Training (CT) models, the code in this section allows us to determine
# what bucket a given control variable value should go into.
# ========================================


def get_qn_bucket_probs():
    """
    Assuming we have 11 CT question buckets (0 to 10), compute P(bucket|question=1) and
    P(bucket|question=0); this is needed so we can probabilistically assign incoming
    training examples to buckets.

    Returns:
      prob_bucket_given_qn: list of floats length 11; P(bucket|question=1)
      prob_bucket_given_notqn: list of floats length 11; P(bucket|question=0)
    """
    prob_qn = 41101 / 131438  # P(question=1), computed across ConvAI2 dataset. ~31%

    # Compute P(bucket), i.e. the total sizes of the buckets.
    # This is done by assuming that buckets 1 to 10 are equal in size, but bucket 0
    # is larger because we have more non-questions than questions. Therefore:
    # P(question=1) = P(bucket=1, question=1) + ... + P(bucket=10, question=1)
    #               = prob_bucket_n * 0.1 + ... + prob_bucket_n * 1
    #               = 5.5 * prob_bucket_n
    # Thus we can derive the value for prob_bucket_n and prob_bucket_0:
    prob_bucket_n = prob_qn / 5.5  # P(bucket=n) for n=1,...,10
    prob_bucket_0 = 1 - 10 * prob_bucket_n  # P(bucket=0)
    prob_bucket = [prob_bucket_0] + [prob_bucket_n] * 10  # list length 11, P(bucket)

    # Compute P(bucket|qn=1) and P(bucket|qn=0) using Bayes Rule:
    # P(bucket|qn=1) = P(bucket) * P(qn=1|bucket) / P(qn=1)
    # P(bucket|qn=0) = P(bucket) * P(qn=0|bucket) / P(qn=0)
    prob_bucket_given_qn = [pb * (i / 10) / prob_qn for i, pb in enumerate(prob_bucket)]
    prob_bucket_given_notqn = [
        pb * ((10 - i) / 10) / (1 - prob_qn) for i, pb in enumerate(prob_bucket)
    ]

    return prob_bucket_given_qn, prob_bucket_given_notqn


PROB_BUCKET_GIVEN_QN, PROB_BUCKET_GIVEN_NOTQN = get_qn_bucket_probs()


def bucket_question(ex, ctrl, num_buckets):
    """
    Given an example (where the target response may or may not be a question) and its
    history, probabilistically determine what question-asking CT bucket to use.

    Inputs:
      ex: message dictionary containing a bool field 'question'
      ctrl: string. The name of the CT control. Should be 'question'.
      num_buckets: int. The number of question-asking CT buckets. Assumed to be 11.
    Returns:
      out: int. bucket number.
    """
    assert num_buckets == 11
    is_qn = int(ex['question'])
    assert is_qn in [0, 1]
    is_qn = bool(is_qn)
    if is_qn:  # Sample from P(bucket|qn=1)
        out = np.random.choice(range(num_buckets), 1, p=PROB_BUCKET_GIVEN_QN)
    else:  # Sample from P(bucket|qn=0)
        out = np.random.choice(range(num_buckets), 1, p=PROB_BUCKET_GIVEN_NOTQN)
    out = int(out[0])
    return out


def sort_into_bucket(val, bucket_lbs):
    """
    Returns the highest bucket such that val >= lower bound for that bucket.

    Inputs:
      val: float. The value to be sorted into a bucket.
      bucket_lbs: list of floats, sorted ascending.

    Returns:
      bucket_id: int in range(num_buckets); the bucket that val belongs to.
    """
    num_buckets = len(bucket_lbs)
    for bucket_id in range(num_buckets - 1, -1, -1):  # iterate descending
        lb = bucket_lbs[bucket_id]
        if val >= lb:
            return bucket_id
    raise ValueError('val %f is not >= any of the lower bounds: %s' % (val, bucket_lbs))


def bucket_contvar(ex, ctrl, num_buckets):
    """
    Given ex, which contains a continuous value for a particular control variable,
    return the bucketed version of that control value.

    Inputs:
      ex: message dictionary. Assume it has key ctrl, mapping to the value.
      ctrl: string. The name of the CT control.
      num_buckets: int. The number of buckets for this control variable.
    """
    if ctrl not in ex.keys():
        raise ValueError(
            "Control %s not found in example. Available keys in "
            "this example: %s" % (ctrl, ', '.join(ex.keys()))
        )

    # Get the control variable value
    ctrl_val = ex[ctrl]  # string. the value of the control variable for this example
    if ctrl == 'avg_nidf':
        ctrl_val = float(ctrl_val)
        assert ctrl_val >= 0
        assert ctrl_val <= 1
    elif ctrl == 'lastuttsim':
        if ctrl_val == 'None':  # bot goes first in conversation
            assert num_buckets == 11
            return 10  # The last bucket is for when the bot goes first
        else:
            ctrl_val = float(ctrl_val)
            assert ctrl_val >= -1
            assert ctrl_val <= 1
    else:
        raise ValueError('Unexpected CT ctrl: %s' % ctrl)

    # Get the bucket lowerbounds
    bucket_lbs = CONTROL2BUCKETLBS[(ctrl, num_buckets)]  # lst len num_buckets of floats
    if ctrl == 'lastuttsim':
        # The 'bot goes first' bucket 10 has no lower bound
        assert len(bucket_lbs) == num_buckets - 1
    else:
        assert len(bucket_lbs) == num_buckets

    # Determine the correct bucket and return the bucket id
    return sort_into_bucket(ctrl_val, bucket_lbs)


# The default embedding size for CT control variable embeddings
CONTROL2DEFAULTEMBSIZE = {'question': 10, 'avg_nidf': 10, 'lastuttsim': 10}

# The default number of buckets for CT control variables
CONTROL2DEFAULTNUMBUCKETS = {
    'question': 11,
    'avg_nidf': 10,
    'lastuttsim': 11,  # 11th bucket is for when the bot goes first in the conversation
}

# This dictionary maps from the name of a CT control variable, to a function that
# takes (ex, ctrl, num_buckets) as input, and returns the correct bucket_id
# for that control and this example.
CONTROL2BUCKETINGFN = {
    'question': bucket_question,
    'avg_nidf': bucket_contvar,
    'lastuttsim': bucket_contvar,
}

# Bucket lowerbounds. These are produced using the get_bucket_lowerbounds.py script.
AVG_NIDF_10BUCKET_LBS = [
    0.0,
    0.1598414705378728,
    0.17498045049881217,
    0.18658836637678175,
    0.19671787445075514,
    0.2070643776875113,
    0.2182630256396894,
    0.23053753067016441,
    0.24624559431359425,
    0.2707252238670671,
]
LASTUTTSIM_10BUCKET_LBS = [
    -0.3870984613895416,
    -0.08026778697967529,
    -0.025567850098013878,
    0.019155802205204964,
    0.06262511014938354,
    0.10953287780284882,
    0.16335178911685944,
    0.2319537252187729,
    0.3283223509788513,
    0.4921867549419403,
]


# This dictionary maps from (CT control variable name, num_buckets) to the lower bounds
CONTROL2BUCKETLBS = {
    ('avg_nidf', 10): AVG_NIDF_10BUCKET_LBS,
    # Note: For lastuttsim, the 11th bucket is for when the bot goes first; it doesn't
    # have a LB but it does have an embedding.
    ('lastuttsim', 11): LASTUTTSIM_10BUCKET_LBS,
}


def get_ctrl_vec(exs, history, control_settings):
    """
    Given a batch of examples with given history, return the bucketed CT control values.
    This is used both when training and evaluating CT systems.

    Inputs:
      exs: list length batch_size of message dictionaries. Each dictionary contains
        a 'text' field, and a field for each CT control we're using, along with the
        value of the CT control variable.
      history: list length batch_size of ConvAI2History objects. These represent the
        conversation history.
      control_settings: dictionary containing info about CT controls.
        See ControllableSeq2seqAgent.control_settings.

    Returns:
      ctrl_vec: torch Tensor shape (batch_size, num_controls), with the bucketed values
        for the CT controls we're using. If there's no CT controls, return None.
    """
    if len(control_settings) == 0:
        return None

    # ctrl_vec is shape (bsz, num_controls) filled with -1's
    ctrl_vec = -torch.ones((len(exs), len(control_settings))).long()

    for batch_idx, ex in enumerate(exs):
        for ctrl, ctrl_info in control_settings.items():
            set_val = ctrl_info['set_value']  # is either int or None
            if set_val is not None:  # if we're using some preset bucket for this ctrl
                bucket = set_val  # override with set_val, an int
            else:  # bucket the control val given in ex
                if ctrl not in ex:
                    raise ValueError(
                        "The CT control '%s' is not present as a key in "
                        "the message dictionary:\n%s\nIf training a CT "
                        "model, perhaps your training data is missing the "
                        "annotations. If talking interactively, perhaps "
                        "you forgot to set --set-controls." % (ctrl, str(ex))
                    )
                num_buckets = ctrl_info['num_buckets']
                bucketing_fn = CONTROL2BUCKETINGFN[ctrl]  # bucketing fn for this ctrl
                bucket = bucketing_fn(ex, ctrl, num_buckets)  # int

            # If we have multiple CT controls, ctrl_idx tells us which order they go in
            ctrl_idx = ctrl_info['idx']  # int
            ctrl_vec[batch_idx, ctrl_idx] = bucket
    return ctrl_vec
