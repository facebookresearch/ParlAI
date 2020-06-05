#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains code for computing Arora-style sentence embeddings, for response-
relatedness control.
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.build_data import modelzoo_path
import torchtext.vocab as vocab
from parlai.utils.misc import TimeLogger
from collections import Counter, deque
import numpy as np
import os
import pickle
import torch

CONTROLLABLE_DIR = 'controllable_dialogue'


class SentenceEmbedder(object):
    """
    A class to produce Arora-style sentence embeddings sent_emb(s) where s is a
    sentence. Also gives relatedness scores cos_sim(word_emb(w), sent_emb(s)) for words
    w with GloVe embeddings word_emb(w).

    See: "A Simple But Tough-To-Beat Baseline For Sentence Embeddings",
    Arora et al, 2017, https://openreview.net/pdf?id=SyK00v5xx
    """

    def __init__(self, word2prob, arora_a, glove_name, glove_dim, first_sv, data_path):
        """
          Inputs:
            word2prob: dict mapping words to their unigram probs
            arora_a: a float. Is the constant (called "a" in the paper)
              used to compute Arora sentence embeddings.
            glove_name: the version of GloVe to use, e.g. '840B'
            glove_dim: the dimension of the GloVe embeddings to use, e.g. 300
            first_sv: np array shape (glove_dim). The first singular value,
              used to compute Arora sentence embeddings. Can be None.
            data_path: The data path (we will use this to download glove)
        """
        self.word2prob = word2prob
        self.arora_a = arora_a
        self.glove_name = glove_name
        self.glove_dim = glove_dim
        self.first_sv = first_sv
        self.data_path = data_path
        if self.first_sv is not None:
            self.first_sv = torch.tensor(self.first_sv)  # convert to torch tensor

        self.min_word_prob = min(word2prob.values())  # prob of rarest word
        self.tt_embs = None  # will be torchtext.vocab.GloVe object
        self.emb_matrix = None  # will be np array shape (vocab_size, glove_dim)

        # Initialize a cache, which holds up to 64 sentences, along with their
        # corresponding word similarity scores (i.e. cosine sim for every word in the
        # vocab). This enables us to repeatedly retrieve sims for sentences we have
        # already processed (useful for batched beam search).
        self.cache_limit = 64
        self.cache_sent2sims = {}  # maps sent to sims. holds up to cache_limit.
        self.cache_sentqueue = deque()  # list of sents. add to right, remove from left

    def get_glove_embs(self):
        """
        Loads torchtext GloVe embs from file and stores in self.tt_embs.
        """
        if not hasattr(self, 'glove_cache'):
            self.glove_cache = modelzoo_path(self.data_path, 'models:glove_vectors')
        print('Loading torchtext GloVe embs (for Arora sentence embs)...')
        self.tt_embs = vocab.GloVe(
            name=self.glove_name, dim=self.glove_dim, cache=self.glove_cache
        )
        print('Finished loading torchtext GloVe embs')

    def get_emb_matrix(self, dictionary):
        """
        Construct an embedding matrix containing pretrained GloVe vectors for all words
        in dictionary, and store in self.emb_matrix. This is needed for response-
        relatedness weighted decoding.

        Inputs:
          dictionary: ParlAI dictionary
        """
        print(
            'Constructing GloVe emb matrix for response-relatedness weighted '
            'decoding...'
        )
        self.emb_matrix = []
        oov_indices = []  # list of dictionary indices for all OOV words
        for idx in range(len(dictionary)):
            word = dictionary[idx]
            if word in self.tt_embs.stoi:
                word_emb = self.tt_embs.vectors[self.tt_embs.stoi[word]]
            else:
                # If word is OOV, enter a zero vector instead.
                # This means that the cosine similarity will always be zero.
                word_emb = torch.zeros(self.glove_dim)
                oov_indices.append(idx)
            self.emb_matrix.append(word_emb)
        self.emb_matrix = np.stack(self.emb_matrix)  # (vocab_size, glove_dim)
        print(
            'Done constructing GloVe emb matrix; found %i OOVs of %i words'
            % (len(oov_indices), len(dictionary))
        )

        # Get the norm of each of the word vectors. This is needed for cosine sims.
        # self.emb_matrix_norm is a np array shape (vocab_size)
        self.emb_matrix_norm = np.linalg.norm(self.emb_matrix, axis=1)

        # For the OOV words which have zero vectors,
        # set the norm to 1.0 so we don't have divide-by-zero errors
        for idx in oov_indices:
            self.emb_matrix_norm[idx] = 1.0

    def get_word_sims(self, sent, sent_emb, dictionary):
        """
        Given a sentence and its Arora-style sentence embedding, compute the cosine
        similarities to it, for all words in the dictionary.

        Inputs:
          sent: string. Used only for caching lookup purposes.
          sent_emb: torch Tensor shape (glove_dim).
          dictionary: ParlAI dictionary

        Returns:
          sims: torch Tensor shape (vocab_size), containing the cosine sims.
        """
        # If we haven't initialized the GloVe emb matrix yet, do so
        if self.emb_matrix is None:
            self.get_emb_matrix(dictionary)

        # If we have already computed sims for this sentence, return it
        if sent in self.cache_sent2sims:
            sims = self.cache_sent2sims[sent]
            return sims

        # Compute the cosine similarities. Implementation from here:
        #  https://codereview.stackexchange.com/questions/55717/efficient-numpy-cosine-distance-calculation
        dotted = self.emb_matrix.dot(sent_emb)  # shape (vocab_size)
        sent_emb_norm = np.linalg.norm(sent_emb)  # norm of the sent emb. scalar
        norms = np.multiply(self.emb_matrix_norm, sent_emb_norm)  # shape (vocab_size)
        sims = np.divide(dotted, norms)  # divide dot prods by norms. shape (vocab_size)
        sims = torch.tensor(sims)  # convert to torch Tensor, shape (vocab_size)

        # Cache sims in self.cache_sent2sims
        self.cache_sentqueue.append(sent)  # append sent to right
        self.cache_sent2sims[sent] = sims  # add (sent, sims) pair to cache
        if len(self.cache_sentqueue) > self.cache_limit:
            to_remove = self.cache_sentqueue.popleft()  # remove from left
            del self.cache_sent2sims[to_remove]  # remove from cache
        assert len(self.cache_sent2sims) == len(self.cache_sentqueue)
        assert len(self.cache_sent2sims) <= self.cache_limit

        return sims

    def embed_sent(self, sent, rem_first_sv=True):
        """
        Produce a Arora-style sentence embedding for a given sentence.

        Inputs:
          sent: tokenized sentence; a list of strings
          rem_first_sv: If True, remove the first singular value when you compute the
            sentence embddings. Otherwise, don't remove it.
        Returns:
          sent_emb: tensor length glove_dim, or None.
              If sent_emb is None, that's because all of the words were OOV for GloVe.
        """
        # If we haven't loaded the torchtext GloVe embeddings, do so
        if self.tt_embs is None:
            self.get_glove_embs()

        # Lookup glove embeddings for words
        tokens = [t for t in sent if t in self.tt_embs.stoi]  # in-vocab tokens
        # glove_oov_tokens = [t for t in sent if t not in self.tt_embs.stoi]
        # if len(glove_oov_tokens)>0:
        #     print("WARNING: tokens OOV for glove: ", glove_oov_tokens)
        if len(tokens) == 0:
            print(
                'WARNING: tried to embed utterance %s but all tokens are OOV for '
                'GloVe. Returning embedding=None' % sent
            )
            return None
        word_embs = [
            self.tt_embs.vectors[self.tt_embs.stoi[t]] for t in tokens
        ]  # list of torch Tensors shape (glove_dim)

        # Get unigram probabilities for the words. If we don't have a word in word2prob,
        # assume it's as rare as the rarest word in word2prob.
        unigram_probs = [
            self.word2prob[t] if t in self.word2prob else self.min_word_prob
            for t in tokens
        ]  # list of floats
        # word2prob_oov_tokens = [t for t in tokens if t not in self.word2prob]
        # if len(word2prob_oov_tokens)>0:
        #     print('WARNING: tokens OOV for word2prob, so assuming they are '
        #           'maximally rare: ', word2prob_oov_tokens)

        # Calculate the weighted average of the word embeddings
        smooth_inverse_freqs = [
            self.arora_a / (self.arora_a + p) for p in unigram_probs
        ]  # list of floats
        sent_emb = sum(
            [word_emb * wt for (word_emb, wt) in zip(word_embs, smooth_inverse_freqs)]
        ) / len(
            word_embs
        )  # torch Tensor shape (glove_dim)

        # Remove the first singular value from sent_emb
        if rem_first_sv:
            sent_emb = remove_first_sv(sent_emb, self.first_sv)

        return sent_emb


def remove_first_sv(emb, first_sv):
    """
    Projects out the first singular value (first_sv) from the embedding (emb).

    Inputs:
      emb: torch Tensor shape (glove_dim)
      first_sv: torch Tensor shape (glove_dim)

    Returns:
      new emb: torch Tensor shape (glove_dim)
    """
    # Calculate dot prod of emb and first_sv using torch.mm:
    # (1, glove_dim) x (glove_dim, 1) -> (1,1) -> float
    dot_prod = torch.mm(torch.unsqueeze(emb, 0), torch.unsqueeze(first_sv, 1)).item()
    return emb - first_sv * dot_prod


def get_word_counts(opt, count_inputs):
    """
    Goes through the dataset specified in opt, returns word counts and all utterances.

    Inputs:
      count_inputs: If True, include both input and reply when counting words and
        utterances. Otherwise, only include reply text.

    Returns:
      word_counter: a Counter mapping each word to the total number of times it appears
      total_count: int. total word count, i.e. the sum of the counts for each word
      all_utts: list of strings. all the utterances that were used for counting words
    """
    # Create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    # Count word frequency for all words in dataset
    word_counter = Counter()
    total_count = 0
    all_utts = []
    log_timer = TimeLogger()
    while True:
        world.parley()

        # Count words in reply
        reply = world.acts[0].get('labels', world.acts[0].get('eval_labels'))[0]
        words = reply.split()
        word_counter.update(words)
        total_count += len(words)
        all_utts.append(reply)

        # Optionally count words in input text
        if count_inputs:
            input = world.acts[0]['text']
            input = input.split('\n')[-1]  # e.g. in ConvAI2, this removes persona
            words = input.split()
            word_counter.update(words)
            total_count += len(words)
            all_utts.append(input)

        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            print(text)

        if world.epoch_done():
            print('EPOCH DONE')
            break

    assert total_count == sum(word_counter.values())

    return word_counter, total_count, all_utts


def learn_arora(opt):
    """
    Go through ConvAI2 data and collect word counts, thus compute the unigram
    probability distribution. Use those probs to compute weighted sentence embeddings
    for all utterances, thus compute first principal component.

    Save all info to arora.pkl file.
    """
    arora_file = os.path.join(opt['datapath'], 'controllable_dialogue', 'arora.pkl')

    opt['task'] = 'fromfile:parlaiformat'
    opt['log_every_n_secs'] = 2

    print('Getting word counts from ConvAI2 train set...')
    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = os.path.join(
        opt['datapath'], 'controllable_dialogue', 'ConvAI2_parlaiformat', 'train.txt'
    )
    # Do include inputs because ConvAI2 train set reverses every convo:
    word_counter_train, total_count_train, all_utts_train = get_word_counts(
        opt, count_inputs=False
    )

    print('Getting word counts from ConvAI2 val set...')
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = os.path.join(
        opt['datapath'], 'controllable_dialogue', 'ConvAI2_parlaiformat', 'valid.txt'
    )
    # Don't include inputs because ConvAI2 val set doesn't reverses convos:
    word_counter_valid, total_count_valid, all_utts_valid = get_word_counts(
        opt, count_inputs=True
    )

    # Merge word counts
    word_counter = word_counter_train
    for word, count in word_counter_valid.items():
        word_counter[word] += count
    total_count = total_count_train + total_count_valid

    # Merge all_utts
    all_utts = all_utts_train + all_utts_valid

    # Compute unigram prob for every word
    print("Computing unigram probs for all words...")
    word2prob = {w: c / total_count for w, c in word_counter.items()}

    # Settings for sentence embedder
    arora_a = 0.0001
    glove_name = '840B'
    glove_dim = 300

    # Embed every sentence, without removing first singular value
    print('Embedding all sentences...')
    sent_embedder = SentenceEmbedder(
        word2prob,
        arora_a,
        glove_name,
        glove_dim,
        first_sv=None,
        data_path=opt['datapath'],
    )
    utt_embs = []
    log_timer = TimeLogger()
    for n, utt in enumerate(all_utts):
        utt_emb = sent_embedder.embed_sent(utt.split(), rem_first_sv=False)
        utt_embs.append(utt_emb)
        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(n, len(all_utts))
            print(text)

    # Use SVD to calculate singular vector
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html
    print('Calculating SVD...')
    utt_embs = np.stack(utt_embs, axis=0)  # shape (num_embs, glove_dim)
    U, s, V = np.linalg.svd(utt_embs, full_matrices=False)
    first_sv = V[0, :]  # first row of V. shape (glove_dim)

    # Remove singular vector from all embs to get complete Arora-style sent embs
    print('Removing singular vec from all sentence embeddings...')
    utt_embs_adj = [
        remove_first_sv(torch.Tensor(emb), torch.Tensor(first_sv)).numpy()
        for emb in utt_embs
    ]  # list of np arrays shape (glove_dim)

    # Make dict mapping ConvAI2 dataset utterances to Arora sent emb
    # We save this to file for convenience (e.g. if you want to inspect)
    utt2emb = {utt: emb for (utt, emb) in zip(all_utts, utt_embs_adj)}

    # Save unigram distribution, first singular value, hyperparameter value for a,
    # info about GloVe vectors used, and full dict of utt->emb to file
    print("Saving Arora embedding info to %s..." % arora_file)
    with open(arora_file, "wb") as f:
        pickle.dump(
            {
                'word2prob': word2prob,  # dict: string to float between 0 and 1
                'first_sv': first_sv,  # np array shape (glove_dim)
                'arora_a': arora_a,  # float, 0.0001
                'glove_name': glove_name,  # string, '840B'
                'glove_dim': glove_dim,  # int, 300
                'utt2emb': utt2emb,  # dict: string to np array shape (glove_dim)
            },
            f,
        )


def load_arora(opt):
    """
    Load the data in the arora.pkl file in data/controllable_dialogue.
    """
    arora_fp = os.path.join(opt['datapath'], CONTROLLABLE_DIR, 'arora.pkl')
    print("Loading Arora embedding info from %s..." % arora_fp)
    with open(arora_fp, "rb") as f:
        data = pickle.load(f)
    print("Done loading arora info.")
    return data


if __name__ == '__main__':
    parser = ParlaiParser()
    opt = parser.parse_args()
    learn_arora(opt)
