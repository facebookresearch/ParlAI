#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains some useful code for handling history in ConvAI2 dialogues, and for
inspecting and reordering the n-best candidates after beam search.
"""

from parlai.core.torch_agent import TorchAgent
from .controls import eval_attr


class ConvAI2History(object):
    """
    An object to store the history of a ConvAI2 conversation in a convenient format.
    """

    def __init__(self, text, assume_persontokens=True, dictionary=None):
        """
        Separates text (the dialogue context) into:
          self.persona_lines: list of strings; the persona lines with "your persona:"
            removed.
          self.partner_utts: list of strings; the partner's utterances.
          self.own_utts: list of strings; the bot's own utterances.
        All of the above are lowercase and tokenized.

        Inputs:
          text: string. Contains several lines separated by \n. The first few lines are
            persona lines beginning "your persona: ", then the next lines are dialogue.
          assume_persontokens: If True, assert that the dialogue lines start with
            __p1__ and __p2__ respectively, and then remove them.
          dictionary: parlai DictionaryAgent.
        """
        p1_token, p2_token = TorchAgent.P1_TOKEN, TorchAgent.P2_TOKEN

        # Split text into lines
        lines = [t.strip() for t in text.split('\n')]

        # Identify the persona lines and remove "your persona"
        persona_lines = [t for t in lines if "your persona:" in t]
        persona_lines = [remove_prefix(pl, "your persona:") for pl in persona_lines]
        persona_lines = [fix_personaline_period(pl) for pl in persona_lines]

        # Identify the dialogue lines. It's assumed that p1 goes first.
        utts = lines[len(persona_lines) :]
        p1_utts = [utts[i] for i in range(0, len(utts), 2)]
        p2_utts = [utts[i] for i in range(1, len(utts), 2)]

        # Check for and remove the __p1__ and __p2__ prefixes
        if assume_persontokens:
            p1_utts = [remove_prefix(utt, p1_token) for utt in p1_utts]
            p2_utts = [remove_prefix(utt, p2_token) for utt in p2_utts]

        # Tokenize and lowercase
        if dictionary is not None:
            persona_lines = [" ".join(dictionary.tokenize(pl)) for pl in persona_lines]
            p1_utts = [" ".join(dictionary.tokenize(utt)) for utt in p1_utts]
            p2_utts = [" ".join(dictionary.tokenize(utt)) for utt in p2_utts]

        # Strip trailing whitespace and discard any empty lines
        self.persona_lines = [l.strip() for l in persona_lines if l.strip() != ""]
        self.partner_utts = [l.strip() for l in p1_utts if l.strip() != ""]
        self.own_utts = [l.strip() for l in p2_utts if l.strip() != ""]


def remove_prefix(utt, prefix):
    """
    Check that utt begins with prefix+" ", and then remove.

    Inputs:
      utt: string
      prefix: string

    Returns:
      new utt: utt with the prefix+" " removed.
    """
    try:
        assert utt[: len(prefix) + 1] == prefix + " "
    except AssertionError as e:
        print("ERROR: utterance '%s' does not start with '%s '" % (utt, prefix))
        print(repr(utt[: len(prefix) + 1]))
        print(repr(prefix + " "))
        raise e
    return utt[len(prefix) + 1 :]


def fix_personaline_period(line):
    """
    Sometimes the tokenized persona lines have a period at the end but no space before
    the period.

    This function fixes it, e.g. changes 'my favorite color is blue.' to 'my favorite
    color is blue .'
    """
    assert len(line) >= 2
    assert line[-1] == "." and line[-2] != " "
    pl = line[:-1] + " ."
    return pl


def show_beam_cands(n_best_beam_preds, history, dictionary):
    """
    Pretty-print the n-best candidates from beam search, along with their probabilities.

    Inputs:
      n_best_beam_preds: list length num_candidates of (prediction, score) pairs.
        prediction is a tensor of word indices, score is a single float tensor.
      history: ConvAI2History
      dictionary: parlai DictionaryAgent
    """
    print("")
    print("persona: ", history.persona_lines)
    print("partner_utts: ", history.partner_utts)
    print("own_utts: ", history.own_utts)
    print("")
    for idx, (pred, score) in enumerate(n_best_beam_preds):
        text = dictionary.vec2txt(pred.tolist())
        text = text.replace('__start__ ', '').replace(' __end__', '')
        print("%i  %.4f  %s" % (idx, score, text))
    print("")


def reorder_extrep2gram_qn(n_best_beam_preds, history, dictionary, verbose):
    """
    Inputs:
        n_best_beam_preds: list length num_candidates of (prediction, score) pairs.
          prediction is a tensor of word indices, score is a single float tensor.
        history: ConvAI2History
        dictionary: parlai DictionaryAgent
        verbose: bool. If True, print out the selection process.

    Outputs: (tensor, tensor) pair which is the chosen (prediction, score)
    """
    # Optionally print out the history
    if verbose:
        print("persona: ", history.persona_lines)
        print("partner_utts: ", history.partner_utts)
        print("own_utts: ", history.own_utts)

    # Go through the candidates, measuring their extrep_2gram level
    # Optionally print out the original ordering
    candidates = []  # list of (orig_idx, pred, text, score, extrep_2gram) tuples
    if verbose:
        print("\nORIGINAL ORDER:")
    for idx, (pred, score) in enumerate(n_best_beam_preds):
        text = dictionary.vec2txt(pred.tolist())
        text = text.replace('__start__ ', '').replace(' __end__', '')
        if verbose:
            print("%i  %.4f  %s" % (idx, score, text))
        extrep_2gram = eval_attr(text, history, 'extrep_2gram')
        candidates.append((idx, pred, text, score, extrep_2gram))

    # Sort the candidates by ascending repetition. Tiebreak using original ranking.
    candidates = sorted(candidates, key=lambda x: (x[4], x[0]))

    # Optionally print out the new ordering
    if verbose:
        print("\nSORTED BY EXTREP_2GRAM:")
        for (idx, _, text, _, extrep_2gram) in candidates:
            print("%i  %.4f  %s" % (idx, extrep_2gram, text))
        print("")

    # Identify the top-ranked (w.r.t. new ordering) candidate that contains '?'
    for (_, pred, text, score, _) in candidates:
        if "?" not in text:
            continue
        return (pred, score)

    # If there was no candidate containing '?', return top-ranked (w.r.t extrep_2gram)
    (_, pred, score, _, _) = candidates[0]
    return (pred, score)
