#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from .regexp_tokenizer import RegexpTokenizer
from .simple_tokenizer import SimpleTokenizer

# need DEFAULTS defined before this import
DEFAULTS = {'corenlp_classpath': os.getenv('CLASSPATH')}
from .corenlp_tokenizer import CoreNLPTokenizer  # noqa: E402


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


# Spacy is optional
try:
    from .spacy_tokenizer import SpacyTokenizer
except ImportError:
    pass


def get_class(name):
    if name == 'spacy':
        return SpacyTokenizer
    if name == 'corenlp':
        return CoreNLPTokenizer
    if name == 'regexp':
        return RegexpTokenizer
    if name == 'simple':
        return SimpleTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)


def get_annotators_for_args(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators


def get_annotators_for_model(model):
    return get_annotators_for_args(model.args)
