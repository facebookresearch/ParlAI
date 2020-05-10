#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utility functions and classes for handling text strings.
"""
import os
import sys as _sys


def normalize_reply(text: str, version=1) -> str:
    """
    Standardize the capitalization and punctuation spacing of the input text.

    Version 1: Fix sentence start casing, and punctuation.

    Version 2: Add trailing period, if missing.
    """

    switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]

    # add spaces so that words and punctuation can be seaprated
    new_text = text.lower()

    # normalize in case of human:
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')

    # split on punctuation to find sentence boundaries
    # capitalize stuff
    tokens = new_text.split(' ')
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in '?.!' and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = ' '.join(tokens)
    new_text = ' ' + new_text + ' '

    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])

    # get rid of surrounding whitespace
    new_text = new_text.strip()
    new_text = new_text.replace('  ', ' ')

    if version > 1 and new_text and new_text[-1] not in '!.?)"\'':
        new_text += '.'

    return new_text


def uppercase(string: str) -> str:
    """
    Make the first character of the string uppercase, if the string is non-empty.
    """
    if len(string) == 0:
        return string
    else:
        return string[0].upper() + string[1:]


def name_to_classname(name: str) -> str:
    words = name.split('_')
    class_name = ''
    for w in words:
        # capitalize the first letter
        class_name += w[0].upper() + w[1:]
    return class_name


def colorize(text, style):
    try:
        # if we're in ipython it's okay to use colors
        __IPYTHON__
        USE_COLORS = True
    except NameError:
        USE_COLORS = _sys.stdout.isatty()

    if not USE_COLORS:
        return text

    colorstyle = os.environ.get('PARLAI_COLORSTYLE')

    if colorstyle is None or colorstyle.lower() == 'steamroller':
        BLUE = '\033[1;94m'
        BOLD_LIGHT_GRAY_NOBK = '\033[1m'
        LIGHT_GRAY_NOBK = '\033[0m'
        MAGENTA = '\033[0;95m'
        HIGHLIGHT_RED_NOBK = '\033[1;31m'
        HIGHLIGHT_BLUE_NOBK = '\033[0;34m'
        RESET = '\033[0;0m'
        if style == 'highlight':
            return HIGHLIGHT_RED_NOBK + text + RESET
        if style == 'highlight2':
            return HIGHLIGHT_BLUE_NOBK + text + RESET
        elif style == 'text':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'bold_text':
            return BOLD_LIGHT_GRAY_NOBK + text + RESET
        elif style == 'labels' or style == 'eval_labels':
            return BLUE + text + RESET
        elif style == 'label_candidates':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'id':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'text2':
            return MAGENTA + text + RESET
        elif style == 'field':
            return HIGHLIGHT_BLUE_NOBK + text + RESET
        else:
            return MAGENTA + text + RESET

    if colorstyle.lower() == 'spermwhale':
        BLUE = '\033[1;94m'
        BOLD_LIGHT_GRAY = '\033[1;37;40m'
        LIGHT_GRAY = '\033[0;37;40m'
        MAGENTA = '\033[0;95m'
        HIGHLIGHT_RED = '\033[1;37;41m'
        HIGHLIGHT_BLUE = '\033[1;37;44m'
        RESET = '\033[0;0m'
        if style == 'highlight':
            return HIGHLIGHT_RED + text + RESET
        if style == 'highlight2':
            return HIGHLIGHT_BLUE + text + RESET
        elif style == 'text':
            return LIGHT_GRAY + text + RESET
        elif style == 'bold_text':
            return BOLD_LIGHT_GRAY + text + RESET
        elif style == 'labels' or style == 'eval_labels':
            return BLUE + text + RESET
        elif style == 'label_candidates':
            return LIGHT_GRAY + text + RESET
        elif style == 'id':
            return LIGHT_GRAY + text + RESET
        elif style == 'text2':
            return MAGENTA + text + RESET
        elif style == 'field':
            return HIGHLIGHT_BLUE + text + RESET
        else:
            return MAGENTA + text + RESET

    # No colorstyle specified/found.
    return text
