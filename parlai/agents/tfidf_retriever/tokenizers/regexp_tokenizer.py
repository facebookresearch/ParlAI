#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Regex based tokenizer that emulates the Stanford/NLTK PTB tokenizers.

However it is purely in Python, supports robust untokenization, unicode,
and requires minimal dependencies.
"""

import regex
import logging
from .tokenizer import Tokens, Tokenizer

logger = logging.getLogger(__name__)


class RegexpTokenizer(Tokenizer):
    DIGIT = r'\p{Nd}+([:\.\,]\p{Nd}+)*'
    TITLE = (r'(dr|esq|hon|jr|mr|mrs|ms|prof|rev|sr|st|rt|messrs|mmes|msgr)'
             r'\.(?=\p{Z})')
    ABBRV = r'([\p{L}]\.){2,}(?=\p{Z}|$)'
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]++'
    HYPHEN = r'{A}([-\u058A\u2010\u2011]{A})+'.format(A=ALPHA_NUM)
    NEGATION = r"((?!n't)[\p{L}\p{N}\p{M}])++(?=n't)|n't"
    CONTRACTION1 = r"can(?=not\b)"
    CONTRACTION2 = r"'([tsdm]|re|ll|ve)\b"
    START_DQUOTE = r'(?<=[\p{Z}\(\[{<]|^)(``|["\u0093\u201C\u00AB])(?!\p{Z})'
    START_SQUOTE = r'(?<=[\p{Z}\(\[{<]|^)[\'\u0091\u2018\u201B\u2039](?!\p{Z})'
    END_DQUOTE = r'(?<!\p{Z})(\'\'|["\u0094\u201D\u00BB])'
    END_SQUOTE = r'(?<!\p{Z})[\'\u0092\u2019\u203A]'
    DASH = r'--|[\u0096\u0097\u2013\u2014\u2015]'
    ELLIPSES = r'\.\.\.|\u2026'
    PUNCT = r'\p{P}'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
            substitutions: if true, normalizes some token types (e.g. quotes).
        """
        self._regexp = regex.compile(
            '(?P<digit>%s)|(?P<title>%s)|(?P<abbr>%s)|(?P<neg>%s)|(?P<hyph>%s)|'
            '(?P<contr1>%s)|(?P<alphanum>%s)|(?P<contr2>%s)|(?P<sdquote>%s)|'
            '(?P<edquote>%s)|(?P<ssquote>%s)|(?P<esquote>%s)|(?P<dash>%s)|'
            '(?<ellipses>%s)|(?P<punct>%s)|(?P<nonws>%s)' %
            (self.DIGIT, self.TITLE, self.ABBRV, self.NEGATION, self.HYPHEN,
             self.CONTRACTION1, self.ALPHA_NUM, self.CONTRACTION2,
             self.START_DQUOTE, self.END_DQUOTE, self.START_SQUOTE,
             self.END_SQUOTE, self.DASH, self.ELLIPSES, self.PUNCT,
             self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()
        self.substitutions = kwargs.get('substitutions', True)

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Make normalizations for special token types
            if self.substitutions:
                groups = matches[i].groupdict()
                if groups['sdquote']:
                    token = "``"
                elif groups['edquote']:
                    token = "''"
                elif groups['ssquote']:
                    token = "`"
                elif groups['esquote']:
                    token = "'"
                elif groups['dash']:
                    token = '--'
                elif groups['ellipses']:
                    token = '...'

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)
