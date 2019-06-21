#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.agents.bert_ranker.bert_dictionary import BertDictionaryAgent


class BertSpanDictionaryAgent(BertDictionaryAgent):
    def __init__(self, opt):
        super().__init__(opt)

    def _find_label_index_in_text(self, text, label):
        return text.find(label)

    def spantokenize(self, text, labels):

        # get the first label
        label = labels[0]

        char_start_position = self._find_label_index_in_text(text, label)

        if char_start_position > 0:
            # label found in text

            assert text[char_start_position : char_start_position + len(label)] == label

            # 1. Token words before the answer
            tokens = self.tokenizer.tokenize(text[:char_start_position])
            start_position = len(tokens)

            # 2. Token words for the answer
            answer_tokens = self.tokenizer.tokenize(
                text[char_start_position : char_start_position + len(label)]
            )
            end_position = start_position + len(answer_tokens)
            tokens.extend(answer_tokens)

            # 3. Token words after the answer
            tokens.extend(
                self.tokenizer.tokenize(text[char_start_position + len(label) :])
            )

            assert tokens[start_position:end_position] == answer_tokens

            tokens_id = self.tokenizer.convert_tokens_to_ids(
                [self.start_token] + tokens + [self.end_token]
            )
            valid = True
        else:
            start_position = 0
            end_position = 0
            tokens_id = self.txt2vec(text)
            valid = False

        return tokens_id, start_position, end_position, valid

    # def txt2vec(self, text, vec_type=list):
    #     tokens = self.tokenizer.tokenize(text)
    #     tokens_id = self.tokenizer.convert_tokens_to_ids(
    #         [self.start_token] + tokens + [self.end_token]
    #     )
    #     return tokens_id
