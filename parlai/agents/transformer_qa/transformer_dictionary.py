#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.dict import DictionaryAgent
import os
try:
    from pytorch_transformers import AdamW, WarmupLinearSchedule
    from pytorch_transformers import (
        BertTokenizer, XLMTokenizer, XLNetTokenizer)
except ImportError:
    raise Exception(
        (
            "transformer_qa needs pytorch-transformers installed. \n "
            "pip install pytorch-transformers"
        )
    )


TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'xlnet': XLNetTokenizer,
    'xlm': XLMTokenizer
}


class TransformerDictionaryAgent(DictionaryAgent):
    """Allow to use the Torch Agent with the wordpiece dictionary of Hugging Face.
    """

    def __init__(self, opt):
        super().__init__(opt)

        tokenizer_class = TOKENIZER_CLASSES[opt["model_type"].lower()]
        self.tokenizer = tokenizer_class.from_pretrained(
            opt["tokenizer_name"] if opt["tokenizer_name"] else opt["model_name_or_path"], do_lower_case=opt["do_lower_case"])
        # to avoid getting the warning for sequences longer than 512 tokens
        self.tokenizer.max_len = int(1e12)

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_idx = self.tokenizer._convert_token_to_id(self.cls_token)
        self.sep_idx = self.tokenizer._convert_token_to_id(self.sep_token)
        self.pad_idx = self.tokenizer._convert_token_to_id(self.pad_token)

    def txt2vec(self, text):
        tokens_id = self.tokenizer.encode(text)
        return tokens_id

    def vec2tokens(self, tensor):
        idxs = [idx.item() for idx in tensor.cpu()]
        filtered_tokens = self.tokenizer.convert_ids_to_tokens(idxs)
        #text = self.tokenizer.decode(idxs)
        return filtered_tokens

    def vec2txt(self, tensor):
        idxs = [idx.item() for idx in tensor.cpu()]
        text = self.tokenizer.decode(idxs)
        return text

    def act(self):
        return {}

    def _find_label_index_in_text(self, text, label):
        return text.find(label)

    def spantokenize(self, text, labels):

        # get the first label
        label = labels[0]

        char_start_position = self._find_label_index_in_text(text, label)

        if char_start_position > 0:
            # label found in text

            assert text[char_start_position:char_start_position +
                        len(label)] == label

            # 1. Token words before the answer
            tokens = self.tokenizer.tokenize(text[:char_start_position])
            start_position = len(tokens)

            # 2. Token words for the answer
            answer_tokens = self.tokenizer.tokenize(
                text[char_start_position:char_start_position + len(label)]
            )
            end_position = start_position + len(answer_tokens)
            tokens.extend(answer_tokens)

            # 3. Token words after the answer
            tokens.extend(
                self.tokenizer.tokenize(text[char_start_position + len(label):])
            )

            assert tokens[start_position:end_position] == answer_tokens

            tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
            valid = True
        else:
            start_position = 0
            end_position = 0
            tokens_id = self.txt2vec(text)
            valid = False

        return tokens_id, start_position, end_position, valid
