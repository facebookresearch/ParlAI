#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.dict import DictionaryAgent
try:
    from pytorch_pretrained_bert import BertTokenizer
except ImportError:
    raise ImportError(("BERT rankers needs pytorch-pretrained-BERT installed. \n "
                       "pip install pytorch-pretrained-bert"))


class BertDictionaryAgent(DictionaryAgent):
    """ Allow to use the Torch Agent with the wordpiece dictionary of Hugging Face.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.tokenizer = BertTokenizer.from_pretrained(opt["bert_vocabulary_path"])
        self.start_token = "[CLS]"
        self.end_token = "[SEP]"
        self.null_token = "[PAD]"
        self.start_idx = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[
            0]  # should be 101
        self.end_idx = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[
            0]  # should be 102
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]  # should be 0
        self[self.start_token] = self.start_idx
        self[self.end_token] = self.end_idx
        self[self.null_token] = self.pad_idx

    def txt2vec(self, text, vec_type=list):
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_id

    def vec2txt(self, tensor):
        idxs = [idx.item() for idx in tensor.cpu()]
        toks = self.tokenizer.convert_ids_to_tokens(idxs)
        return " ".join(toks)
