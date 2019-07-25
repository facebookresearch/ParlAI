#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.torch_agent import TorchAgent, Output
from .transformer_dictionary import TransformerDictionaryAgent
from parlai.core.utils import round_sigfigs
from parlai.zoo.bert.build import download
import os
import torch
import collections
from parlai.core.distributed_utils import is_distributed
try:
    from pytorch_transformers import AdamW, WarmupLinearSchedule
    from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                      BertForQuestionAnswering,
                                      XLMConfig, XLMForQuestionAnswering,
                                      XLNetConfig,
                                      XLNetForQuestionAnswering)
except ImportError:
    raise Exception(
        (
            "transformer_qa needs pytorch-transformers installed. \n "
            "pip install pytorch-transformers"
        )
    )

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering),
    'xlm': (XLMConfig, XLMForQuestionAnswering),
}


class TransformerQaAgent(TorchAgent):
    """
    QA based on Hugging Face BERT implementation.
    """

    def __init__(self, opt, shared=None):

        init_model, _ = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        # set up model and optimizers
        if shared:
            self.model = shared["model"]
            self.metrics = shared["metrics"]
        else:
            self.build_model()
            if init_model:
                print("Loading existing model parameters from " + init_model)
                self.load(init_model)
            self.metrics["loss"] = 0.0
            self.metrics["examples"] = 0

        if self.use_cuda:
            self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if shared:
            # We don't use get here because hasattr is used on optimizer later.
            if "optimizer" in shared:
                self.optimizer = shared["optimizer"]
        else:
            optim_params = [p for p in self.model.parameters()
                            if p.requires_grad]
            self.init_optim(optim_params)
            self.build_lr_scheduler()

    def _split_context_question(self, text):
        split = text.split("\n")
        question = split[-1]
        context = "\n".join(split[0:-1])
        return context, question

    def _vectorize(self, observations):
        """Convert a list of observations into input tensors for the BertForQuestionAnswering model ."""

        b_tokens_ids = []
        b_start_position = []
        b_end_position = []
        b_valid_obs = []
        b_segment_ids = []

        for obs in observations:

            context, question = self._split_context_question(obs["text"])

            question_tokens_id = (
                [self.dict.cls_idx] +
                self.dict.txt2vec(question) +
                [self.dict.sep_idx]
            )

            segment_ids = [0] * len(question_tokens_id)

            if "labels" in obs:
                # train
                context_tokens_ids, context_start_position, context_end_position, valid_obs = self.dict.spantokenize(
                    context, obs["labels"]
                )
            else:
                # valid / test
                valid_obs = True
                context_tokens_ids = self.dict.txt2vec(context)
                context_start_position = 0
                context_end_position = len(context_tokens_ids)

            segment_ids += [1] * len(context_tokens_ids)

            start_position = context_start_position + len(question_tokens_id)
            end_position = context_end_position + len(question_tokens_id)
            tokens_ids = question_tokens_id + context_tokens_ids

            if self.text_truncate > 0:
                if len(tokens_ids) > self.text_truncate:
                    original_len = len(tokens_ids)
                    tokens_ids = tokens_ids[-self.text_truncate:]
                    segment_ids = segment_ids[-self.text_truncate:]

                    num_tokens_removed = original_len - len(tokens_ids) - 1

                    if start_position <= num_tokens_removed:
                        # answer truncated - invalid data point
                        valid_obs = False
                        start_position = 0
                        end_position = 0
                    else:
                        start_position -= num_tokens_removed
                        end_position -= num_tokens_removed

            # close the sentence with end_idx
            tokens_ids += [self.dict.sep_idx]
            segment_ids.append(1)

            b_tokens_ids.append(tokens_ids)
            b_start_position.append(start_position)
            b_end_position.append(end_position)
            b_valid_obs.append(valid_obs)
            b_segment_ids.append(segment_ids)

        # if self.text_truncate is defined, make all vectors of that size
        max_tokens_length = max(self.text_truncate, max(
            [len(tokens_id) for tokens_id in b_tokens_ids]))

        b_input_mask = []

        for tokens_ids, valid_obs, segment_ids in zip(
            b_tokens_ids, b_valid_obs, b_segment_ids
        ):

            if valid_obs:
                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(tokens_ids)
            else:
                # invalid data point
                input_mask = [0] * len(tokens_ids)

            # Zero-pad up to the sequence length.
            while len(tokens_ids) < max_tokens_length:
                tokens_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            b_input_mask.append(input_mask)

            assert len(tokens_ids) == max_tokens_length
            assert len(input_mask) == max_tokens_length
            assert len(segment_ids) == max_tokens_length

        return (
            torch.tensor(b_tokens_ids, dtype=torch.long).to(self.device),
            torch.tensor(b_segment_ids, dtype=torch.long).to(self.device),
            torch.tensor(b_input_mask, dtype=torch.long).to(self.device),
            torch.tensor(b_start_position, dtype=torch.long).to(self.device),
            torch.tensor(b_end_position, dtype=torch.long).to(self.device),
        )

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(
            enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def _get_prediction(
        self, start_logits, end_logits, input_ids, start_context_position=None
    ):

        start_indexes = self._get_best_indexes(
            start_logits, self.opt["n_best_size"])
        end_indexes = self._get_best_indexes(
            end_logits, self.opt["n_best_size"])

        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction", ["start_index",
                                 "end_index", "start_logit", "end_logit"]
        )
        prelim_predictions = []

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions
                if end_index < start_index:
                    continue
                if start_context_position and start_index < start_context_position:
                    # answer in the question
                    continue
                length = end_index - start_index + 1
                if length > self.text_truncate:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index],
                        end_logit=end_logits[end_index],
                    )
                )

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= self.opt["n_best_size"]:
                break

            if pred.start_index > 0:  # this is a non-null prediction
                answer_text = self.dict.vec2txt(
                    input_ids[pred.start_index:pred.end_index]
                )
                nbest.append(
                    _NbestPrediction(
                        text=answer_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                    )
                )

        if len(nbest) > 0:
            # return the best prediciton
            return nbest[0].text
        else:
            # return an empty prediction
            return ""

    def report(self):
        base = super().report()
        m = {}
        examples = self.metrics["examples"]
        if examples > 0:
            m["examples"] = examples
            m["loss"] = self.metrics["loss"]
            m["mean_loss"] = self.metrics["loss"] / examples

        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def train_step(self, batch):

        tensors = self._vectorize(batch["observations"])

        inputs = {'input_ids': tensors[0],
                  'token_type_ids': None if self.opt["model_type"] == 'xlm' else tensors[1],
                  'attention_mask': tensors[2],
                  'start_positions': tensors[3],
                  'end_positions': tensors[4]}

        outputs = self.model(**inputs)

        loss = outputs[0]

        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu local machine.

        self.metrics["examples"] += len(batch["observations"])
        self.metrics["loss"] += loss

        self.zero_grad()
        self.backward(loss)
        self.update_params()

        # predictions
        b_start_logits = outputs[1]
        b_end_logits = outputs[2]

        predictions = []
        for start_logits, end_logits, input_ids in zip(
            b_start_logits, b_end_logits, tensors[0]
        ):
            prediction = self._get_prediction(
                start_logits, end_logits, input_ids)
            predictions.append(prediction)

        return Output(predictions)

    def eval_step(self, batch):

        b_tokens_ids, b_segment_ids, b_input_mask, b_start_context_position, _ = self._vectorize(
            batch["observations"]
        )

        inputs = {'input_ids': b_tokens_ids,
                  'token_type_ids': b_segment_ids,
                  'attention_mask': b_input_mask
                  }

        with torch.no_grad():
            outputs = self.model(**inputs)

        b_start_logits = outputs[0]
        b_end_logits = outputs[1]
        predictions = []
        for start_logits, end_logits, input_ids, start_context_position in zip(
            b_start_logits, b_end_logits, b_tokens_ids, b_start_context_position
        ):
            prediction = self._get_prediction(
                start_logits, end_logits, input_ids, start_context_position
            )
            predictions.append(prediction)

        # print(predictions)

        return Output(predictions)

    def share(self):
        """Share model parameters."""
        shared = super().share()
        shared["model"] = self.model
        shared["metrics"] = self.metrics
        return shared

    @staticmethod
    def add_cmdline_args(parser):
        TorchAgent.add_cmdline_args(parser)
        parser = parser.add_argument_group("BERT QA Arguments")
        parser.add_argument(
            "--n-best-size",
            default=20,
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json "
            "output file.",
        )
        parser.add_argument("--model-type", default=None, type=str, required=True,
                            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
        parser.add_argument("--model-name-or-path", default=None, type=str, required=True,
                            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
        parser.add_argument("--config-name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer-name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument('--fp16-opt-level', type=str, default='O1',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--t-total", default=None, type=int,
                            help="Total number of training steps for WarmupLinearSchedule.")
        parser.add_argument(
            "--do-lower-case",
            type='bool',
            default=False,
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        parser.set_defaults(dict_maxexs=0)  # skip building dictionary

    @staticmethod
    def dictionary_class():
        return TransformerDictionaryAgent

    def build_model(self):

        self.opt["model_type"] = self.opt["model_type"].lower()
        config_class, model_class = MODEL_CLASSES[self.opt["model_type"]]
        config = config_class.from_pretrained(
            self.opt["config_name"] if self.opt["config_name"] else self.opt["model_name_or_path"])
        self.model = model_class.from_pretrained(self.opt["model_name_or_path"], from_tf=bool(
            '.ckpt' in self.opt["model_name_or_path"]), config=config)

        self.n_gpu = 0
        if is_distributed():
            # use multiple machines
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[
                    self.opt['gpu']], broadcast_buffers=False
            )
        else:
            if self.use_cuda:
                self.n_gpu = torch.cuda.device_count()
                if self.n_gpu > 0:
                    #  training on multiple GPUs, single machine
                    self.model = torch.nn.DataParallel(self.model)

    def init_optim(self, params, optim_states=None, saved_optim_type=None):

        weight_decay = 0.0
        if self.opt.get('weight_decay', 0.0):
            weight_decay = self.opt.get('weight_decay', 0.0)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.opt["learningrate"], eps=self.opt["adam_epsilon"])

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.opt["fp16_opt_level"])

    def build_lr_scheduler(self, states=None, hard_reset=False):
        super().build_lr_scheduler()
        if 't_total' in self.opt and self.opt['t_total']:
            self.scheduler = WarmupLinearSchedule(
                self.optimizer, warmup_steps=self.opt['warmup_updates'], t_total=self.opt['t_total'])

    def update_params(self):
        super().update_params()
        if 't_total' in self.opt and self.opt['t_total']:
            self.scheduler.step(self._number_training_updates)
