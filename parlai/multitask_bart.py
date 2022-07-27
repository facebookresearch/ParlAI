#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension

See https://arxiv.org/abs/1910.13461.

The BART agent can be instantiated as simply `-m bart`,
however it is recommended to specify `--init-model zoo:bart/bart_large/model`
or `-mf zoo:bart/bart_large/model` to ensure correct dictionaries are saved.
"""
from __future__ import annotations


#  from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch import nn

#  import parlai.utils.fsdp as fsdp_utils
from parlai.agents.bart.convert_fairseq_to_parlai import ConversionScript
from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerEncoder,
    create_embeddings,
)
from parlai.agents.transformer.modules.modular import swappable
from parlai.agents.transformer.transformer import (
    #  _check_positional_embeddings,
    add_common_cmdline_args,
)

#  from parlai.core.agents import compare_init_model_opts
#  from parlai.core.message import Message
from parlai.core.torch_generator_agent import SearchBlocklist
from parlai.core.metrics import AverageMetric, FairseqBleuMetric, SumMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import (
    Batch,
    DictionaryAgent,
    History,
    Output,
    TorchAgent,
)
from parlai.core.torch_generator_agent import (
    BeamSearch,
    DelayedBeamSearch,
    GreedySearch,
    NucleusSampling,
    PPLMetric,
    SearchBlocklist,
    TopKSampling,
    TorchGeneratorModel,
    TorchGeneratorAgent,
)

#  from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.metrics import ExactMatchMetric, F1Metric
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.io import PathManager
from parlai.utils.logging import logging
from parlai.utils.misc import AttrDict, recursive_getattr, warn_once
from parlai.utils.torch import (
    PipelineHelper,
    argsort,
    neginf,
    total_parameters,
    trainable_parameters,
)
from parlai.utils.typing import TShared
from parlai.zoo.bart.build import BART_ARGS, CONVERSION_ARGS, download
from parlai.agents.bart.bart import BartAgent


class MultiTaskBatch(Batch):
    batchsize: int
    is_training: bool
    text_vec: Optional[torch.LongTensor]
    label_vec: Optional[torch.LongTensor]
    labels: Optional[List[str]]
    label_vec_dialog_act: Optional[torch.LongTensor]
    labels_dialog_act: Optional[List[str]]
    valid_indices: Optional[torch.LongTensor]
    candidates: Optional[List[List[str]]]
    candidate_vecs: Optional[List[List[torch.LongTensor]]]
    image: Optional[List[Any]]
    _context_original_length = Optional[torch.LongTensor]
    _context_truncate_rate = Optional[torch.LongTensor]
    _context_truncated_length = Optional[torch.LongTensor]
    _label_original_length = Optional[torch.LongTensor]
    _label_truncate_rate = Optional[torch.LongTensor]
    _label_truncated_length = Optional[torch.LongTensor]

    #  class MultiTaskTorchAgent(TorchAgent):
    """
    This overrides this method, it doesn't check much, make sure to update this afterwards
    """


@swappable(encoder=TransformerEncoder)
class TransformerGeneratorModel(TorchGeneratorModel):
    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx, **kwargs)
        self.embeddings = create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        self.encoder = self.build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            encoder_class=self.swappables.encoder,  # type: ignore
        )

    @classmethod
    def build_encoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        reduction_type='mean',
        encoder_class: Type[TransformerEncoder] = TransformerEncoder,
        **kwargs,
    ) -> TransformerEncoder:
        return encoder_class(
            opt=opt,
            embedding=embedding,
            vocabulary_size=len(dictionary),
            padding_idx=padding_idx,
            reduction_type=reduction_type,
            **kwargs,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        if mask is not None:
            mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)


@dataclass
class OutputClassifier:
    logits: torch.LongTensor = None
    #  logits_2: torch.LongTensor = None
    loss: Optional[torch.FloatTensor] = None
    labels_domain_act: torch.Tensor = None
    labels_entity: torch.Tensor = None
    preds: torch.Tensor = None
    literal_preds: List[str] = None
   #   accuracy_domain_act: float = None
    #  precision_domain_act: float = None
    #  recall_domain_act: float = None
    #  f1_domain_act: float = None
    accuracy_entity: float = None
    precision_entity: float = None
    recall_entity: float = None
    f1_entity: float = None


@dataclass
class OutputGenerator:
    logits: torch.LongTensor = None
    loss: Optional[torch.FloatTensor] = None
    preds: Optional[torch.LongTensor] = None
    encoder_states: torch.FloatTensor = None
    target_tokens: torch.Tensor = None
    labels: torch.Tensor = None
    notnull: torch.Tensor = None
    encoder_states: torch.Tensor = None


@dataclass
class GlobalModelOutput:
    classifier_decoder_1: OutputClassifier = None
    decoder: OutputGenerator = None


class TransformerPooler(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dense = nn.Linear(opt['embedding_size'], opt['embedding_size'])
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TransformerDecoderClassifier(nn.Module):
    """
    Preparing Labels
    Loss function

    See what's a smart way to handle params of decoder, couple everything in Opt or something else
    """

    def __init__(self, opt: Opt, **kwargs):
        super().__init__()
        self.opt = opt
        self.num_domain_act = len(self.opt.domain_act_dict_label2idx)
        self.num_entities = len(self.opt.entity_dict_label2idx)
        self.build_model()
        self.pooler = TransformerPooler(opt)
        self.criterion = self.build_criterion()

    def build_model(self):
        self.non_linear = F.relu

        dim = self.opt['embedding_size']
        dim_hidden = self.opt['ffn_size']

        self.lin1 = nn.Linear(dim, dim_hidden//4)
        self.lin2 = nn.Linear(dim_hidden//4, dim_hidden//8)
        self.lin3 = nn.Linear(dim_hidden//8, self.num_entities)
        #  self.lin2 = nn.Linear(dim_hidden, self.num_domain_act)
        #  self.lin3 = nn.Linear(dim_hidden, self.num_entities)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.lin3.weight)

    def build_criterion(self):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
        #  if not self.opt.fp16: return torch.nn.CrossEntropyLoss(ignore_index=self.opt.NULL_IDX, reduction='none')
        #  else: return FP16SafeCrossEntropy(ignore_index=self.opt.NULL_IDX, reduction='none')

    def forward(self, encoder_state, ys_dialog_act):
        ys_domain_act, ys_entity = ys_dialog_act

        x = self.pooler(encoder_state)
        #  x = self.non_linear(self.lin1(x))
        logits = self.lin3(self.non_linear(self.lin2(self.non_linear(self.lin1(x)))))
        loss = self.criterion(logits, ys_entity.float())

        #  logits_1 = self.lin2(x)
        #  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.opt.NULL_IDX)
        #  print(ys_domain_act.shape)

        #  label_1 = []
        #  for ys_da in ys_domain_act:
            #  non_zero = torch.nonzero(ys_da)[0]
            #  label_1.append(non_zero)
            #  if len(non_zero) > 1:
            #  #  from random import randint
            #  label_1.append(non_zero[randint(0, len(non_zero)-1)])
            #  else: label_1.append(non_zero)

        #  label_1 = torch.tensor(label_1).cuda()

        #  loss_1 = self.criterion(logits_1, ys_domain_act.float())
        #  loss_1 = loss_fn(logits_1, label_1)

        #  logits_2 = self.lin3(x)
        #  loss_2 = self.criterion(logits_2, ys_entity.float())

        # calculate loss
        #  loss = loss_1 + loss_2
        #  preds = self.evaluate(logits)
        output = OutputClassifier(
            logits=logits,
            #  logits_2=logits_2,
            loss=loss,
            labels_domain_act=ys_domain_act,
            labels_entity=ys_entity,
        )
        return output

    def evaluate(self, model_output):
        #  from sklearn.metrics import precision_recall_fscore_support

        #  probs = F.softmax(model_output.logits_1, dim=1)
        #  _, preds = torch.max(probs, axis=1)
        #  literal_preds = [
            #  self.opt.domain_act_dict_idx2label[c.item()] for c in preds
        #  ]

        #  label_1 = []
        #  for ys_da in model_output.labels_domain_act:
            #  label_1.append(torch.nonzero(ys_da)[0])
        #  label_1 = torch.tensor(label_1).cuda()

        #  model_output.accuracy_domain_act = torch.sum(preds == label_1) / len(preds)*

        #  detached_domain_act_labels = (
            #  model_output.labels_domain_act.detach().cpu().int()
        #  )

        #  p, r, f , _ = precision_recall_fscore_support(
            #  preds.cpu(), label_1.detach().cpu(), zero_division=1
        #  )
        #  (
            #  model_output.precision_dom#  ain_act,
            #  model_output.recall_domain_act,
            #  model_output.f1_domain_act
        #  ) = p.mean(), r.mean(), f.mean()


        #  model_output.accuracy_entity = 0

        #  model_output.preds, model_output.literal_preds = preds, literal_preds
        #  return model_output

#          output_domain_act = torch.sigmoid(model_output.logits_1)
        output_entity = torch.sigmoid(model_output.logits)

        #  predicted_domain_act = np.round(output_domain_act.detach().cpu())
        predicted_entity = np.round(output_entity.detach().cpu()).int()

        def get_accuracy(x, y):
            acc = 0
            for _x, _y in zip(x, y):
                # treat overprediction problem, only exact matching
                if torch.count_nonzero(_x) != torch.count_nonzero(_y):
                    continue
                gt_indices = torch.where(_y>0)
                # only inspecting those slots which are active
                predicted_slots = _x[gt_indices]
                acc += sum(predicted_slots)/torch.count_nonzero(_y)
            return acc/len(x)

        def get_accuracy_generative(x, y):
            assert x.shape == y.shape
            predicted_labels, entity_labels = [], []
            for _x, _y in zip(x, y):
                batch_predictions = []
                batch_gt = []
                _pred_indices = torch.where(_x > 0)[0].tolist()
                _gt_indices = torch.where(_y > 0)[0].tolist()

                for pred_indice, gt_indice  in zip(_pred_indices, _gt_indices):
                    predicted_val_str = self.opt.entity_dict_idx2label[pred_indice]
                    predicted_val_str = " ".join(predicted_val_str.split("-"))
                    predicted_val_str = " ".join(predicted_val_str.split("_"))

                    batch_predictions.append(predicted_val_str)

                    gt_val_str = self.opt.entity_dict_idx2label[gt_indice]
                    gt_val_str = " ".join(gt_val_str.split("-"))
                    gt_val_str = " ".join(gt_val_str.split("_"))

                    batch_gt.append(gt_val_str)


                predicted_labels.append(', '.join(batch_predictions))
                entity_labels.append(', '.join(batch_gt))

            return predicted_labels, entity_labels

        detached_entity_labels = (
            model_output.labels_entity.detach().cpu()
        )

        predictions, labels = get_accuracy_generative(predicted_entity, detached_entity_labels)
        predictions, labels = " ; ".join(predictions), " ; ".join(labels)

        accuracy_entity =  ExactMatchMetric.compute(predictions, [labels])

        precision = F1Metric.compute(predictions, [labels], "precision")
        recall = F1Metric.compute(predictions, [labels], output="recall")
        f1 = F1Metric.compute(predictions, [labels], output="f1")


        if not isinstance(accuracy_entity, int) or not isinstance(accuracy_entity, float):
            accuracy_entity = accuracy_entity.value()
        if not isinstance(precision, int) or not isinstance(precision, float):
            precision = precision.value()
        if not isinstance(recall, int) or not isinstance(recall, float):
            recall = recall.value()
        if not isinstance(f1, int) or not isinstance(f1, float):
            f1 = f1.value()


        (
            model_output.accuracy_entity,
            model_output.precision_entity,
            model_output.recall_entity,
            model_output.f1_entity,
        ) = accuracy_entity, precision, recall, f1

        return model_output


class TransformerDecoderGenerator(TransformerDecoder):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(opt, *args, **kwargs)
        #  self.opt = opt
        self.criterion = self.build_criterion()
        self.show_token_details = opt.get(
            'verbose', False
        ) or 'token_losses' in opt.get('display_add_fields', '')
        self.skip_generation = opt.get('skip_generation', False)
        self.rank_candidates = opt['rank_candidates']
        self.compute_tokenized_bleu = opt.get('compute_tokenized_bleu', False)
        label_truncate = opt.get('label_truncate') or opt.get('truncate')
        self.label_truncate = label_truncate if label_truncate >= 0 else None
        self.beam_size = opt.get('beam_size', 1)
        self.beam_min_length = opt.get('beam_min_length', 1)
        self.beam_context_block_ngram = opt.get('beam_context_block_ngram', -1)
        self.beam_block_ngram = opt.get('beam_block_ngram', -1)
        self.temperature = opt.get('temperature', 1.0)
        assert self.temperature > 0, '--temperature must be greater than 0'
        self.beam_block_list: Optional[SearchBlocklist] = None

    def forward(self, encoder_states, ys):
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(
            1, 0, seqlen - 1
        )  # performs trimming as per seq_len, [16, 79]
        if (ys[:, 0] == self.opt.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(
            bsz, inputs
        )  # [16, 79]
        latent, _ = super().forward(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        if logits.size(1) != ys.size(1):
            logits = logits[:, 1:, :]
            preds = preds[:, 1:]

        logits_view = logits.reshape(-1, logits.size(-1))
        loss = self.criterion(logits_view, ys.view(-1))
        loss = loss.view(logits.shape[:-1]).sum(dim=1)

        # target tokens compute, not sure
        notnull = ys.ne(self.opt.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)

        return OutputGenerator(
            logits=logits,
            preds=preds,
            loss=loss,
            target_tokens=target_tokens,
            labels=ys,
            notnull=notnull,
            encoder_states=encoder_states,
        )

    def build_criterion(self):
        if not self.opt.fp16:
            return torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        else:
            return FP16SafeCrossEntropy(ignore_index=0, reduction='none')

    def _get_initial_forced_decoder_input(
        self, bsz: int, inputs: torch.LongTensor
    ):
        """
        Return initial input to the decoder.

        Override TGA._get_initial_forced_decoder_input to seed EOS BOS.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        tens = (
            torch.LongTensor([self.opt.END_IDX, self.opt.START_IDX])
            .to(inputs)
            .detach()
            .expand(bsz, 2)
        )
        return torch.cat([tens, inputs], 1)

    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[str, Any],
        inds: Union[List[int], torch.LongTensor],
    ) -> Optional[Dict[str, Any]]:
        """
        Incremental state is weird to handle when we seed decoder with two inputs
        initially.
        """
        # we only have this method called when it's actually being used
        assert incremental_state is not None
        assert len(incremental_state) > 0

        for incr_state_l in incremental_state.values():
            assert 'self_attn' in incr_state_l
            assert 'prev_mask' in incr_state_l['self_attn']
            self_attn_mask = incr_state_l['self_attn']['prev_mask']
            # check this is on the very first run with incremental state
            if self_attn_mask.ndim == 3 and tuple(
                self_attn_mask.shape[1:]
            ) == (2, 2):
                # cut off the inappropriate incremental state
                incr_state_l['self_attn']['prev_mask'] = self_attn_mask[
                    :, -1:, :
                ]

        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.layers)
        }

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.opt.END_IDX:
                break
            elif i != self.opt.START_IDX:
                new_vec.append(i)
        return self.opt.dict.vec2txt(new_vec)

    def output(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute output logits.

        Override standard TGM output to _not_ prevent generation of BOS.
        """
        # tensor.shape -> [batch_size, variable seq len, embedding_size] [16, n, 1024]
        # embeddings.weight.shape -> [50264 (vocab_size), 1024]
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # output.shape -> [16, n, 50264]

        return output

    def get_prefix_tokens(
        self, batch: MultiTaskBatch
    ) -> Optional[torch.LongTensor]:
        return None

    def evaluate(self, batch, model_output):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        cand_scores = None
        token_losses = None
        text_token_info = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            #  loss, model_output = self.compute_loss(batch, return_output=True)
            if self.show_token_details:
                token_losses = self._construct_label_token_losses(
                    batch.label_vec, model_output
                )

        beam_preds_scores = None
        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            prefix_tokens = self.get_prefix_tokens(batch)
            beam_preds_scores, beams = self._generate(
                batch,
                model_output.encoder_states,
                self.beam_size,
                maxlen,
                prefix_tokens=prefix_tokens,
            )
            preds, _, _ = zip(*beam_preds_scores)

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            beam_texts_token_info: List[List[List[Tuple]]] = []
            for beam in beams:
                beam_texts.append([])
                if self.show_token_details:
                    beam_texts_token_info.append([])

                for (
                    tokens,
                    score,
                    token_metadata,
                ) in beam.get_rescored_finished():
                    try:
                        if self.show_token_details:
                            beam_texts_token_info[-1].append(
                                self._construct_generated_token_details(
                                    tokens, token_metadata
                                )
                            )
                        beam_texts[-1].append(
                            (self._v2t(tokens), score.item())
                        )
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

        cand_choices = None
        cand_scores = None
        if self.rank_candidates:
            cand_choices, cand_scores = self.rank_eval_label_candidates(
                batch, bsz
            )

        text = (
            [self._v2t(pred_data[0]) for pred_data in beam_preds_scores]
            if beam_preds_scores is not None
            else None
        )

        if self.show_token_details and beam_preds_scores is not None:
            text_token_info = []
            for beam_text_token_info in beam_texts_token_info:
                text_token_info.append(beam_text_token_info[0])

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
        retval = Output(
            text,
            cand_choices,
            token_losses=token_losses,
            cand_scores=cand_scores,
        )

        if not self.skip_generation:
            retval.beam_texts = beam_texts
            retval.beam_texts_token_info = beam_texts_token_info
            retval.text_token_info = text_token_info
        return retval

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        if mask is not None:
            mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def _construct_label_token_losses(self, labels, model_output):
        # Get non-aggregated losses
        scores, _, _ = model_output
        score_view = scores.reshape(-1, scores.size(-1))
        losses = self.criterion(score_view, labels.view(-1)).view(
            len(labels), -1
        )

        # Zip decoded tokens with losses
        token_losses = []
        for i, label in enumerate(labels):
            token_losses.append(
                list(
                    zip(
                        [self.dict[token] for token in label.tolist()],
                        losses[i].tolist(),
                    )
                )
            )
        return token_losses

    def _generate(
        self,
        batch: Batch,
        encoder_states: torch.LongTensor,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.

        :return:
            tuple (beam_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score, token_metadata) tuples for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        #  model = self.model
        #  if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #  model = self.model.module
        #  encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            batch_context_list = self._get_batch_context(batch).tolist()
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                .set_batch_context(batch_context_list, batch_idx)
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                for _ in range(bsz)
            ]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = (
            torch.arange(bsz)
            .to(dev)
            .unsqueeze(1)
            .repeat(1, beam_size)
            .view(-1)
        )
        encoder_states = self.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = super().forward(
                decoder_input, encoder_states, incr_state
            )
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = self.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts]
                prefix_mask = torch.ones_like(score, dtype=torch.bool)
                prefix_mask[
                    :, :, prefix_toks
                ] = False  # everything except prefix toks should be neginf
                score[prefix_mask] = neginf(score.dtype)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = self.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [
            n_best_list[0] for n_best_list in n_best_beam_preds_scores
        ]

        return beam_preds_scores, beams

    def _get_batch_context(self, batch):
        """
        Version of TGA._get_context() that operates on full batches for speed.
        """
        if self.beam_context_block_ngram <= 0:
            # We aren't context blocking, return empty tensor of the correct size
            return torch.zeros(batch.batchsize, 0, dtype=torch.long)

        ctxt = batch.text_vec
        if self.beam_block_full_context:
            ctxt = batch.full_text_vec
        return ctxt

    def _get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Return next decoder input.

        :param prev_input:
            previous input to decoder
        :param selection:
            token selections for current timestep
        :param inds:
            incremental state indices

        :return decoder input:
            return decoder input for next timestep
        """
        prev_input = torch.index_select(prev_input, 0, incr_state_inds)
        decoder_input = torch.cat([prev_input, selection], dim=-1)
        return decoder_input

    def _add_generation_metrics(self, batch, preds):
        """
        Can be overridden to allow for some metrics on the generations calculated at
        eval.
        """
        self.record_local_metric(
            'gen_n_toks',
            AverageMetric.many([p.size(0) for p in preds], [1] * len(preds)),
        )

    def _compute_fairseq_bleu(self, batch: Batch, preds):
        """
        Compute BLEU score between text and label, using the FAIRSeq BLEU Scorer.

        :param batch:
            Batch of observations
        :param texts:
            list of string predictions
        """
        all_results = []
        label_vec = batch.label_vec
        assert label_vec is not None, "label_vec must exist for fairseq bleu"
        for i, t in enumerate(preds):
            result = FairseqBleuMetric.compute_many(
                t,
                label_vec[i].unsqueeze(0),
                pad_idx=self.NULL_IDX,
                end_idx=self.END_IDX,
                unk_idx=self.dict[self.dict.unk_token],
            )
            if result is None:
                return
            all_results.append(result)

        bleu_scores = list(zip(*all_results))
        for k in range(4):
            self.record_local_metric(f'fairseq_bleu{k + 1}', bleu_scores[k])

    def rank_eval_label_candidates(self, batch, batchsize):
        """
        Rank label_candidates during eval_step.

        Can be overridden to allow for different ways of ranking candidates. Must have
        `--rank-candidates` set to True. By default, we roughly compute PPL to rank the
        candidates.
        """
        # compute roughly ppl to rank candidates
        cand_choices = []
        cand_choices_scores = []
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        for i in range(batchsize):
            num_cands = len(batch.candidate_vecs[i])
            enc = self.model.reorder_encoder_states(
                encoder_states, [i] * num_cands
            )
            cands, _ = self._pad_tensor(batch.candidate_vecs[i], is_label=True)
            cands = cands.to(batch.text_vec.device)
            scores, _ = self.model.decode_forced(enc, cands)
            score_view = scores.reshape(num_cands * cands.size(1), -1)
            cand_losses = F.cross_entropy(
                score_view, cands.view(-1), reduction='none'
            ).view(num_cands, cands.size(1))
            # now cand_losses is cands x seqlen size, but we still need to
            # check padding and such
            mask = (cands != self.NULL_IDX).float()
            cand_scores = (cand_losses * mask).sum(dim=1) / (
                mask.sum(dim=1) + 1e-9
            )
            sorted_scores, ordering = cand_scores.sort()
            cand_choices.append([batch.candidates[i][o] for o in ordering])
            cand_choices_scores.append(sorted_scores.tolist())

        return cand_choices, cand_choices_scores

    def _treesearch_factory(self, device, verbose=False):
        method = self.opt.get('inference', 'greedy')
        beam_size = self.opt.get('beam_size', 1)
        if method == 'greedy':
            return GreedySearch(
                beam_size,
                min_length=0,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == 'beam':
            return BeamSearch(
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == 'delayedbeam':
            return DelayedBeamSearch(
                self.opt['topk'],
                self.opt['beam_delay'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == 'topk':
            return TopKSampling(
                self.opt['topk'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == 'nucleus':
            return NucleusSampling(
                self.opt['topp'],
                beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.opt.NULL_IDX,
                bos_token=self.opt.START_IDX,
                eos_token=self.opt.END_IDX,
                device=device,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Can't use inference method {method}")

    def _construct_generated_token_details(self, tokens, tokens_metadata):
        tokens_as_txt = [self.dict[int(token)] for token in tokens]
        return list(zip(tokens_as_txt, tokens_metadata))

    def _get_initial_decoder_input(
        self, bsz: int, beam_size: int, dev: torch.device
    ) -> torch.LongTensor:
        """
        Override to seed decoder with EOS BOS token.
        """
        return (
            torch.LongTensor([self.opt.END_IDX, self.opt.START_IDX])  # type: ignore
            .expand(bsz * beam_size, 2)
            .to(dev)
        )


class MultiTaskBartModel(TransformerGeneratorModel):
    """
    BART Model.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        self.opt = opt
        self.opt.dict = dictionary
        self._task_specific_init()
        super().__init__(opt, dictionary, **kwargs)
        self.build_decoder(opt, self.embeddings, dictionary, **kwargs)

    def _task_specific_init(self):
        self.opt.domain_act_list = [
            'None',
            'Taxi-Request',
            'Police-Inform',
            'Hotel-Inform',
            'Hotel-Request',
            'Police-Request',
            'Hospital-Request',
            'Hospital-Inform',
            'general-greet',
            'Restaurant-Request',
            'Attraction-Inform',
            'Restaurant-Inform',
            'Taxi-Inform',
            'Attraction-Request',
            'general-bye',
            'Train-Inform',
            'general-thank',
            'Train-Request',
        ]
        self.opt.entity_list = [
            'none',
            'Attraction-Inform_none',
            'Attraction-Inform_type',
            'Attraction-Inform_area',
            'Attraction-Inform_name',
            'Attraction-Inform_entrancefee',
            'Attraction-Request_phone',
            'Attraction-Request_postcode',
            'Attraction-Request_entrancefee',
            'Attraction-Request_name',
            'Attraction-Request_address',
            'Attraction-Request_type',
            'Attraction-Request_area',
            'Attraction-Request_parking',
            'general-bye_none',
            'general-thank_none',
            'general-greet_none',
            'Restaurant-Inform_booktime',
            'Restaurant-Inform_bookday',
            'Restaurant-Request_ref',
            'Restaurant-Request_address',
            'Restaurant-Request_phone',
            'Restaurant-Request_pricerange',
            'Restaurant-Request_postcode',
            'Restaurant-Request_name',
            'Restaurant-Request_area',
            'Restaurant-Inform_none',
            'Restaurant-Inform_food',
            'Restaurant-Inform_pricerange',
            'Restaurant-Inform_bookpeople',
            'Restaurant-Inform_area',
            'Restaurant-Inform_name',
            'Restaurant-Request_food',
            'Hotel-Inform_none',
            'Hotel-Inform_choice',
            'Hotel-Inform_area',
            'Hotel-Inform_bookpeople',
            'Hotel-Inform_internet',
            'Hotel-Inform_bookday',
            'Hotel-Inform-bookpeople',
            'Hotel-Inform_bookstay',
            'Hotel-Inform_parking',
            'Hotel-Inform_pricerange',
            'Hotel-Inform_name',
            'Hotel-Inform_stars',
            'Hotel-Inform_type',
            'Hotel-Request_pricerange',
            'Hotel-Request_parking',
            'Hotel-Request_address',
            'Hotel-Request_name',
            'Hotel-Request_type',
            'Hospital-Inform_none',
            'Hospital-Inform_department',
            'Hospital-Request_phone',
            'Hospital-Request_name',
            'Hospital-Request_postcode',
            'Hospital-Request_address',
            'Hotel-Request_stars',
            'Hotel-Request_ref',
            'Hotel-Request_area',
            'Hotel-Request_internet',
            'Hotel-Request_phone',
            'Hotel-Request_postcode',
            'Train-Inform_none',
            'Train-Inform_day',
            'Train-Inform_departure',
            'Train-Inform_arriveby',
            'Train-Inform_leaveat',
            'Train-Inform_destination',
            'Train-Inform_bookpeople',
            'Train-Inform_price',
            'Train-Request_ref',
            'Train-Request_name',
            'Train-Request_price',
            'Taxi-Request_name',
            'Train-Request_trainid',
            'Train-Request_duration',
            'Train-Request_leaveat',
            'Train-Request_arriveby',
            'Taxi-Inform_departure',
            'Taxi-Inform_none',
            'Taxi-Inform_destination',
            'Taxi-Inform_leaveat',
            'Taxi-Inform_arriveby',
            'Taxi-Inform_bookpeople',
            'Taxi-Request_phone',
            'Taxi-Request_type',
            'Police-Inform_none',
            'Police-Request_name',
            'Police-Request_address',
            'Police-Request_phone',
            'Police-Request_postcode',
            'Police-Request_department',
        ]

        self.opt.domain_act_dict_label2idx = {
            v: k for k, v in enumerate(self.opt.domain_act_list)
        }
        self.opt.domain_act_dict_idx2label = {
            k: v for k, v in enumerate(self.opt.domain_act_list)
        }
        self.opt.entity_dict_idx2label = {
            k: v for k, v in enumerate(self.opt.entity_list)
        }
        self.opt.entity_dict_label2idx = {
            v: k for k, v in enumerate(self.opt.entity_list)
        }

        self.opt.fp16 = self.opt['fp16']
        self.opt.NULL_IDX = 0
        self.opt.START_IDX = 1
        self.opt.END_IDX = 2

    def build_decoder(self, opt, embedding, dictionary, **kwargs):
        if not self.opt['disable_classification_decoder']:
            self.classifier_decoder_1 = TransformerDecoderClassifier(opt=opt)
        else:
            self.classification_decoder = None
        if not self.opt['disable_pretrained_decoder']:
            self.decoder = TransformerDecoderGenerator(
                opt, embedding, dictionary
            )
        else:
            self.decoder = None
        #  self.classifier_decoder_1 = None
        #  self.decoder = TransformerDecoderGenerator(opt, embedding, dictionary)

    def forward(
        self,
        *xs,
        ys_dst=None,
        ys_dialog_act=None,
        prev_enc=None,
        maxlen=None,
        bsz=None,
    ):
        # this assert needs to be managed differently because we may not perform DST always with this architecture
        assert (
            ys_dst is not None
        ), "Greedy decoding in TGModel.forward no longer supported."
        self.longest_label = max(self.longest_label, ys_dst.size(1))

        # use cached encoding if available
        encoder_states = (
            prev_enc if prev_enc is not None else self.encoder(*xs)
        )

        # use teacher forcing

        # This is being done to take into account that only some decoders
        # might be enabled
        generative_model_output = None
        classifier_model_output = None

        if self.decoder is not None:
            generative_model_output = self.decoder(encoder_states, ys_dst)

        if self.classifier_decoder_1 is not None:
            classifier_model_output = self.classifier_decoder_1(
                encoder_states[0], ys_dialog_act
            )  # encoder state is a tuple, classifier needs only the first element

        global_model_output = GlobalModelOutput(
            classifier_decoder_1=classifier_model_output,
            decoder=generative_model_output,
        )
        return global_model_output


class MultitaskBartAgent(BartAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('Multi Task Bart Args')
        group.add_argument(
            '--not_load_decoder_pretrained_weights',
            type='bool',
            default=False,
            help='whether to use pre-trained weights of original BART decoder',
        )
        group.add_argument('--fp16', type=str, default=True, help='use fp16')
        parser.add_argument(
            '--disable_classification_decoder',
            type=bool,
            default=False,
            help='disable classification decoder',
        )
        parser.add_argument(
            '--disable_pretrained_decoder',
            type=bool,
            default=False,
            help='whether to use pre-trained weights of original BART decoder',
        )
        return parser

    def build_model(self) -> MultiTaskBartModel:
        """
        Build and return model.
        """
        model = MultiTaskBartModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def compute_loss(self, batch, return_output=False):
        if batch.label_vec_dst is None:
            raise ValueError('Cannot compute loss without a label.')
        global_model_output = self.model(
            *self._model_input(batch),
            ys_dst=batch.label_vec_dst,
            ys_dialog_act=batch.label_vec_dialog_act,
        )
        loss_decoders = []

        if global_model_output.classifier_decoder_1 is not None:
            output = global_model_output.classifier_decoder_1

            self.record_local_metric(
                'loss_classifier_decoder_1',
                AverageMetric.many([output.loss] * len(batch.valid_indices)),
            )
            loss_decoders.append(output.loss)

        # save loss to metrics
        #  notnull = batch.label_vec.ne(self.NULL_IDX)
        #  target_tokens = notnull.long().sum(dim=-1)
        #  correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        # cross entropy loss
        if global_model_output.decoder is not None:
            output = global_model_output.decoder
            notnull = output.notnull
            correct = ((batch.label_vec == output.preds) * notnull).sum(dim=-1)
            self.record_local_metric(
                'loss_decoder',
                AverageMetric.many(output.loss, output.target_tokens),
            )
            # perplexity
            self.record_local_metric(
                'ppl', PPLMetric.many(output.loss, output.target_tokens)
            )
            # token-wise accuracy
            self.record_local_metric(
                'token_acc', AverageMetric.many(correct, output.target_tokens)
            )
            # utterance-wise exact match
            self.record_local_metric(
                'token_em', AverageMetric.many(correct == output.target_tokens)
            )
            loss_decoder = output.loss.sum()
            loss_decoder = (loss_decoder) / (output.target_tokens.sum())
            loss_decoders.append(loss_decoder)

        loss = sum(loss_decoders)

        #  self.record_local_metric('Combined Loss', AverageMetric.many(loss_decoder))
        # actually do backwards loss
        #  loss = loss.sum()
        #  loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, global_model_output)
        else:
            return loss

    def load_state_dict(self, state_dict):
        output = self.model.load_state_dict(state_dict, strict=False)
        if len(output.unexpected_keys) > 1:
            warn_once(
                "The weights seems to have keys which cannot be loaded, this is unexpected if you want to load pre-trained weights, training will terminate now"
            )
            if self.opt['disable_pretrained_decoder']:
                warn_once(
                    "Decoder is not being loaded up with pre-trained weights"
                )
            else:
                warn_once(
                    "If training needs to be performed without loading decoder pre-trained weights, pass the --not_load_decoder_pretrained_weights"
                )
                exit(0)
        if len(output.missing_keys) > 1:
            warn_once(
                "New keys have been added to existing model weights, this is expected if you want to make modifications to model architecture which is supposed to load with pre-trained weights"
            )

    def _set_label_vec(self, obs, add_start, add_end, label_truncate):
        if "dialog_act" in obs:
            obs = super()._set_label_vec(
                obs, add_start, add_end, label_truncate
            )
            # there can be multiple dialog acts, we take the first one
            dialog_act_entry = obs["dialog_act"]
            domain_act_list, entity_list = [], []

            for domain_act_, values in dialog_act_entry.items():
                domain_act_list.append(domain_act_)
                for entity_, slot_value in values:
                    entity_list.append(entity_)

            if not entity_list:
                entity_list.append('none')

            domain_act_indices = [
                self.opt.domain_act_dict_label2idx[x] for x in domain_act_list
            ]
            entity_indices = [
                self.opt.entity_dict_label2idx[x] for x in entity_list
            ]

            domain_act_multi_hot_label = 0
            for x in domain_act_indices:
                one_hot = F.one_hot(
                    torch.tensor(x), len(self.opt.domain_act_dict_label2idx)
                )
                domain_act_multi_hot_label += one_hot

            entity_multi_hot_label = 0

            for x in entity_indices:
                one_hot = F.one_hot(
                    torch.tensor(x), len(self.opt.entity_dict_label2idx)
                )
                entity_multi_hot_label += one_hot

            if self.use_cuda:
                domain_act_multi_hot_label = domain_act_multi_hot_label.cuda()
                entity_multi_hot_label = entity_multi_hot_label.cuda()

            obs["domain_act_vec"] = domain_act_multi_hot_label
            obs["entity_vec"] = entity_multi_hot_label

        return obs

    def batchify(self, obs_batch, sort=False):
        """
        Manage dialog act labels. Adds them in Batch namedtuple
        """
        if len(obs_batch) == 0:
            return Batch(batchsize=0)

        valid_obs = [
            (i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)
        ]

        if len(valid_obs) == 0:
            return Batch(batchsize=0)

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs = x_lens = context_original_lengths = None
        context_truncate_rate = context_truncated_lengths = None
        if any(ex.get('text_vec') is not None for ex in exs):
            if any('context_original_length' in ex for ex in exs):
                context_truncate_rate = torch.LongTensor(
                    [ex.get('context_truncate_rate', 0) for ex in exs]
                )
                context_original_lengths = torch.LongTensor(
                    [ex.get('context_original_length', 0) for ex in exs]
                )
            if any('context_truncated_length' in ex for ex in exs):
                context_truncated_lengths = torch.LongTensor(
                    [ex.get('context_truncated_length', 0) for ex in exs]
                )
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = self._pad_tensor(_xs)
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = labels_avail or any(
            'eval_labels_vec' in ex for ex in exs
        )

        ys_dst = y_dst_lens = labels_dst = label_dst_original_lengths = None
        label_dst_truncate_rate = label_dst_truncated_lengths = None
        if some_labels_avail:
            if any('label_original_length' in ex for ex in exs):
                label_dst_truncate_rate = torch.LongTensor(
                    [ex.get('label_truncate_rate', 0) for ex in exs]
                )
                label_dst_original_lengths = torch.LongTensor(
                    [ex.get('label_original_length', 0) for ex in exs]
                )
            if any('label_truncated_length' in ex for ex in exs):
                label_dst_truncated_lengths = torch.LongTensor(
                    [ex.get('label_truncated_length', 0) for ex in exs]
                )
            field = 'labels' if labels_avail else 'eval_labels'

            domain_act_field = 'domain_act'
            entity_field = 'entity'

            # generative labels
            label_vecs_dst = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels_dst = [ex.get(field + '_choice') for ex in exs]

            # classifier labels
            label_vecs_domain_act = torch.stack(
                [ex.get(domain_act_field + '_vec', self.EMPTY) for ex in exs]
            )
            label_vecs_entity = torch.stack(
                [ex.get(entity_field + '_vec', self.EMPTY) for ex in exs]
            )

            labels_domain_act = [
                ex.get(domain_act_field + '_choice') for ex in exs
            ]
            labels_entity = [ex.get(entity_field + '_choice') for ex in exs]

            y_dst_lens = [y.shape[0] for y in label_vecs_dst]
            ys_dst, y_dst_lens = self._pad_tensor(
                label_vecs_dst, is_label=True
            )

            if sort and xs is None:
                (
                    ys_dst,
                    valid_inds,
                    label_vecs_dst,
                    labels_dst,
                    y_dst_lens,
                ) = argsort(
                    y_dst_lens,
                    ys_dst,
                    valid_inds,
                    label_vecs_dst,
                    labels_dst,
                    y_dst_lens,
                    descending=True,
                )

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        # reward
        rewards = None
        if any('reward' in ex for ex in exs):
            rewards = torch.Tensor([ex.get('reward', 0) for ex in exs])

        # make sure we're only passing around tensors
        valid_inds = torch.LongTensor(valid_inds)

        is_training = any('labels' in obs for obs in obs_batch)

        return MultiTaskBatch(
            batchsize=len(valid_inds),
            is_training=is_training,
            text_vec=xs,
            label_vec_dst=ys_dst,
            labels_dst=labels_dst,
            label_vec_dialog_act=[label_vecs_domain_act, label_vecs_entity],
            labels_dialog_act=[labels_domain_act, labels_entity],
            valid_indices=valid_inds,
            candidates=cands,
            candidate_vecs=cand_vecs,
            image=imgs,
            rewards=rewards,
            observations=exs if self.is_debug else None,
            _context_original_length=context_original_lengths,
            _context_truncate_rate=context_truncate_rate,
            _context_truncated_length=context_truncated_lengths,
            _label_original_length=label_dst_original_lengths,
            _label_truncate_rate=label_dst_truncate_rate,
            _label_truncated_length=label_dst_truncated_lengths,
        )

    def eval_step(self, batch):
        """
        Depending on which decoder is active, runs evaluation accordingly
        """

        # for classification
        self.model.eval()
        global_model_output = self.model(
            *self._model_input(batch),
            ys_dst=batch.label_vec_dst,
            ys_dialog_act=batch.label_vec_dialog_act,
        )  # this uses forward of BartModel
        output = Output()

        if global_model_output.classifier_decoder_1:
            model_output = global_model_output.classifier_decoder_1
            model_output = self.model.classifier_decoder_1.evaluate(
                model_output
            )
#              self.record_local_metric(
                #  'accuracy_domain_act',
                #  AverageMetric.many(
                    #  [model_output.accuracy_domain_act]
                    #  * len(batch.valid_indices)
                #  ),
            #  )
            self.record_local_metric(
                'accuracy_entity',
                AverageMetric.many(
                    [model_output.accuracy_entity] * len(batch.valid_indices)
                ),
            )
   #           self.record_local_metric(
                #  'precision_domain_act',
                #  AverageMetric.many(
                    #  [model_output.precision_domain_act]
                    #  * len(batch.valid_indices)
                #  ),
            #  )
            #  self.record_local_metric(
                #  'recall_domain_act',
                #  AverageMetric.many(
                    #  [model_output.precision_domain_act]
                    #  * len(batch.valid_indices)
                #  ),
            #  )
            #  self.record_local_metric(
                #  'f1_domain_act',
                #  AverageMetric.many(
                    #  [model_output.f1_domain_act] * len(batch.valid_indices)
                #  ),
   #           )
            self.record_local_metric(
                'precision_entity',
                AverageMetric.many(
                    [model_output.precision_entity] * len(batch.valid_indices)
                ),
            )
            self.record_local_metric(
                'recall_entity',
                AverageMetric.many(
                    [model_output.precision_entity] * len(batch.valid_indices)
                ),
            )
            self.record_local_metric(
                'f1_entity',
                AverageMetric.many(
                    [model_output.f1_entity] * len(batch.valid_indices)
                ),
            )

            #  output.literal_preds = model_output.literal_preds

        if global_model_output.decoder:
            model_output = self.model.decoder.evaluate(
                batch, global_model_output.decoder
            )
            for k, v in model_output.items():
                output[
                    k
                ] = v  # text, text_candidates, token_losses, cand_scores, beam_texts, beam_texts_token_info, text_token_info

        return output
