#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RAG Model Types.

RAG Token and RAG Sequence are outlined in https://arxiv.org/abs/2005.11401

RAG Turn is outlined in https://arxiv.org/abs/2104.07567
"""
from abc import ABC, abstractmethod
import torch
import torch.nn
import torch.nn.functional as F
import torch.cuda
from typing import Any, Dict, List, Optional, Tuple, Union


from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch
from parlai.utils.torch import padded_tensor, FP16_PAD_SIZE

from parlai.agents.rag.modules import RagDecoder, RagModel
from parlai.agents.rag.retrievers import Document


def sum_across_turns(
    tensor: torch.Tensor,
    input_turns_cnt: torch.LongTensor,
    discount: Optional[float] = None,
) -> torch.Tensor:
    """
    Sum a tensor across turns.

    :param tensor:
        input tensor, of length equivalent to total turns across all batch items
    :param input turns:
        a batch-length tensor indicating number of turns for each batch item
    :param discount:
        an optional discount factor

    :return tensor:
        return batch-length tensor with turns summed.
    """
    if tensor.dim() > 1:
        new_tensor = tensor.new(input_turns_cnt.size(0), *tensor.shape[1:])
    else:
        new_tensor = tensor.new(input_turns_cnt.size(0))
    offset = 0
    for i, it in enumerate(input_turns_cnt):
        tens = tensor[offset : offset + it]
        if discount is not None:
            tens = tens * (discount ** torch.arange(it).flip(0).to(tensor))
        new_tensor[i] = tens.sum()
        offset += it
    return new_tensor


# Reshape from [batch_size, n_docs, dims] to [batch_size * n_docs, dims]
def _stack_ctxt(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(-1, *tensor.shape[2:])


# Reshape from [batch_size * n_docs, dims] to [batch_size, n_docs, dims]
def _unstack_ctxt(tensor: torch.Tensor, n_docs: int) -> torch.Tensor:
    return tensor.view(-1, n_docs, *tensor.shape[1:])


def get_forced_decoder_inputs(
    inputs: torch.LongTensor,
    bsz: int,
    start_idx: int,
    end_idx: int,
    generation_model: str,
    start_param: Optional[torch.nn.Parameter] = None,
) -> torch.LongTensor:
    """
    Return the forced decoder inputs, depending on given parameters.

    These inputs are not formatted for RAG models.

    They merely correspond to the appropriate seq2seq_decoder input.
    """
    if generation_model == 'bart':
        tens = torch.LongTensor([end_idx, start_idx]).to(inputs).detach().expand(bsz, 2)
    elif start_param is not None:
        tens = start_param.detach().expand(bsz, 1).to(inputs)
    else:
        tens = torch.LongTensor([start_idx]).expand(inputs.size(0), 1).to(inputs)
    dec_inputs = torch.cat([tens, inputs], 1)
    return dec_inputs  # type: ignore


def fix_incremental_state(
    generation_model: str, incremental_state: Dict[int, Any]
) -> Dict[int, Any]:
    """
    Fix incremental state. Essentially takes BART into account.

    :param generation_model:
        generation model
    :param incremental_state:
        incremental decoded state
    """
    if generation_model == 'bart':
        for incr_state_l in incremental_state.values():
            assert 'self_attn' in incr_state_l
            assert 'prev_mask' in incr_state_l['self_attn']
            self_attn_mask = incr_state_l['self_attn']['prev_mask']
            # check this is on the very first run with incremental state
            if self_attn_mask.ndim == 3 and tuple(self_attn_mask.shape[1:]) == (2, 2):
                # cut off the inappropriate incremental state
                incr_state_l['self_attn']['prev_mask'] = self_attn_mask[:, -1:, :]
    elif generation_model == 't5':
        # No current solution for t5 exists.
        incremental_state = {}

    return incremental_state


class RagModelInterface(ABC):
    """
    Define an interface for the RAG Model Types.
    """

    def __init__(self, opt: Opt, null_idx: int):
        self.opt = opt
        self.generation_model = opt['generation_model']
        self.thorough = opt['thorough']
        self.n_docs = opt['n_docs']
        self.null_idx = null_idx

    ##############################
    # Input/Output Augmentations #
    ##############################
    @abstractmethod
    def get_initial_decoder_input(self, input: torch.LongTensor) -> torch.LongTensor:
        """
        Return the initial input for the decoder during generation.
        """

    @abstractmethod
    def get_initial_forced_decoder_input(
        self,
        bsz: int,
        inputs: torch.LongTensor,
        n_docs: int,
        start_idx: int,
        end_idx: int,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """
        Return the initial input to the decoder during training.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode
        :param n_docs:
            number of docs per input
        :param start_idx:
            start token idx
        :param end_idx:
            end token idx
        :param input_turns_cnt:
            an optional tensor containing the number of turns of each corresponding context.

        :return initial_input:
            initial input for the decoder.
        """

    @abstractmethod
    def set_input_turn_cnt_vec(
        self, observation: Message, model: RagModel, query_str: str
    ) -> Message:
        """
        Optionally set the input turn vec.
        """

    ############################
    # Generation Augmentations #
    ############################
    @abstractmethod
    def augment_batch_for_generation(self, batch: Batch, model: RagModel) -> Batch:
        """
        Optionally augment the batch for generation.

        :param batch:
            batch to augment
        :param model:
            model to possibly help with augmenting

        :return batch:
            return batch with appropriate augmentations.
        """

    @abstractmethod
    def rerank_beams(
        self,
        model: RagModel,
        batch: Batch,
        n_best_beam_preds_scores: List[
            List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]
        ],
    ) -> List[List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]]:
        """
        Optionally rerank beams.
        """

    @abstractmethod
    def reorder_encoder_states(
        self, encoder_states: Tuple[torch.Tensor, ...], indices: torch.LongTensor
    ) -> Tuple[
        torch.Tensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
        List[List[Document]],
        torch.Tensor,
    ]:
        """
        Reorder the encoder states, for beam search.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """

    @abstractmethod
    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[int, Any],
        inds: Union[List[int], torch.LongTensor],
        decoder: RagDecoder,
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state, for incremental decoding.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Each RagModelType will require specialized reordering, depending on the method used.
        """

    ###########################################
    # Loss Computation/Output Marginalization #
    ###########################################
    @abstractmethod
    def compute_loss(
        self,
        criterion: torch.nn.Module,
        scores: torch.Tensor,
        preds: torch.LongTensor,
        enc_state: Tuple[Any],
        label_vec: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Compute Loss.

        :param criterion:
            torch criterion module
        :param scores:
            model scores
        :param preds:
            model "predicions" of tokens
        :param enc_state:
            encoder states
        :param label_vec:
            target tokens

        :return (loss, metric_loss, correct_tokens, target_tokens):
            loss: the loss through which we backpropagate
            metric_loss: loss we use for metrics
            correct_tokens: correct predictions from the model
            target_tokens: the ground truth tokens.
        """

    @abstractmethod
    def marginalize(
        self,
        out_probs: torch.Tensor,
        doc_probs: torch.Tensor,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Marginalize over doc scores.

        :param out_probs:
            tensor of dimension [bsz, n_docs, seqlen, voc_dim]
        :param doc_probs:
            tensor of dimension [bsz, n_docs]
        :param input_turns_cnt:
            optional tensor indicating # of input turns per item.

        :return output:
            return decoder output with docs marginalized.
        """


class RagSequence(RagModelInterface):
    """
    Provides an interface for RAG-Sequence.

    RAG Sequence considers documents independently during training and inference.

    At train time, the document probability is incorporated via addition to the first token
    scores; this allows for backprop to the retriever.

    At inference time, document scores are incorporated only during final beam selection.
    """

    def get_initial_decoder_input(self, input: torch.LongTensor) -> torch.LongTensor:
        """
        Don't repeat decoder input for rag sequence.
        """
        return input

    def get_initial_forced_decoder_input(
        self,
        bsz: int,
        inputs: torch.LongTensor,
        n_docs: int,
        start_idx: int,
        end_idx: int,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """
        Return the initial input to the decoder during training.

        Repeat each input n_docs times.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode
        :param n_docs:
            number of docs per input
        :param start_idx:
            start token idx
        :param end_idx:
            end token idx
        :param input_turns_cnt:
            an optional tensor containing the number of turns of each corresponding context.

        :return initial_input:
            initial input for the decoder.
        """
        inputs = get_forced_decoder_inputs(
            inputs, bsz, start_idx, end_idx, self.generation_model
        )
        inputs = inputs.repeat(1, n_docs).reshape(-1, inputs.size(1))  # type: ignore
        return inputs

    def get_generation_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        Optional[torch.LongTensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        For RAG Sequence, we retrieve prior to generation.
        """
        assert batch.text_vec is not None
        return (batch.text_vec, None, None, None)

    def reorder_encoder_states(
        self, encoder_states: Tuple[torch.Tensor, ...], indices: torch.LongTensor
    ) -> Tuple[
        torch.Tensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
        List[List[Document]],
        torch.Tensor,
    ]:
        """
        Reorder the encoder states.

        For RAG Sequence, only reorder enc_out and mask.
        """
        enc, mask, *_ = encoder_states
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask, None, None, None  # type: ignore

    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[int, Any],
        inds: Union[List[int], torch.LongTensor],
        decoder: RagDecoder,
    ) -> Dict[int, dict]:
        """
        For RAG Sequence, each doc/context pair is it's own batch item.

        So, we can simply reorder normally.
        """
        assert incremental_state is not None
        incremental_state = fix_incremental_state(
            self.generation_model, incremental_state
        )
        if not incremental_state:
            return incremental_state
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(decoder.layers)
        }

    def get_ctxt_index(self, batch: Batch, batch_idx: int) -> int:
        """
        Map the batch_idx back to the appropriate batch item during generation.

        :param batch:
            batch being considered
        :param batch_idx:
            idx into batch.text_vec

        :return mapped_idx:
            return the appropriate batch index
        """
        n_docs = batch.doc_log_probs.size(1)  # type: ignore
        return batch_idx // n_docs

    def rerank_beams(
        self,
        model: RagModel,
        batch: Batch,
        n_best_beam_preds_scores: List[
            List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]
        ],
    ) -> List[List[Tuple[torch.LongTensor, Optional[Dict]]]]:
        """
        Rerank beams in RAG-Sequence, accounting for document probabilities as well.

        Fast decoding combines probabilities of equivalently generated beams across
        different documents.

        Thorough decoding rescores the beam outputs with another forward pass.

        Iteration for fast decoding is as follows:
            for each example x_i:
                for each document d_i:
                    marginalize probability over
                    gen. beams. [b_i, ..., b_n] for d_i

        Iteration for thorough decoding is as follows:
            for each example x_i:
                obtain unique hypotheses
                rescore with model, xs=text_vec, ys=hyps
                sort by lowest loss (highest score)

        Thorough decoding impl. verified via OSS impl:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/rag/modeling_rag.py#L954

        :param batch:
            current batch
        :param n_best_beam_preds_scores:
            bsz-length list of Tuples of predictions and scores

        :return List((pred, score)):
            return a re-ranked version of the n_best_beam_preds_scores
        """
        new_n_best: List[List[List[torch.LongTensor]]] = []
        doc_log_probs = batch.doc_log_probs  # type: ignore
        bsz, n_docs = doc_log_probs.shape
        for i in range(bsz):
            if self.thorough:
                hyps = [
                    hyp[0]
                    for h in n_best_beam_preds_scores[i * n_docs : (i + 1) * n_docs]
                    for hyp in h
                ]
                sorted_by_score = self.thorough_generation(
                    hyps,
                    batch.src_text_vec[i : i + 1],
                    self.null_idx,
                    model,  # type: ignore
                )
            else:
                # mapping from tokens to (tokens, score)
                sorted_by_score = self.fast_generation(
                    list(range(i * n_docs, (i + 1) * n_docs)),
                    n_best_beam_preds_scores,
                    doc_log_probs[i],
                    n_docs,
                )
            new_n_best.append(sorted_by_score)  # type: ignore

        return new_n_best

    def augment_batch_for_generation(self, batch: Batch, model: RagModel) -> Batch:
        """
        Augment batch for generation.

        For RAG Sequence, we retrieve prior to generation, as we do not consider the
        document probabilities until after generating all of the beams.

        :param batch:
            batch to augment
        :param model:
            model to possibly help with augmenting

        :return batch:
            return batch with text vec swapped out.
        """
        (expanded_input, _, doc_scores) = model.retrieve_and_concat(
            batch.text_vec,
            batch.text_vec.ne(self.null_idx).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
        )
        doc_log_probs = F.log_softmax(doc_scores, dim=1)
        batch.src_text_vec = batch.text_vec
        batch.text_vec = expanded_input
        batch.doc_log_probs = doc_log_probs
        batch.batchsize = batch.text_vec.size(0)

        return batch

    @classmethod
    def fast_generation(
        cls,
        doc_indices: List[int],
        n_best_beam_preds_scores: List[
            List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]
        ],
        doc_log_probs: torch.Tensor,
        n_docs: int,
    ):
        """
        Apply RAG Sequence fast decoding, for a single batch item.

        :param doc_indices:
            indices into beam preds scores
        :param n_best_beam_preds_scores:
            bsz-length list of Tuples of predictions and scores
        :param doc_log_probs:
            probabilities for each document
        :param n_docs:
            number of docs per example

        :return sorted_hyps:
            return list of (hyp, score, token metadata) tuples, sorted by their score.
        """
        marginalized_hypos: Dict[
            str, Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]
        ] = {}
        for doc_idx in doc_indices:
            doc_hypos = n_best_beam_preds_scores[doc_idx]
            doc_score = doc_log_probs[doc_idx % n_docs]
            for hypo, hypo_score, token_metadata in doc_hypos:
                score = hypo_score + doc_score
                hypo_tokens = str(hypo.tolist())
                if hypo_tokens in marginalized_hypos:
                    marginalised_hypo = marginalized_hypos[hypo_tokens]
                    marginalised_hypo = (
                        marginalised_hypo[0],
                        torch.log(marginalised_hypo[1].exp() + score.exp()),
                        marginalised_hypo[2],
                    )
                else:
                    marginalized_hypos[hypo_tokens] = (hypo, score, token_metadata)
        sorted_by_score = sorted(marginalized_hypos.values(), key=lambda h: -h[1])
        return sorted_by_score

    @classmethod
    def thorough_generation(
        cls,
        hyps: List[torch.LongTensor],
        new_input: torch.LongTensor,
        null_idx: int,
        model: RagModel,
    ) -> List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]:
        """
        Apply RAG-sequence thorough generation for a single batch item.

        Recomputes model scores with given hypotheses, sorts accordingly.

        :param hyps:
            list of candidate hypotheses
        :param new_input:
            input for the model

        :return sorted_hyps:
            return list of (hyp, score, token_metadata) tuples, sorted by their score.
        """
        # deduplicate, exclude BOS Token
        hyps = list({str(h.tolist()): h[1:] for h in hyps}.values())  # type: ignore
        new_input = new_input.repeat(len(hyps), 1)  # type: ignore
        new_ys, _ = padded_tensor(
            hyps, fp16friendly=new_input.size(1) % FP16_PAD_SIZE == 0, pad_idx=null_idx
        )
        new_ys = new_ys.to(new_input.device)
        scores, *_ = model.seq2seq_forward_pass(new_input, new_ys)
        loss = cls._rag_sequence_loss(
            new_ys.unsqueeze(1).unsqueeze(-1), scores.unsqueeze(1), null_idx
        )  # type: ignore
        sorted_by_score = [
            (hyps[idx], loss[idx], None) for idx in loss.sort()[-1]
        ]  # sort ascending
        return sorted_by_score

    @classmethod
    def _rag_sequence_loss(
        cls, target: torch.LongTensor, scores: torch.Tensor, null_idx: int
    ) -> torch.Tensor:
        """
        RAG Sequence loss.

        :param target:
            target tokens
        :param scores:
            model log probs
        :param null_idx:
            padding index to ignore for loss computation

        :return loss:
            return NLL Loss
        """
        ll = scores.gather(dim=-1, index=target)
        pad_mask = target.eq(null_idx)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
        ll = ll.squeeze(-1)
        ll = ll.sum(2)  # sum over tokens
        ll = ll.logsumexp(1)  # sum over docs
        nll_loss = -ll
        loss = nll_loss

        return loss

    def set_input_turn_cnt_vec(
        self, observation: Message, model: RagModel, query_str: str
    ) -> Message:
        """
        Ignore for RAG Sequence / Token.
        """
        return observation

    def compute_loss(
        self,
        criterion: torch.nn.Module,
        scores: torch.Tensor,
        preds: torch.LongTensor,
        enc_state: Tuple[Any],
        label_vec: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Compute RAG Sequence Loss.

        :param criterion:
            torch criterion module
        :param scores:
            model scores
        :param preds:
            model "predicions" of tokens
        :param enc_state:
            encoder states
        :param label_vec:
            target tokens

        :return (loss, metric_loss, correct_tokens, target_tokens):
            loss: the loss through which we backpropagate
            metric_loss: loss we use for metrics
            correct_tokens: correct predictions from the model
            target_tokens: the ground truth tokens.
        """
        if scores.size(2) != label_vec.size(1):
            assert self.generation_model == 'bart'
            # ignore start
            scores = scores[:, :, 1:, :]
            preds = preds[:, 1:]  # type: ignore

        # compute rag sequence loss
        seq_preds = scores.max(-1)[-1]
        bsz = scores.size(0)
        n_docs = scores.size(1)
        target = label_vec.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        loss = self._rag_sequence_loss(target, scores, self.null_idx)  # type: ignore

        # compute relevant metric counters
        metric_loss = loss.tolist()
        notnull = label_vec.ne(self.null_idx)
        target_tokens = notnull.long().sum(dim=-1)

        shape = (bsz, n_docs, -1)
        notnull_seq = target.ne(self.null_idx)
        correct = (
            (target.view(*shape) == seq_preds.view(*shape)) * notnull_seq.view(*shape)
        ).sum(dim=-1)
        correct_mean = correct.float().mean(dim=1)

        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        return loss, metric_loss, correct_mean, target_tokens

    def marginalize(
        self,
        out_probs: torch.Tensor,
        doc_probs: torch.Tensor,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        RAG Sequence marginalization over doc scores.

        RAG Sequence simply adds the doc scores to the first token probabilities.

        :param out_probs:
            tensor of dimension [bsz, n_docs, seqlen, voc_dim]
        :param doc_probs:
            tensor of dimension [bsz, n_docs]
        :param input_turns_cnt:
            optional tensor indicating # of input turns per item.

        :return output:
            return decoder output with docs marginalized.
        """
        doc_probs = doc_probs.unsqueeze(-1).unsqueeze(-1)
        first_token_scores = out_probs[:, :, :1, :]
        if self.generation_model == 'bart':
            # bypass first token in BART
            second_token_scores = out_probs[:, :, 1:2, :]
            remainder = out_probs[:, :, 2:, :]
            output = torch.cat(
                [first_token_scores, second_token_scores + doc_probs, remainder], dim=2
            )
        else:
            remainder = out_probs[:, :, 1:, :]
            output = torch.cat([first_token_scores + doc_probs, remainder], dim=2)
        return output


class RagToken(RagModelInterface):
    """
    Provides an interface for RAG-Token.

    RAG Token considers documents jointly during training and inference; output
    distributions are computed via summing across latent document probabilities.
    """

    def get_initial_decoder_input(self, input: torch.LongTensor) -> torch.LongTensor:
        """
        Repeat the decoder input accordingly.
        """
        return input.repeat(1, self.n_docs).reshape(-1, input.size(1))  # type: ignore

    def get_initial_forced_decoder_input(
        self,
        bsz: int,
        inputs: torch.LongTensor,
        n_docs: int,
        start_idx: int,
        end_idx: int,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """
        Return the initial input to the decoder during training.

        Repeat inputs n_docs times.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode
        :param n_docs:
            number of docs per input
        :param start_idx:
            start token idx
        :param end_idx:
            end token idx
        :param input_turns_cnt:
            an optional tensor containing the number of turns of each corresponding context.

        :return initial_input:
            initial input for the decoder.
        """
        inputs = get_forced_decoder_inputs(
            inputs, bsz, start_idx, end_idx, self.generation_model
        )
        inputs = inputs.repeat(1, n_docs).reshape(-1, inputs.size(1))  # type: ignore
        return inputs

    def get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Repeat decoder input accordingly for RAG Token.

        RAG Token marginalizes over all the documents each turn; this means that during generation,
        although the input will be of length [n_docs*bsz], the output will be of length [bsz].

        Thus, we need to make sure that we repeat the beam selection n_docs times and continue to
        marginalize over documents accordingly.

        :param prev_input:
            previous [bsz*n_docs, seqlen] input to the decoder.
        :param selection:
            [bsz]-length beam selection ids for the decoder
        :param incr_state_inds:
            beam indices that are continuing in next generation.

        :return dec_input:
            return the decoder input with appropriate repeated selections.
        """
        prev_input = _unstack_ctxt(prev_input, self.n_docs).index_select(
            0, incr_state_inds
        )  # type: ignore
        dec_input = torch.cat(
            [prev_input, selection.repeat_interleave(self.n_docs, 1).unsqueeze(-1)],
            dim=-1,
        )
        dec_input = _stack_ctxt(dec_input)
        return dec_input  # type: ignore

    def reorder_encoder_states(
        self, encoder_states: Tuple[torch.Tensor, ...], indices: torch.LongTensor
    ) -> Tuple[
        torch.Tensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
        List[List[Document]],
        torch.Tensor,
    ]:
        """
        Reorder all RAG encoder states.
        """
        enc, mask, input_turns_cnt, docs, doc_probs = encoder_states
        n_docs = doc_probs.shape[1]
        enc = _stack_ctxt(_unstack_ctxt(enc, n_docs).index_select(0, indices))
        mask = _stack_ctxt(_unstack_ctxt(mask, n_docs).index_select(0, indices))
        doc_probs = doc_probs.index_select(0, indices)

        return enc, mask, input_turns_cnt, docs, doc_probs  # type: ignore

    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[int, Any],
        inds: Union[List[int], torch.LongTensor],
        decoder: RagDecoder,
    ) -> Dict[int, dict]:
        """
        For RAG Token, we send each decoder input through n_docs times.

        Similarly to reordering the encoder states, we need to reorder according to the
        documents dimensions.
        """
        assert incremental_state is not None
        incremental_state = fix_incremental_state(
            self.generation_model, incremental_state
        )
        if not incremental_state:
            return incremental_state
        for incr_state_l in incremental_state.values():
            for key in incr_state_l:
                for sub_key in incr_state_l[key]:
                    incr_state_l[key][sub_key] = _unstack_ctxt(
                        incr_state_l[key][sub_key], self.n_docs
                    )

        new_incr_state = {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(decoder.layers)
        }

        for incr_state_l in new_incr_state.values():
            for key in incr_state_l:
                for sub_key in incr_state_l[key]:
                    incr_state_l[key][sub_key] = _stack_ctxt(incr_state_l[key][sub_key])

        return new_incr_state

    def rerank_beams(
        self,
        model: RagModel,
        batch: Batch,
        n_best_beam_preds_scores: List[
            List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]
        ],
    ) -> List[List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]]:
        """
        We don't re-rank beams for RAG Token.
        """
        return n_best_beam_preds_scores

    def augment_batch_for_generation(self, batch: Batch, model: RagModel) -> Batch:
        """
        No augmentation for RAG Token.
        """
        return batch

    def set_input_turn_cnt_vec(
        self, observation: Message, model: RagModel, query_str: str
    ) -> Message:
        """
        Ignore for RAG Sequence / Token.
        """
        return observation

    def compute_loss(
        self,
        criterion: torch.nn.Module,
        scores: torch.Tensor,
        preds: torch.LongTensor,
        enc_state: Tuple[Any],
        label_vec: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Compute RAG Token Loss.

        This is a simple NLL Loss.

        :param criterion:
            presumably the NLL criterion.
        :param scores:
            model scores
        :param preds:
            model "predicions" of tokens
        :param enc_state:
            encoder states
        :param label_vec:
            target tokens

        :return (loss, metric_loss, correct_tokens, target_tokens):
            loss: the loss through which we backpropagate
            metric_loss: loss we use for metrics
            correct_tokens: correct predictions from the model
            target_tokens: the ground truth tokens.
        """
        if scores.size(1) != label_vec.size(1):
            assert self.generation_model == 'bart'
            # ignore start
            scores = scores[:, 1:, :]
            preds = preds[:, 1:]  # type: ignore

        # compute loss
        score_view = scores.reshape(-1, scores.size(-1))
        loss = criterion(score_view, label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)

        # calculate metric counters
        metric_loss = loss.tolist()
        notnull = label_vec.ne(self.null_idx)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((label_vec == preds) * notnull).sum(dim=-1)

        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        return loss, metric_loss, correct, target_tokens

    def marginalize(
        self,
        out_probs: torch.Tensor,
        doc_probs: torch.Tensor,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        RAG Token marginalization.

        RAG Token sums token probabilities across documents, accounting for each document
        in its final output distribution.

        :param out_probs:
            tensor of dimension [bsz, n_docs, seqlen, voc_dim]
        :param doc_probs:
            tensor of dimension [bsz, n_docs]
        :param input_turns_cnt:
            optional tensor indicating # of input turns per item.

        :return tensor:
            return decoder output with docs marginalized.
        """
        log_prob_sum = out_probs + doc_probs.unsqueeze(-1).unsqueeze(-1)
        output = torch.logsumexp(log_prob_sum, dim=1)  # sum across documents
        return output


class RagTurn(RagModelInterface):
    """
    Provides an interface for RAG-Turn.

    RAG Turn performs retrieval over each turn of dialogue, and marginalizes
    over n_turns context/document pairs for each batch item.

    RAG Turn Doc-Then-Turn marginalizes over documents, then over turns.

    RAG Turn Doc Only considers each turn's documents independently, and thus
    resembles RAG Token on a doc level, and RAG Sequence on a turn level.
    """

    def __init__(self, opt: Opt, null_idx: int):
        super().__init__(opt, null_idx)
        self.turn_marginalize = opt['rag_turn_marginalize']
        self.n_turns = opt['rag_turn_n_turns']
        assert 0 < opt['rag_turn_discount_factor'] <= 1.0
        self.discount_factor = opt['rag_turn_discount_factor']

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        rag_turn_group = parser.add_argument_group('RAG-Turn Args')
        rag_turn_group.add_argument(
            '--rag-turn-n-turns',
            type=int,
            default=2,
            help='how many turns to split up retrieval into. '
            'The most recent text is split by delimiter; all turns after (n-1)th turn '
            'are combined.',
        )
        rag_turn_group.add_argument(
            '--rag-turn-marginalize',
            type=str,
            default='doc_then_turn',
            choices=['doc_only', 'doc_then_turn'],
            help='how to marginalize rag-turn. ',
        )
        rag_turn_group.add_argument(
            '--rag-turn-discount-factor',
            type=float,
            default=1.0,
            help='discount factor for turns beyond most recent one. We employ exponential discounting. '
            'Only considered if 0 < factor < 1.0. ',
        )
        return parser

    def get_initial_decoder_input(self, input: torch.LongTensor) -> torch.LongTensor:
        """
        Repeat the decoder input accordingly.
        """
        return input.repeat(1, self.n_docs).reshape(-1, input.size(1))  # type: ignore

    def get_initial_forced_decoder_input(
        self,
        bsz: int,
        inputs: torch.LongTensor,
        n_docs: int,
        start_idx: int,
        end_idx: int,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """
        Return the initial input to the decoder during training.

        Repeat inputs n_docs * n_turns times.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode
        :param n_docs:
            number of docs per input
        :param start_idx:
            start token idx
        :param end_idx:
            end token idx
        :param input_turns_cnt:
            an optional tensor containing the number of turns of each corresponding context.

        :return initial_input:
            initial input for the decoder.
        """
        if input_turns_cnt is not None:
            inputs = inputs.repeat_interleave(input_turns_cnt, dim=0)  # type: ignore
            bsz = input_turns_cnt.sum()  # type: ignore
        inputs = get_forced_decoder_inputs(
            inputs, bsz, start_idx, end_idx, self.generation_model
        )
        inputs = inputs.repeat(1, n_docs).reshape(-1, inputs.size(1))  # type: ignore
        return inputs

    def get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Repeat decoder input accordingly for RAG Token.

        RAG Turn marginalizes over all the documents within turns; this means that during generation,
        although the input will be of length [n_docs*n_turns*bsz], the output will be of length [n_turns*bsz].

        Thus, we need to make sure that we repeat the beam selection n_docs times and continue to
        marginalize over documents accordingly.

        :param prev_input:
            previous [bsz*n_docs*n_turns, seqlen] input to the decoder.
        :param selection:
            [bsz*n_turns]-length beam selection ids for the decoder
        :param incr_state_inds:
            beam indices that are continuing in next generation.

        :return dec_input:
            return the decoder input with appropriate repeated selections.
        """
        prev_input = _unstack_ctxt(prev_input, self.n_docs).index_select(
            0, incr_state_inds
        )  # type: ignore
        dec_input = torch.cat(
            [prev_input, selection.repeat_interleave(self.n_docs, 1).unsqueeze(-1)],
            dim=-1,
        )
        dec_input = _stack_ctxt(dec_input)

        return dec_input  # type: ignore

    def get_ctxt_index(self, batch: Batch, batch_idx: int) -> int:
        """
        Map the batch_idx back to the appropriate batch item during generation.

        For RAG Turn Doc-Only, we have n_turns*bsz batch items.
        This means we need to map back to the appropriate context idx.

        :param batch:
            batch
        :param batch_idx:
            original batch idx

        :return new_batch_idx:
            return mapped batch idx
        """
        if self.turn_marginalize == 'doc_only':
            assert hasattr(batch, 'input_turns_cnt')
            offset = 0
            mapping = {}
            for i, it in enumerate(batch.input_turns_cnt):
                mapping.update({j + offset: i for j in range(it)})
                offset += it.item()
            batch_idx = mapping[batch_idx]

        return batch_idx

    def reorder_encoder_states(
        self, encoder_states: Tuple[torch.Tensor, ...], indices: torch.LongTensor
    ) -> Tuple[
        torch.Tensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
        List[List[Document]],
        torch.Tensor,
    ]:
        """
        Reorder the encoder states.

        For RAG Turn Doc-Then-Turn, we need to repeat the indices n_turns times.
        """
        enc, mask, input_turns_cnt, docs, doc_probs = encoder_states

        if self.turn_marginalize == 'doc_then_turn':
            n_inputs = input_turns_cnt.size(0)
            old_inds = indices.clone()
            indices = (
                indices.view(n_inputs, -1)
                .repeat_interleave(input_turns_cnt, dim=0)
                .view(-1)
            )  # type: ignore
            input_turns_cnt = input_turns_cnt.index_select(0, old_inds)

        n_docs = doc_probs.shape[1]
        enc = _stack_ctxt(_unstack_ctxt(enc, n_docs).index_select(0, indices))
        mask = _stack_ctxt(_unstack_ctxt(mask, n_docs).index_select(0, indices))
        doc_probs = doc_probs.index_select(0, indices)

        return enc, mask, input_turns_cnt, docs, doc_probs  # type: ignore

    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[int, Any],
        inds: Union[List[int], torch.LongTensor],
        decoder: RagDecoder,
    ) -> Dict[int, dict]:
        """
        Unsupported for Rag Turn.
        """
        return None  # type: ignore

    def rerank_beams(
        self,
        model: RagModel,
        batch: Batch,
        n_best_beam_preds_scores: List[
            List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]
        ],
    ) -> List[List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]]:
        """
        Re-rank beams.

        We only re-rank beams in the Doc Only marginalization scheme, as we
        generate beams for each turn.

        Fast decoding is similar to RAG Sequence fast decoding, except we simply
        rank according to beam score; one can additionally apply a discount factor
        to beams generated based on documents retrieved given prior dialogue context.

        Thorough decoding is identical RAG Sequence.
        """
        new_n_best: List[
            List[Tuple[torch.LongTensor, torch.Tensor, Optional[Dict]]]
        ] = []
        if self.turn_marginalize == 'doc_only' and not self.thorough:
            # no doc log probs here; just re-sorting beams
            input_turns_cnt = batch.input_turns_cnt
            offset = 0
            for it in input_turns_cnt:
                n_best_i = []
                for i, turn in enumerate(
                    n_best_beam_preds_scores[offset : offset + it]
                ):
                    for beam in turn:
                        new_beam = (
                            beam[0],
                            beam[1] * self.discount_factor ** (it - i - 1),
                            beam[2],
                        )
                        n_best_i.append(new_beam)
                new_n_best.append(sorted(n_best_i, key=lambda x: -x[1]))
                offset += it
        elif self.turn_marginalize == 'doc_only' and self.thorough:
            hyps = [hyp[0] for h in n_best_beam_preds_scores for hyp in h]
            sorted_by_score = RagSequence.thorough_generation(
                hyps, batch.src_text_vec[0:1], self.null_idx, model
            )
            new_n_best.append(sorted_by_score)
        elif batch.batchsize > 1:
            raise RuntimeError(
                'Please set batchsize to 1 when evaluating RAG Turn Doc-Then-Turn Models'
            )
        else:
            new_n_best = n_best_beam_preds_scores

        return new_n_best

    def augment_batch_for_generation(self, batch: Batch, model: RagModel) -> Batch:
        """
        Augment batch for doc_only turn marginalization.

        src_text_vec and input_turns_cnt are each used during beam re-ranking;
        setting batch.batchsize lets this interact nicely with TGA._generate.

        :param batch:
            batch to augment
        :param model:
            model to possibly help with augmenting

        :return batch:
            return batch with appropriate augmentations.
        """
        if self.turn_marginalize == 'doc_only':
            input_turns_cnt = batch.input_turn_cnt_vec
            batch.batchsize = input_turns_cnt.sum().item()
            batch.src_text_vec = batch.text_vec
            batch.input_turns_cnt = input_turns_cnt
        return batch

    def set_input_turn_cnt_vec(
        self, observation: Message, model: RagModel, query_str: str
    ) -> Message:
        """
        Compute the number of turns of input, and set the vec accordingly.

        :param observation:
            observation in which to set the vec
        :param model:
            model provided for access to retriever tokenizer
        :param query_str:
            the query string for computation of the input turns.

        :return observation:
            return the observation with the input turn vec set appropriately.
        """
        delimiter = model.get_retriever_delimiter()
        split_text_raw = query_str.split(delimiter)
        split_text: List[str] = []
        if self.n_turns > 1 and len(split_text_raw) > self.n_turns:
            end_off = self.n_turns - 1
            split_text = [delimiter.join(split_text_raw[:-end_off])] + split_text_raw[
                -end_off:
            ]
        else:
            split_text = split_text_raw

        input_turns_cnt = torch.LongTensor([len(split_text)])
        query_vecs = [model.tokenize_query(q) for q in split_text]
        # Override query vec
        observation.force_set('query_vec', query_vecs)
        observation['input_turn_cnt_vec'] = input_turns_cnt
        return observation

    def compute_loss(
        self,
        criterion: torch.nn.Module,
        scores: torch.Tensor,
        preds: torch.LongTensor,
        enc_state: Tuple[Any, ...],
        label_vec: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Compute Loss for Rag Turn.

        RAG Turn Doc-Then-Turn computes loss with a normal NLL Loss
        (everything is marginalized beforehand)

        RAG Turn Doc-Only computes loss for each input turn; this loss can be
        weighted with a discount factor, applying less weight to prior turns (only for
        backpropagation purposes).

        :param criterion:
            torch criterion module
        :param scores:
            model scores
        :param preds:
            model "predicions" of tokens
        :param enc_state:
            encoder states
        :param label_vec:
            target tokens

        :return (loss, metric_loss, correct_tokens, target_tokens):
            loss: the loss through which we backpropagate
            metric_loss: loss we use for metrics
            correct_tokens: correct predictions from the model
            target_tokens: the ground truth tokens.
        """
        if scores.size(1) != label_vec.size(1):
            # ignore start
            scores = scores[:, 1:, :]
            preds = preds[:, 1:]  # type: ignore

        input_turns_cnt = enc_state[2]
        real_bsz = label_vec.size(0)
        resize_label = real_bsz != scores.size(0)
        if resize_label:
            assert self.turn_marginalize == 'doc_only'
            label_vec = label_vec.repeat_interleave(
                input_turns_cnt, dim=0
            )  # type: ignore

        # compute loss
        score_view = scores.reshape(-1, scores.size(-1))
        loss = criterion(score_view, label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        metric_loss = loss.tolist()

        if resize_label:
            assert self.turn_marginalize == 'doc_only'
            loss = sum_across_turns(
                loss, input_turns_cnt, discount=self.discount_factor
            )
            metric_loss = sum_across_turns(loss, input_turns_cnt).tolist()

        # compute metric counters
        notnull = label_vec.ne(self.null_idx)
        target_tokens = metric_target_tokens = notnull.long().sum(dim=-1)
        correct = metric_correct = ((label_vec == preds) * notnull).sum(dim=-1)
        if resize_label:
            metric_target_tokens = sum_across_turns(target_tokens, input_turns_cnt)
            metric_correct = sum_across_turns(correct, input_turns_cnt)

        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        return loss, metric_loss, metric_correct, metric_target_tokens

    def marginalize(
        self,
        out_probs: torch.Tensor,
        doc_probs: torch.Tensor,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        RAG Turn Marginalize over doc scores.

        For Doc-Then-Turn, we marginalize over documents first, then over turns; one can
        optionally apply an exponential discount factor to probabilities from the past.

        For Doc-Only, we only marginalize over documents; turns are considered
        independently.

        :param out_probs:
            tensor of dimension [bsz, n_docs, seqlen, voc_dim]
        :param doc_probs:
            tensor of dimension [bsz, n_docs]
        :param input_turns_cnt:
            optional tensor indicating # of input turns per item.

        :return tensor:
            return decoder output with docs marginalized.
        """
        if self.turn_marginalize == 'doc_then_turn' and out_probs.size(
            0
        ) != doc_probs.size(0):
            # Need to adjust out probs during generation
            out_probs = out_probs.repeat_interleave(
                input_turns_cnt, dim=0
            )  # type: ignore

        log_prob_sum = out_probs + doc_probs.unsqueeze(-1).unsqueeze(-1)
        if self.turn_marginalize == 'doc_only':
            output = torch.logsumexp(log_prob_sum, dim=1)  # sum across documents only
        else:
            turns = []
            offset = 0
            for it in input_turns_cnt:
                turns.append(
                    log_prob_sum[offset : offset + it].logsumexp(dim=1, keepdim=False)
                )
                offset += it
            output = torch.stack(
                [
                    (
                        t
                        * self.discount_factor
                        ** torch.arange(t.size(0)).flip(0).to(t).view(t.size(0), 1, 1)
                    ).logsumexp(dim=0)
                    for i, t in enumerate(turns)
                ]
            )
        return output
