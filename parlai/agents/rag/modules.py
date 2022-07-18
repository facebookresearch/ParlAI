#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Modules for RAG.
"""
import torch
import torch.cuda
import torch.nn
import torch.nn.functional as F
from typing import Any, Tuple, Dict, Optional, List, Union, Type

from parlai.agents.hugging_face.t5 import (
    ParlaiT5Encoder,
    ParlaiT5Decoder,
    build_t5,
    set_device,
)
from parlai.agents.transformer.modules import (
    TransformerEncoder,
    TransformerDecoder,
    get_n_positions_from_options,
    create_embeddings,
)
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.torch import padded_tensor

from parlai.agents.rag.retrievers import retriever_factory, Document


class RagModel(TorchGeneratorModel):
    """
    RagModel.

    The RagModel operates in the following phases:
    1) retrieve:    given a tokenized query, return relevant documents
    2) expand:      given queries and documents, expand the inputs n_docs times,
                    concatenating each document with a relevant context
    3) encode:      given expanded input, encode into encoder representations
    4) decoding:    given encoder outputs, compute n_docs decoder representations for
                    each batch item.
    5) marginalize: given the decoded representations, marginalize over the documents
    appropriately.

    The RagModel overloads the `encoder` and `decoder` attributes of your standard
    `TorchGeneratorModel` to accomplish the five phases above.
    """

    def __init__(self, opt, dictionary, retriever_shared=None):
        from parlai.agents.rag.rag import RAG_MODELS

        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.fp16 = (
            not opt['no_cuda'] and torch.cuda.is_available() and opt.get('fp16', False)
        )
        self.dict = dictionary
        self.embeddings = create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        # attrs
        self.rag_model_type = opt['rag_model_type']
        self._rag_model_interface = RAG_MODELS[self.rag_model_type](opt, self.pad_idx)
        self.generation_model = opt['generation_model']
        self.n_extra_positions = opt['n_extra_positions']
        self.n_positions = get_n_positions_from_options(opt) + opt['n_extra_positions']
        assert opt['n_extra_positions'] >= 0
        self.expanded_input_truncate = min(
            opt['text_truncate'] or opt['truncate'], get_n_positions_from_options(opt)
        )
        if self.n_extra_positions > 0:
            # This attribute is overloaded.
            # when n_extra_positions == 0, it is the truncation of the full expanded input
            # when >0, it is the maximum length of the knowledge tokens.
            self.expanded_input_truncate = self.n_extra_positions
        self.min_doc_token_length = opt['min_doc_token_length']

        # modules
        self.retriever = retriever_factory(opt, dictionary, shared=retriever_shared)
        self.seq2seq_encoder = self.build_encoder(
            opt,
            dictionary=dictionary,
            embedding=self.embeddings,
            padding_idx=self.pad_idx,
        )
        self.seq2seq_decoder = self.build_decoder(
            opt,
            embedding=self.embeddings,
            dictionary=dictionary,
            padding_idx=self.pad_idx,
        )

    @classmethod
    def build_encoder(
        cls,
        opt: Opt,
        *args,
        dictionary: Optional[DictionaryAgent] = None,
        embedding: Optional[torch.nn.Embedding] = None,
        encoder_class: Optional[Type] = None,
        **kwargs,
    ):
        if encoder_class is None:
            assert dictionary is not None
            return RagEncoder(
                opt=opt, dictionary=dictionary, embedding=embedding, **kwargs
            )
        else:
            return encoder_class(opt, *args, **kwargs)

    @classmethod
    def build_decoder(
        cls,
        opt: Opt,
        *args,
        embedding: Optional[torch.nn.Embedding] = None,
        n_positions: Optional[int] = None,
        decoder_class: Optional[Type] = None,
        **kwargs,
    ):
        if decoder_class is None:
            return RagDecoder(
                opt=opt, embedding=embedding, n_positions=n_positions, **kwargs
            )
        else:
            return decoder_class(opt, *args, **kwargs)

    def tokenize_query(self, query: str) -> List[int]:
        """
        Tokenize the query for the retriever.
        """
        return self.retriever.tokenize_query(query)

    def get_retriever_delimiter(self) -> str:
        """
        Return the retriever's delimiter.
        """
        return self.retriever.get_delimiter()

    def encoder(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
        Optional[List[List[Document]]],
        Optional[torch.Tensor],
    ]:
        """
        Retrieve documents and expand input via concatenation.

        Then, encode as usual in the seq2seq encoder.

        :param input:
            2D [bsz, seqlen] input to the encoder
        :param input_lengths:
            1D [bsz] lengths of each input item
        :param query_vec:
            2D [bsz*n_turns, seqlen] input for the retriever
        :param input_turns_cnt:
            1D [bsz] number of dialogue turns for each input example

        :return (encoder_out, encoder_mask, input_turns_cnt, top_docs, top_doc_scores):
            encoder_out: encoded representations of context/document pairs
            encoder_mask: mask for enc_out
            input_turns_cnt: pass along the input turns count for the decoder
            top_docs: List of top Documents for each batch example
            top_doc_scores: scores for each retrieved document.
        """
        # Retrieve, get expanded input
        if all([tensor is not None for tensor in [input_lengths, query_vec]]):
            expanded_input, top_docs, top_doc_scores = self.retrieve_and_concat(
                input, input_lengths, query_vec, input_turns_cnt
            )
        else:
            expanded_input = input
            top_docs = top_doc_scores = None

        # Run through seq2seq encoder
        tensor, mask = self.seq2seq_encoder(
            expanded_input, positions, segments
        )  # type: ignore

        return tensor, mask, input_turns_cnt, top_docs, top_doc_scores

    def decoder(
        self,
        input: torch.LongTensor,
        encoder_state: Tuple[Any, ...],
        incr_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Decode, RAG-Style.

        Obtain decoder representations as usual, then marginalize appropriately.

        :param input:
            input for the decoder
        :param encoder_state:
            RAG encoder states
        :param incr_state:
            incremental decoder state

        :return (output, new_incr_state):
            return the output token distribution, as well as new incremental state.
        """
        # 1. Get decoder outputs
        enc_out, enc_mask, input_turns_cnt, docs, doc_scores = encoder_state
        dec_out, new_incr_state = self.seq2seq_decoder(
            input, (enc_out, enc_mask), incr_state
        )  # type: ignore
        dec_out = self.decoder_output(dec_out)

        if all([obj is not None for obj in [docs, doc_scores]]):
            # 2. Get logprobs
            n_docs = doc_scores.size(1)
            out_probs = F.log_softmax(
                dec_out, dim=-1, dtype=torch.float32  # type: ignore
            ).view(
                input.shape[0] // n_docs, n_docs, -1, dec_out.size(-1)
            )  # [bsz * beam_size, n_docs, input_len, esz]

            # 3. Marginalize
            marginalized = self._rag_model_interface.marginalize(
                out_probs, F.log_softmax(doc_scores, dim=1), input_turns_cnt
            )
        else:
            # With RAG Sequence Generation, we do not marginalize over documents.
            marginalized = dec_out

        return marginalized, new_incr_state

    def seq2seq_forward_pass(
        self, xs: torch.LongTensor, ys: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Any, ...]]:
        """
        Simulate a standard seq2seq encoder/decoder forward pass.

        Used in thorough decoding.

        :param xs:
            input tokens
        :param ys:
            teacher forced decoder outputs

        :return (logits, preds, encoder_states):
            logits: token output distribution
            preds: max probability token at each output position
            encoder_states: output states from the encoder
        """
        encoder_states = self.seq2seq_encoder(xs)  # type: ignore
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        dec_inputs = self._rag_model_interface.get_initial_forced_decoder_input(
            bsz,
            inputs,
            n_docs=1,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            input_turns_cnt=None,
        )
        latent, _ = self.seq2seq_decoder(
            dec_inputs, encoder_states, None
        )  # type: ignore
        logits = self.decoder_output(latent)
        _, preds = logits.max(dim=-1)
        return logits, preds, encoder_states

    def decoder_output(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Output layer for the decoder; maps latent state to token distributions.

        :param latent:
            final representations from last decoder layer.

        :return logits:
            return output distribution over tokens.
        """
        return F.linear(latent, self.embeddings.weight)

    def retrieve_and_concat(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, List[List[Document]], torch.Tensor]:
        """
        Retrieve documents, concat with input.

        :param input:
            2D [bsz, seqlen] input to the encoder
        :param input_lengths:
            1D [bsz] lengths of each input item
        :param query_vec:
            2D [bsz*n_turns, seqlen] input for the retriever
        :param input_turns_cnt:
            1D [bsz] number of dialogue turns for each input example

        :return (expanded_input, top_docs, top_doc_scores):
            expanded_input: [bsz * n_docs, seqlen+doc_len] tensor of context/document inputs
            top_docs: List of top documents for each input
            top_doc_scores: document scores for each document
        """
        # 1. Retrieve
        top_docs, top_doc_scores = self.retriever.retrieve(query_vec)

        # 2. Expand the input
        if input_turns_cnt is not None:
            input = input.repeat_interleave(input_turns_cnt, dim=0)  # type: ignore
            input_lengths = input_lengths.repeat_interleave(
                input_turns_cnt, dim=0
            )  # type: ignore
        expanded_input = self.concat_docs_and_input(
            input, input_lengths, top_docs, top_doc_scores.size(1)
        )

        return expanded_input, top_docs, top_doc_scores

    def concat_docs_and_input(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        top_docs: List[List[Document]],
        max_num_docs: int,
        right_padded: bool = True,
    ) -> torch.LongTensor:
        """
        Add document tokens to input tokens.

        :param input:
            original input tokens
        :param input_lengths:
            original input lengths
        :param top_docs:
            list of n_docs top documents for each input sequence
        :param max_num_docs:
            maximum number of docs out of all examples
        :param right_padded:
            whether the input is right padded.

        :return (tokens, lengths):
            return expanded token vectors & corresponding lengths
        """
        max_len = self.expanded_input_truncate
        expanded_input = []
        for i, docs in enumerate(top_docs):
            for rank in range(len(docs)):
                input_i = input[i, :]
                doc = docs[rank]
                doc_tokens = self.dict.txt2vec(doc.get_passage_str())
                if self.generation_model == 'bart' and self.n_extra_positions <= 0:
                    # move SOS to start of passage since we append question to end
                    input_i = input_i[1:]
                    sample_doc_tokens = torch.LongTensor(
                        [self.start_idx] + doc_tokens
                    ).to(input)
                else:
                    sample_doc_tokens = torch.LongTensor(doc_tokens).to(input)

                if self.n_extra_positions <= 0:
                    # Prepend document to text
                    input_i_len = input_lengths[i]
                    new_input_length = min(
                        self.expanded_input_truncate - self.min_doc_token_length,
                        input_i_len,
                    )
                    if right_padded:
                        input_i = input_i[input_i_len - new_input_length : input_i_len]
                    else:
                        input_i = input_i[input_i.size(0) - new_input_length :]

                    doc_max_len = max(max_len - len(input_i), 0)
                    sample_doc_tokens = sample_doc_tokens[:doc_max_len]
                    expanded_input.append(
                        torch.cat([sample_doc_tokens, input_i])[:max_len]
                    )
                else:
                    # Append Document to text
                    sample_doc_tokens = sample_doc_tokens[:max_len]
                    input_i_new = input_i.new(
                        self.n_positions - self.n_extra_positions
                    ).fill_(self.pad_idx)
                    input_i_new[input_i_new.size(0) - input_i.size(0) :] = input_i
                    expanded_input.append(torch.cat([input_i_new, sample_doc_tokens]))
            # append extra null inputs if there are diff # of docs per input
            expanded_input += [
                input[i, :].new(input[i, :].size()).fill_(self.pad_idx)
            ] * (max_num_docs - len(docs))
        expanded_input, _ = padded_tensor(
            expanded_input,
            fp16friendly=self.fp16 and right_padded,
            max_len=max_len if self.n_extra_positions <= 0 else None,
            pad_idx=self.pad_idx,
            left_padded=not right_padded,
        )
        expanded_input = expanded_input.to(input.device)
        return expanded_input  # type: ignore

    def output(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        RAG "output" is already scaled in RagModel.decoder.
        """
        return tensor

    def reorder_encoder_states(
        self,
        encoder_states: Tuple[torch.Tensor, ...],
        indices: Union[List[int], torch.LongTensor],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Reorder the encoder states.

        Each RAG Model type prepares encoder states for generation differently.
        """
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(
                encoder_states[0].device
            )  # type: ignore
        return self._rag_model_interface.reorder_encoder_states(encoder_states, indices)

    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[str, Any],
        inds: Union[List[int], torch.LongTensor],
    ) -> Optional[Dict[int, dict]]:
        """
        TODO: Determine how to do this
        """
        return self._rag_model_interface.reorder_decoder_incremental_state(
            incremental_state, inds, self.seq2seq_decoder
        )

    def decode_forced(
        self, encoder_states: Tuple[torch.Tensor, ...], ys: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Decode with a fixed, true sequence, computing loss.

        Override TGM.decode_forced to both:
        1) handle BART eos/bos issues, and
        2) appropriately get forced decoder input.

        :param encoder_states:
            encoder output states
        :param ys:
            teacher forced label

        :return logits, preds:
            logits: output token distribution (as logits, not probs)
            preds: tokens corresponding with max probs according to output distribution.
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any() and self.generation_model != 'bart':
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        doc_scores = encoder_states[-1]

        inputs = self._rag_model_interface.get_initial_forced_decoder_input(
            bsz,
            inputs,
            n_docs=doc_scores.size(1) if doc_scores is not None else None,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            input_turns_cnt=encoder_states[2],
        )
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=-1)
        return logits, preds  # type: ignore


class RagEncoder(TransformerEncoder):
    """
    Subclass TransformerEncoder to use additional positions if desired.
    """

    def __init__(
        self,
        opt: Opt,
        dictionary: DictionaryAgent,
        embedding: Optional[torch.nn.Embedding] = None,
        padding_idx: int = 0,
    ):
        """
        RagEncoder initialization.

        The Rag Seq2seq encoder is just a regular encoder
        """
        n_init_positions = get_n_positions_from_options(opt) + opt['n_extra_positions']
        super().__init__(
            opt=opt,
            vocabulary_size=len(dictionary),
            embedding=embedding,
            padding_idx=padding_idx,
            reduction_type='none',
            n_positions=n_init_positions,
        )


class RagDecoder(TransformerDecoder):
    """
    RagDecoder is a subclass of TransformerDecoder.

    No further modifications necessary.
    """

    pass


class T5RagModel(RagModel):
    """
    T5 For RAG.
    """

    def __init__(self, opt, dictionary, retriever_shared=None):
        opt['t5'] = build_t5(opt)
        if opt['t5_model_parallel']:
            opt['t5'].parallelize()
        else:
            opt['t5'].deparallelize()
        super().__init__(opt, dictionary, retriever_shared)
        self.embedding_size = opt['t5'].model_dim
        self.t5 = opt.pop('t5', None)
        self.paralleled = not opt['t5_model_parallel']

    @classmethod
    def build_encoder(
        cls,
        opt: Opt,
        *args,
        dictionary: Optional[DictionaryAgent] = None,
        embedding: Optional[torch.nn.Embedding] = None,
        encoder_class: Optional[Type] = None,
        **kwargs,
    ):
        return RagModel.build_encoder(
            opt,
            encoder=opt['t5'].get_encoder(),
            encoder_class=ParlaiT5Encoder,
            **kwargs,
        )

    @classmethod
    def build_decoder(
        cls,
        opt: Opt,
        *args,
        embedding: Optional[torch.nn.Embedding] = None,
        n_positions: Optional[int] = None,
        decoder_class: Optional[Type] = None,
        **kwargs,
    ):
        return RagModel.build_decoder(
            opt,
            decoder=opt['t5'].get_decoder(),
            decoder_class=ParlaiT5Decoder,
            **kwargs,
        )

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Optional[Dict[int, dict]]:
        return None

    @set_device
    def decoder_output(self, latent: torch.Tensor):
        tensor = latent * (self.t5.model_dim**-0.5)
        logits = self.t5.lm_head(tensor)
        return logits
