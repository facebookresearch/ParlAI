#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.

Original Paper: https://arxiv.org/abs/2005.11401

As used in ParlAI: https://arxiv.org/abs/2104.07567
"""
from abc import ABC, abstractstaticmethod
import os
import torch
import torch.nn
import torch.cuda
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from parlai.agents.bart.bart import BartAgent
import parlai.agents.hugging_face.hugging_face  # noqa: F401
from parlai.agents.hugging_face.t5 import T5Agent
from parlai.agents.transformer.polyencoder import PolyencoderAgent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, normalize_answer, F1Metric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import History, Batch
from parlai.core.torch_generator_agent import PPLMetric, TorchGeneratorAgent, TreeSearch
from parlai.utils.distributed import sync_parameters
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
from parlai.utils.misc import recursive_getattr
import parlai.utils.pickle
from parlai.utils.torch import total_parameters, trainable_parameters, PipelineHelper
from parlai.utils.typing import TShared

from parlai.agents.rag.args import setup_rag_args
from parlai.agents.rag.model_types import (
    RagModelInterface,
    RagSequence,
    RagToken,
    RagTurn,
)
from parlai.agents.rag.modules import RagModel, T5RagModel
from parlai.agents.rag.retrievers import Document


class BaseGenerationAgentMixin(ABC):
    """
    A Base Generation Agent Mixin.
    """

    @abstractstaticmethod
    def build_rag_model(opt: Opt, dictionary: DictionaryAgent) -> RagModel:
        """
        Build and return a RAG Model.
        """


class TransformerGeneratorRagAgent(TransformerGeneratorAgent, BaseGenerationAgentMixin):
    @staticmethod
    def build_rag_model(opt: Opt, dictionary: DictionaryAgent) -> RagModel:
        return RagModel(opt, dictionary)


class BartRagAgent(BartAgent, BaseGenerationAgentMixin):
    @staticmethod
    def build_rag_model(opt: Opt, dictionary: DictionaryAgent) -> RagModel:
        return RagModel(opt, dictionary)


class T5RagAgent(T5Agent, BaseGenerationAgentMixin):
    @staticmethod
    def build_rag_model(opt: Opt, dictionary: DictionaryAgent) -> T5RagModel:
        return T5RagModel(opt, dictionary)

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ) -> Tuple[List[Tuple[torch.LongTensor, torch.Tensor]], List[TreeSearch]]:
        """
        Override since T5 needs to call TGA generate.
        """
        return TorchGeneratorAgent._generate(
            self, batch, beam_size, max_ts, prefix_tokens
        )


GENERATION_AGENTS = {
    'transformer/generator': TransformerGeneratorRagAgent,
    'bart': BartRagAgent,
    't5': T5RagAgent,
}

RAG_MODELS = {'sequence': RagSequence, 'token': RagToken, 'turn': RagTurn}


class RagAgent(TransformerGeneratorRagAgent, BartRagAgent, T5RagAgent):
    """
    RagAgent.

    The RAG Agent interacts with the RAG model mostly via it's RAG Model interface.
    """

    _generation_agent: Union[
        Type[T5RagAgent], Type[BartRagAgent], Type[TransformerGeneratorRagAgent]
    ]
    _rag_model_interface: RagModelInterface

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add RAG Args.
        """
        PolyencoderAgent.add_cmdline_args(parser, partial_opt=None)
        TransformerGeneratorRagAgent.add_cmdline_args(parser, partial_opt)
        parser = setup_rag_args(parser)
        RagTurn.add_cmdline_args(parser, partial_opt)
        if partial_opt and partial_opt.get('generation_model') == 'bart':
            BartRagAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        T5RagAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        # BART Agent sets these to True; doesn't let you set anything else
        parser.set_defaults(fp16=False, force_fp16_tokens=False)
        return parser

    @property
    def generation_model(self) -> str:
        return self._generation_model

    @generation_model.setter
    def generation_model(self, model: str):
        self._generation_model = model
        self._generation_agent = GENERATION_AGENTS[model]

    @property
    def rag_model_type(self) -> str:
        return self._rag_model_type

    @rag_model_type.setter
    def rag_model_type(self, model: str):
        self._rag_model_type = model
        self._rag_model_interface = RAG_MODELS[model](self.opt, self.NULL_IDX)

    @property
    def retriever_query(self) -> str:
        return self._retriever_query

    @retriever_query.setter
    def retriever_query(self, query: str):
        self._retriever_query = query
        if query == 'one_turn':
            self._query_key = 'text'
        elif query == 'full_history':
            self._query_key = 'full_text'

    def __init__(self, opt: Opt, shared: TShared = None):
        self.opt = opt
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
        self.generation_model = opt['generation_model']
        self.regret = opt['regret']
        self.regret_intermediate_maxlen = opt['regret_intermediate_maxlen']
        self.retriever_query = opt['rag_retriever_query']

        # Super call
        self._generation_agent.__init__(self, opt, shared)  # type: ignore
        self.rag_model_type = opt['rag_model_type']

        if not shared and self.regret:
            self.regret_model = self.build_regret_model()
        elif shared:
            self.regret_model = shared.get('regret_model')

    def build_regret_model(self) -> RagModel:
        """
        Build and return regret RagModel.

        Assume dictionary is the same.
        """
        model_file = self.opt['regret_model_file']
        if model_file:
            assert os.path.exists(
                model_file
            ), 'specify correct path for --regret-model-file'
            regret_opt = Opt.load(f'{model_file}.opt')
            regret_opt['n_docs'] = self.opt['n_docs']  # Urgent that this is the same
            # add keys that were not in this model when originally trained
            regret_opt.update(
                {k: v for k, v in self.opt.items() if k not in regret_opt}
            )
            retriever_shared = None
            if all(
                [
                    regret_opt[k] == self.opt[k]
                    for k in [
                        'rag_retriever_type',
                        'path_to_index',
                        'path_to_dpr_passages',
                    ]
                ]
            ):
                logging.warning('Sharing retrievers between model and regret model!')
                retriever_shared = self.model.encoder.retriever.share()

            model = RagModel(regret_opt, self.dict, retriever_shared=retriever_shared)
            with PathManager.open(self.opt['regret_model_file'], 'rb') as f:
                states = torch.load(
                    f,
                    map_location=lambda cpu, _: cpu,
                    pickle_module=parlai.utils.pickle,
                )
            assert 'model' in states
            model.load_state_dict(states['model'])
            if self.model_parallel:
                ph = PipelineHelper()
                ph.check_compatibility(self.opt)
                self.regret_model = ph.make_parallel(self.regret_model)
            else:
                self.regret_model.cuda()
            if self.fp16:
                self.regret_model = self.regret_model.half()

            sync_parameters(self.regret_model)
            train_params = trainable_parameters(self.regret_model)
            total_params = total_parameters(self.regret_model)
            logging.info(
                f"Total regret parameters: {total_params:,d} ({train_params:,d} trainable)"
            )
        else:
            model = self.model

        return model

    def share(self):
        shared = super().share()
        if self.regret:
            shared['regret_model'] = self.regret_model
        return shared

    ########################################
    # BART-Specific Overrides              #
    ########################################

    def _convert_model(self, opt: Opt) -> Opt:
        """
        Override BartAgent._convert_model to use RagConversionScript.
        """
        return self._generation_agent._convert_model(self, opt)  # type: ignore

    def _initialize_bart(self, opt: Opt) -> Opt:
        return self._generation_agent._initialize_bart(self, opt)  # type: ignore

    ########################################
    # TorchGeneratorAgent Overrides        #
    ########################################

    ###### 0. Act/Observe ######
    def observe(self, observation: Union[Dict, Message]) -> Message:
        """
        Overrides TA.observe to tokenize the retriever query.
        """
        observation = Message(observation)
        observation = self._generation_agent.observe(self, observation)
        if observation.is_padding():
            return observation
        if 'query_vec' not in observation:
            self._set_query_vec(observation)
        if 'input_turn_cnt_vec' not in observation:
            self._set_input_turn_cnt_vec(observation)
        return observation

    ###### 1. Model Inputs ######

    def _model_input(
        self, batch: Batch
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Get input for RagModel.

        :param batch:
            batch to process

        :return (text_vec, text_lengths, query_vec, input_turn_cnt_vec):
            text_vec - tokenized batch
            text_lengths - length of each item in the batch
            query_vec - tokenized vectors for retrieval mechanism
            input_turn_cnt_vec - count of input turns for each batch item
        """
        return (
            batch.text_vec,
            batch.text_vec.ne(self.NULL_IDX).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
        )

    def _encoder_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        Optional[torch.LongTensor],
        Optional[torch.LongTensor],
        Optional[torch.LongTensor],
    ]:
        """
        Called directly when generating.

        Some RAG Model Types require different encoder inputs.

        :param batch:
            batch to process

        :return (text_vec, text_lengths, query_vec, input_turn_cnt_vec):
            text_vec - tokenized batch
            text_lengths - length of each item in the batch
            query_vec - tokenized vectors for retrieval mechanism
            input_turn_cnt_vec - count of input turns for each batch item
        """
        if hasattr(self._rag_model_interface, 'get_generation_input'):
            return self._rag_model_interface.get_generation_input(batch)  # type: ignore
        return self._model_input(batch)

    ##### 2. Standard TGA Function Overrides #####
    def build_model(self) -> RagModel:
        """
        Build and return RagModel.
        """
        model = self._generation_agent.build_rag_model(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when are adding extra special tokens.

        Modify TGA._resize_token_embeddings to access correct modules within RAG.
        """
        # map extra special tokens carefully
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in [
            'embeddings.weight',
            'seq2seq_encoder.embeddings.weight',
            'seq2seq_decoder.embeddings.weight',
        ]:
            # get new_embs
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[emb_weights] = new_embs

        return state_dict

    def build_dictionary(self) -> DictionaryAgent:
        """
        Build and return dictionary.
        """
        return self._generation_agent.build_dictionary(self)

    @staticmethod
    def update_state_dict(
        opt: Opt, state_dict: Dict[str, torch.Tensor], model: torch.nn.Module
    ):
        """
        Update the given state dict to be RAG-ified.

        :param opt:
            options
        :param state_dict:
            weights to load
        :param model:
            underlying model that will load the state_dict

        :return updated_state_dict:
            return state_dict with appropriate keys/values
        """
        # 1. Substitute all "encoder" and "decoder" keys with "seq2seq_encoder" and "seq2seq_decoder"
        if not [k for k in state_dict if k.startswith('seq2seq')]:
            for k in list(state_dict.keys()):
                if k.startswith('encoder') or k.startswith('decoder'):
                    weights = state_dict.pop(k)
                    state_dict[f'seq2seq_{k}'] = weights
        # 2. Retriever state
        if not [k for k in state_dict if 'retriever' in k]:
            retriever_state = {
                f"retriever.{k}": v
                for k, v in model.retriever.state_dict().items()  # type: ignore
            }
            state_dict.update(retriever_state)
        # 3. Handle n_positional difference
        if opt.get('n_extra_positions', 0) > 0:
            key = 'seq2seq_encoder.position_embeddings.weight'
            init_weight = (
                model.seq2seq_encoder.position_embeddings.weight  # type: ignore
            )
            if state_dict[key].size(0) < opt['n_positions'] + opt['n_extra_positions']:
                # Make sure we're not adding more positions to a model trained
                # with extra positions
                state_dict[key] = torch.cat(
                    [
                        state_dict[key].to(init_weight),  # type: ignore
                        init_weight[-opt['n_extra_positions'] :, :],  # type: ignore
                    ],
                    dim=0,
                )
        return state_dict

    def _should_override_dpr_model_weights(self, opt: Opt):
        """
        Determine if we need to override the DPR Model weights.

        Under certain circumstances, one may wish to specify a different `--dpr-model-
        file` for a pre-trained, RAG model. Thus, we additionally check to make sure
        that the loaded DPR model weights are not overwritten by the state loading.
        """
        override_dpr = False
        overrides = opt.get('override', {})
        if overrides.get('dpr_model_file') and os.path.exists(
            overrides['dpr_model_file']
        ):
            override_dpr = True
            logging.warning(
                f"Overriding DPR Model with {modelzoo_path(opt['datapath'], opt['dpr_model_file'])}"
            )
        return override_dpr

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Potentially update state dict with relevant RAG components.

        Useful when initializing from a normal seq2seq model.
        """
        try:
            if self._should_override_dpr_model_weights(self.opt):
                state_dict.update(
                    {
                        f"retriever.{k}": v
                        for k, v in self.model.retriever.state_dict().items()  # type: ignore
                    }
                )
            self.model.load_state_dict(state_dict)
        except RuntimeError as msg:
            state_dict = self.update_state_dict(self.opt, state_dict, self.model)
            msg_ = str(msg)
            if 'size mismatch' in msg_ and 'embedding' in msg_:
                if hasattr(self, 'special_toks') and len(self.special_toks) > 0:
                    state_dict = self._resize_token_embeddings(state_dict, msg_)
                    self.resized_embeddings = True  # make note that we resized here
                else:
                    raise RuntimeError(
                        f'{msg_}\n'
                        '-----------------\n'
                        'Could not load the model due to a size mismatch in the '
                        'embeddings. A common reason for this is trying to load '
                        'a model trained with fp16 but loaded without fp16. Try '
                        'adding --fp16 true or --force-fp16-tokens true.'
                    )
            self.model.load_state_dict(state_dict)

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        """
        Override TA.batchify to incorporate query and input turn vecs.
        """
        assert not sort
        if len(obs_batch) == 0:
            return Batch(batchsize=0)

        valid_exs = [ex for ex in obs_batch if self.is_valid(ex)]

        if len(valid_exs) == 0:
            return Batch(batchsize=0)

        batch = self._generation_agent.batchify(self, obs_batch, sort)

        if any(ex.get('query_vec') is not None for ex in valid_exs):
            _qs = []
            for ex in valid_exs:
                q = ex.get('query_vec', self.EMPTY)
                if type(q) is list and type(q[0]) is list:
                    # handling input turns
                    _qs += q
                else:
                    _qs.append(q)
            self.set_batch_query(batch, _qs)

        batch.input_turn_cnt_vec = None
        if any(ex.get('input_turn_cnt_vec') is not None for ex in valid_exs):
            batch.input_turn_cnt_vec = torch.cat(
                [
                    ex.get('input_turn_cnt_vec', torch.LongTensor([1]))
                    for ex in valid_exs
                ],
                dim=0,
            )
        return batch

    def vectorize(self, *args, **kwargs):
        """
        Override TA.vectorize to ensure proper super class handles vectorization.
        """
        return self._generation_agent.vectorize(self, *args, **kwargs)

    def _set_text_vec(
        self, obs: Message, history: History, truncate: Optional[int]
    ) -> Message:
        """
        Override TA._set_text_vec to ensure proper gen agent handles _set_text_vec.
        """
        return self._generation_agent._set_text_vec(self, obs, history, truncate)

    ##### 3. Custom Batch handling #####

    def set_batch_query(self, batch: Batch, queries: List[torch.LongTensor]) -> Batch:
        """
        Put the queries in the batch.

        :param batch:
            batch to put queries in
        :param queries:
            list of query tokens, presumably

        :return batch:
            return the batch, with queries.
        """
        qs, q_lens = self._pad_tensor(queries)
        batch.query_vec = qs
        batch.query_lengths = torch.LongTensor(q_lens)
        return batch

    def _set_query_vec(self, observation: Message) -> Message:
        """
        Tokenize the query for retrieval.

        :param observation:
            observation with input text.

        :return observation:
            return observation with query vec.
        """
        query_str = observation[self._query_key]
        if hasattr(self.model, 'module'):
            observation['query_vec'] = self.model.module.tokenize_query(query_str)
        else:
            observation['query_vec'] = self.model.tokenize_query(query_str)
        return observation

    def _set_input_turn_cnt_vec(self, observation: Message) -> Message:
        """
        Compute the number of turns of context for the observation query.

        :param observation:
            observation with input text.

        :return observation:
            return observation with input turn vec.
        """
        observation = self._rag_model_interface.set_input_turn_cnt_vec(
            observation, self.model, observation[self._query_key]
        )
        return observation

    ##### 4. Decoder Input Functions #####

    def _get_initial_decoder_input(
        self, bsz: int, beam_size: int, dev: torch.device
    ) -> torch.LongTensor:
        """
        Override TGA._get_initial_decoder_input to seed decoder with proper amount of
        inputs.

        This is called during generation, so additional inputs are necessary only for
        RAG Token (which retrieves during the generation step).
        """
        dec_input = self._generation_agent._get_initial_decoder_input(
            self, bsz, beam_size, dev
        )
        rag_dec_input = self._rag_model_interface.get_initial_decoder_input(dec_input)
        return rag_dec_input

    def _get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Override TGA._get_next_decoder_input to repeat decoder input appropriately.
        """
        if hasattr(self._rag_model_interface, 'get_next_decoder_input'):
            dec_input = self._rag_model_interface.get_next_decoder_input(  # type: ignore
                prev_input, selection, incr_state_inds
            )
        else:
            dec_input = self._generation_agent._get_next_decoder_input(
                self, prev_input, selection, incr_state_inds
            )
        return dec_input

    ####################################
    # Model Generation Overrides       #
    ####################################

    ##### Standard TGA Generation Overrides #####

    def _get_context(self, batch: Batch, batch_idx: int) -> torch.LongTensor:
        """
        Override TGA._get_context for rag-sequence/turn models.

        Reason: the batchsize is artificially higher (n_docs * batchsize)
        """
        if hasattr(self._rag_model_interface, 'get_ctxt_index'):
            batch_idx = self._rag_model_interface.get_ctxt_index(  # type: ignore
                batch, batch_idx
            )
        return self._generation_agent._get_context(self, batch, batch_idx)

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ) -> Tuple[List[Tuple[torch.LongTensor, torch.Tensor]], List[TreeSearch]]:
        """
        Override TGA._generate to potentially call ReGReT.

        TGA._generate is implemented in _rag_generate
        """
        if self.regret:
            beam_preds_scores, _ = self._regret_generate(
                batch, beam_size, self.regret_intermediate_maxlen, prefix_tokens
            )
            preds, _ = zip(*beam_preds_scores)
            new_batch = self._regret_rebatchify(batch, preds)  # type: ignore
            gen_outs = self._rag_generate(new_batch, beam_size, max_ts, prefix_tokens)
        else:
            gen_outs = self._rag_generate(batch, beam_size, max_ts, prefix_tokens)

        return gen_outs

    def _rerank_beams(
        self,
        batch: Batch,
        n_best_beam_preds_scores: List[List[Tuple[torch.LongTensor, torch.Tensor]]],
    ) -> List[List[Tuple[torch.LongTensor, torch.Tensor]]]:
        """
        Optionall rerank beams, according to RAG Model type.

        :param batch:
            current batch
        :param n_best_beam_preds_scores:
            bsz-length list of Tuples of predictions and scores

        :return List((pred, score)):
            return a re-ranked version of the n_best_beam_preds_scores
        """
        return self._rag_model_interface.rerank_beams(
            self.model, batch, n_best_beam_preds_scores
        )

    ##### RAG-Specific Generation Functions #####

    def _rag_generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ) -> Tuple[List[Tuple[torch.LongTensor, torch.Tensor]], List[TreeSearch]]:
        """
        Separate from _generate to handle regret.
        """
        batch = self._rag_model_interface.augment_batch_for_generation(
            batch, self.model
        )
        return self._generation_agent._generate(
            self, batch, beam_size, max_ts, prefix_tokens
        )

    def _regret_generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        """
        Swap generation model prior to full generation.
        """
        model, model_type = self.model, self.rag_model_type
        self.model = self.regret_model
        self.rag_model_type = self.regret_model.rag_model_type

        outs = self._rag_generate(batch, beam_size, max_ts, prefix_tokens)

        self.model = model
        self.rag_model_type = model_type
        return outs

    def _regret_rebatchify(
        self, batch: Batch, pred_vecs: List[torch.LongTensor]
    ) -> Batch:
        """
        Create a new batch for the model.

        Given a model's prediction/generation, create a new batch with this text substituted in as the query.

        :param batch:
            prior model batch
        :param pred_vecs:
            predicted, tokenized text from initial batch

        :return batch:
            return a new batch for the model.
        """
        new_queries: List[torch.LongTensor] = []
        query_lens: torch.LongTensor = batch.query_lengths
        query_vec: torch.LongTensor = batch.query_vec

        for i in range(batch.batchsize):
            vec_i = pred_vecs[i]
            txt_i = self._v2t(vec_i)
            query_i = torch.LongTensor(self.model.tokenize_query(txt_i))
            if self.retriever_query == 'one_turn':
                new_queries.append(query_i)
            else:
                query_i = torch.cat([query_vec[i][: query_lens[i]], query_i], dim=0)
                new_queries.append(query_i)  # type: ignore

        self.set_batch_query(batch, new_queries)

        if (
            batch.input_turn_cnt_vec is not None
            and self.retriever_query == 'full_history'
        ):
            batch.input_turn_cnt_vec += 1

        return batch

    ########################################
    # Computing Loss / Recording Metrics   #
    ########################################

    def _record_retrieval_metrics(self, batch: Batch, encoder_state: Tuple[Any, ...]):
        """
        Compute retrieval metrics, given retrieved documents.

        Only works when `--debug` is set.

        If there is knowledge in the Batch, we compute the following metrics:
        A) Doc Level:
        1. recall @ 1 --> is the correct document the first document?
        2. recall @ N --> is the correct document in the first N docs?

        B) Passage Level:
        1. recall @ 1 --> is the correct passage in the first document?
        2. recall @ N --> is the correct passage in the first N docs?

        Only works in debug mode.

        :param batch:
            training/eval batch
        :param encoder_state:
            encoder states from RagEncoder
        """
        if batch.valid_indices is None or batch.observations is None:
            return
        docs: List[List[Document]] = []
        _, _, input_turns_cnt, docs, _ = encoder_state
        if input_turns_cnt is not None:
            new_docs = []
            offset = 0
            for it in input_turns_cnt:
                docs_it = [dd for d in docs[offset : offset + it] for dd in d]
                new_docs.append(docs_it)
                offset += it
            docs = new_docs
        title_key = self.opt['gold_knowledge_title_key']
        passage_key = self.opt['gold_knowledge_passage_key']
        batchsize = len(batch.valid_indices)
        n_docs = self.opt['n_docs']
        metrics = {
            k: [0] * batchsize
            for k in [
                'doc_r@1',
                f'doc_r@{n_docs}',
                'passage_r@1',
                f'passage_r@{n_docs}',
                'title@1_f1',
                'passage@1_f1',
            ]
        }
        for i in range(batchsize):
            ex = batch.observations[i]
            label_title = normalize_answer(ex.get(title_key, ''))
            label_passage = normalize_answer(ex.get(passage_key, ''))

            for rank, doc in enumerate(docs[i]):
                model_title = normalize_answer(doc.get_title())
                model_passage = normalize_answer(doc.get_text())

                title_exact_match = model_title == label_title
                passage_match = (
                    model_passage in label_passage or label_passage in model_passage
                )

                if rank == 0:
                    metrics['doc_r@1'][i] = int(title_exact_match)
                    metrics['passage_r@1'][i] = int(passage_match)
                    metrics['title@1_f1'][i] = F1Metric.compute(
                        guess=model_title, answers=[label_title]
                    ).value()
                    metrics['passage@1_f1'][i] = F1Metric.compute(
                        guess=model_passage, answers=[label_passage]
                    ).value()
                metrics[f'doc_r@{n_docs}'][i] = int(
                    metrics[f'doc_r@{n_docs}'][i] or title_exact_match
                )
                metrics[f'passage_r@{n_docs}'][i] = int(
                    metrics[f'passage_r@{n_docs}'][i] or passage_match
                )

        for m in metrics:
            self.record_local_metric(m, AverageMetric.many(metrics[m], [1] * batchsize))

    def get_model_output(self, batch: Batch) -> Tuple[Any, ...]:
        """
        Return model output.

        :param batch:
            batch to process

        :return model_output:
            return output from model
        """
        if not self.regret:
            model_output = self.model(
                *self._model_input(batch), ys=batch.label_vec
            )  # type: ignore
            scores, preds, enc_state, *_ = model_output
        else:
            with torch.no_grad():
                beam_preds_scores, beams = self._regret_generate(
                    batch, self.beam_size, self.regret_intermediate_maxlen
                )
            regret_preds, _ = zip(*beam_preds_scores)
            new_batch = self._regret_rebatchify(batch, regret_preds)  # type: ignore
            regret_model_output = self.model(
                *self._model_input(new_batch), ys=batch.label_vec
            )  # type: ignore
            regret_scores, preds, enc_state = regret_model_output
            scores = regret_scores

        return (scores, preds, enc_state)

    def compute_loss(
        self, batch: Batch, return_output: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override standard TGA.compute_loss to call relevant RAG Model Interface.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model_output = self.get_model_output(batch)
        scores, preds, enc_state, *_ = model_output

        self._record_retrieval_metrics(batch, enc_state)
        (
            loss,
            metric_loss,
            metric_correct,
            metric_target_tokens,
        ) = self._rag_model_interface.compute_loss(
            self.criterion, scores, preds, enc_state, batch.label_vec
        )

        self.record_local_metric(
            'loss', AverageMetric.many(metric_loss, metric_target_tokens)
        )
        self.record_local_metric(
            'ppl', PPLMetric.many(metric_loss, metric_target_tokens)
        )
        self.record_local_metric(
            'token_acc', AverageMetric.many(metric_correct, metric_target_tokens)
        )
        self.record_local_metric(
            'token_em',
            AverageMetric.many(
                [x == y for x, y in zip(metric_correct, metric_target_tokens)]
            ),
        )

        if return_output:
            return loss, model_output
        else:
            return loss

    def _construct_token_losses(self, labels, model_output):
        """
        Override default `_construct_token_loss` so that sequence model does not throw
        an error while being evaled with the `--verbose` flag.

        TODO: implement this
        """
        return None
