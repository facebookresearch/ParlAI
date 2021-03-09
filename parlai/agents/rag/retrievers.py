#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
import copy
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig
from typing import Tuple, List, Optional
from typing_extensions import final

from parlai.agents.rag.classes import Document
from parlai.agents.rag.conversion_utils import BertConversionUtils, DPR_BERT_MODEL_PATH
from parlai.agents.rag.interfaces import RagQueryEncoder, RagRetriever
from parlai.agents.transformer.modules import DefaultTransformerEncoder
from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.utils import logging
from parlai.utils.typing import TShared


class RetrieverType(Enum):
    DPR = 'dpr'
    TFIDF = 'tfidf'
    TFIDF_AND_DPR = 'tfidf_and_dpr'


BLANK_DOC = Document('', '', '')


class RagRetrieverBase(nn.Module, ABC):
    """
    RAG Retriever.

    A RAG Retriever just needs to implement a retrieve_and_score method.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        super().__init__()
        self.retriever_type = RetrieverType(opt['rag_retriever_type'])
        self.opt = opt
        self.print_docs = opt.get('print_docs', False)
        self.max_query_len = opt['rag_query_truncate'] or 1024
        self.sentence_tokenize_docs = opt['sentence_tokenize_docs']
        self.doc_enc_grad_ctxt = torch.no_grad

    @final
    def score_and_rank(
        self,
        input_text: List[str],
        query_enc: torch.Tensor,
        documents: List[List[Document]],
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Score list of given documents; return ranked order.

        :param input_text:
            input text
        :param query_enc:
            encoding of the query (input) text
        :param documents:
            list of documents for each input string

        :return (docs, scores):
            return sorted list of documents, and their respective scores.
        """
        if hasattr(self, 'n_final_docs'):
            n_docs = self.n_final_docs
        else:
            n_docs = self.n_docs
        if not any([i for i in input_text]):
            # dummy batch
            docs = [[BLANK_DOC] * n_docs] * len(input_text)
            scores = query_enc.new(len(input_text), n_docs).fill_(0)
            return docs, scores
        ranked_docs: List[List[Document]] = []
        ranked_scores: List[torch.Tensor] = []
        max_n_docs = min(max((len(d) for d in documents)), n_docs)
        for i in range(len(documents)):
            docs_i = documents[i]
            if mask is not None:
                mask_i = mask[i]
            score_i = self.score(
                [input_text[i]], query_enc[i : i + 1], [docs_i], mask_i
            )[0]
            if score_i.size(0) > 1:
                docs, scores = argsort_scores_and_docs(score_i, docs_i, n_docs)
            else:
                docs, scores = docs_i, score_i
            ranked_docs.append(docs)
            if scores.size(0) < max_n_docs:
                scores = torch.cat(
                    (scores, scores.new(max_n_docs - scores.size(0)).fill_(0))
                )
                docs += [BLANK_DOC] * (max_n_docs - scores.size(0))
            ranked_scores.append(scores)
        return ranked_docs, torch.stack(ranked_scores)

    @abstractmethod
    def retrieve_and_score(
        self,
        query_text: List[str],
        query_vectors: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieve and score documents, given a query vector.

        NOTE: Subclasses **MUST** implement a retrieve method

        :param query_text:
            raw query texts
        :param query_vectors:
            encoding of query
        :param mask:
            query encoding mask

        :return (docs, scores):
            docs: list of (text, title) tuples for each batch example.
            scores: document scores (given query vecs)
        """
        raise RuntimeError(
            f"RagRetriever.retrieve() not implemented for retriever type {self.opt['rag_retriever_type']}"
        )

    def score(
        self,
        query_text: List[str],
        query_vectors: torch.Tensor,
        documents: List[List[Document]],
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Score documents given a query text, query vector.

        Useful for post-hoc scoring of documents sans-retrieval.

        :param query_text:
            raw query texts
        :param query_vectors:
            encoding of query
        :param documents:
            document sets for each query text
        :param mask:
            query encoding mask.

        :return scores:
            return [bsz, n_docs] document scores tensor
        """
        raise RuntimeError(
            f"RagRetriever.score() not implemented for retriever type {self.opt['rag_retriever_type']}"
        )


def argsort_scores_and_docs(
    scores: torch.Tensor, docs: List[Document], n_docs: int
) -> Tuple[List[Document], torch.Tensor]:
    """
    Sort scores and documents by score, return n_docs ranked docs/scores.

    :param scores:
        scores with which to rank
    :param docs:
        docs to argsort
    :param n_docs:
        number of docs to return

    :return:
        (docs, scores) --> sorted documents, according to scores.
    """
    scores_sorter = scores.sort(descending=True)
    ranked_docs = [docs[idx] for idx in scores_sorter.indices[:n_docs]]
    ranked_scores = scores_sorter.values[:n_docs]
    return ranked_docs, ranked_scores


class TFIDFRetriever(RagRetrieverBase, RagRetriever):
    """
    Use TFIDF to score wikipedia documents.
    """

    def __init__(self, opt: Opt, shared: TShared = None, **kwargs):
        """
        Init a TFIDFRetrieverAgent.
        """
        super().__init__(opt, shared=shared)
        tfidf_opt = {
            'model': 'rag_tfidf_retriever',
            'model_file': opt.get(
                'tfidf_model_path', 'zoo:wikipedia_full/tfidf_retriever/model'
            ),
            'tfidf_model_path': opt.get('tfidf_model_path'),
            'retriever_num_retrieved': opt['n_docs'],
            'retriever_mode': 'keys',
            'override': {'model': 'rag_tfidf_retriever', 'remove_title': False},
        }
        self.n_docs = opt['n_docs']
        self.split_docs = opt['tfidf_split_docs']
        self.max_doc_paragraphs = opt['tfidf_max_doc_paragraphs']
        assert self.max_doc_paragraphs != 0
        if not shared:
            self.tfidf_retriever = create_agent(tfidf_opt)
        else:
            self.tfidf_retriever = shared['tfidf_retriever']

    def share(self) -> TShared:
        shared = super().share()
        shared['tfidf_retriever'] = self.tfidf_retriever
        return shared

    def retrieve_and_score(
        self,
        query_text: List[str],
        query_vectors: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieve and score using TFIDF.

        Given query text, return closest query documents.

        :param query_text:
            raw query texts
        :param query_vectors:
            encoding of query
        :param mask:
            query encoding mask

        :return (docs, scores):
            docs: list of (text, title) tuples for each batch example
            scores: doc scores
        """
        docs = []
        scores = []

        for query in query_text:
            self.tfidf_retriever.observe({'text': query, 'episode_done': True})
            act = self.tfidf_retriever.act()
            if 'candidate_scores' not in act:
                scores_i = [0] * self.n_docs
                docs_i = [Document(text='', title='', docid=None)] * self.n_docs
            else:
                scores_i = act['candidate_scores']
                candidate_docs = act['text_candidates']
                ids_i = act['candidate_ids']
                if self.split_docs:
                    candidate_docs = []
                    for j, cand in enumerate(act['text_candidates']):
                        title = cand.split('\n\n')[0]
                        paragraphs = cand.split('\n\n')[1:]
                        candidate_docs += [
                            Document(text=p, title=title, docid=f"{ids_i[j]}_{k}")
                            for k, p in enumerate(paragraphs)
                        ]
                else:

                    def _build_doc(idx, cand):
                        title = cand.split('\n\n')[0]
                        paragraphs = cand.split('\n\n')[1:]
                        if self.max_doc_paragraphs > 0:
                            paragraphs = paragraphs[: self.max_doc_paragraphs]
                        return Document(
                            title=title, text=' '.join(paragraphs), docid=ids_i[idx]
                        )

                    candidate_docs = [
                        _build_doc(j, c) for j, c in enumerate(act['text_candidates'])
                    ]
                docs_i = candidate_docs[: self.n_docs]
                if len(docs_i) < self.n_docs:
                    # Something went wrong with TFIDF here; need to add null docs
                    logging.warn(
                        f'Ex has less than {self.n_docs} TFIDF docs: {len(docs_i)}'
                    )
                    num_null = self.n_docs - len(docs_i)
                    docs_i += [Document(text='', title='', docid=None)] * num_null
                    scores_i = np.append(scores_i, [0] * num_null)
            docs.append(docs_i)
            scores.append(torch.Tensor(scores_i))

        return (docs, torch.stack(scores).to(query_vectors))


class RagDprEncoder(nn.Module, DefaultTransformerEncoder, RagQueryEncoder):
    """
    Basically provide a wrapper around TransformerEncoder to load RAG Query/Document
    models.
    """

    def __init__(
        self,
        opt: Opt,
        vocabulary_size: int,
        manifest: DefaultTransformerEncoder.Manifest = None,
        embedding: Optional[nn.Embedding] = None,
        padding_idx: int = 0,
        reduction_type: str = 'mean',
        n_positions: Optional[int] = None,
        n_segments: Optional[int] = None,
        embeddings_scale: Optional[bool] = None,
        # RAG-Specific
        dpr_model: str = 'bert',
        pretrained_path: str = DPR_BERT_MODEL_PATH,
        encoder_type: str = 'query',
    ):
        opt = copy.deepcopy(opt)
        if dpr_model in ['bert', 'bert_from_parlai_rag']:
            # Override options
            config: BertConfig = BertConfig.from_pretrained('bert-base-uncased')
            opt["n_heads"] = config.num_attention_heads
            opt["n_layers"] = config.num_hidden_layers
            opt["embedding_size"] = config.hidden_size
            opt["ffn_size"] = config.intermediate_size
            vocabulary_size = config.vocab_size
            opt["dropout"] = config.hidden_dropout_prob
            opt["attention_dropout"] = config.attention_probs_dropout_prob
            padding_idx = config.pad_token_id
            reduction_type = 'first'
            n_positions = config.max_position_embeddings
            opt["activation"] = config.hidden_act
            opt["variant"] = 'xlm'
            n_segments = config.type_vocab_size
            embedding = torch.nn.Embedding(
                vocabulary_size, opt["embedding_size"], padding_idx=padding_idx
            )
        super().__init__(
            opt=opt,
            manifest=manifest,
            vocabulary_size=vocabulary_size,
            embedding=embedding,
            padding_idx=padding_idx,
            reduction_type=reduction_type,
            n_positions=n_positions,
            n_segments=n_segments,
            embeddings_scale=embeddings_scale,
        )

        self._load_state(dpr_model, pretrained_path, encoder_type)

    def _load_state(self, dpr_model: str, pretrained_path: str, encoder_type: str):
        """
        Load pre-trained model states.

        :param dpr_model:
            which dpr model type we're using
        :param pretrained_path:
            path to pretrained model
        :param encoder_type:
            whether this is a query or document encoder
        """
        if dpr_model == 'bert':
            state_dict = BertConversionUtils.load_bert_state(
                self.state_dict(),
                pretrained_dpr_path=pretrained_path,
                encoder_type=encoder_type,
            )
            self.load_state_dict(state_dict)
        elif dpr_model == 'bert_from_parlai_rag':
            state_dict = torch.load(pretrained_path)["model"]
            key = f"{encoder_type}_encoder."
            state_dict = {
                k.split(key)[-1]: v for k, v in state_dict.items() if key in k
            }
            self.load_state_dict(state_dict)


class RagDprQueryEncoder(RagDprEncoder):
    """
    Query Encoder for DPR.
    """

    def __init__(self, *args, **kwargs):
        kwargs['encoder_type'] = 'query'
        super().__init__(*args, **kwargs)
