#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Retrievers for RAG.
"""
from abc import ABC, abstractmethod
import copy
import csv
import gzip
import numpy as np
import os
from parlai.core.message import Message
import torch
import torch.cuda
import torch.nn
import transformers
from tqdm import tqdm

try:
    from transformers import BertTokenizerFast as BertTokenizer
except ImportError:
    from transformers import BertTokenizer
from typing import Tuple, List, Dict, Union, Optional, Any
from typing_extensions import final
from sklearn.feature_extraction.text import TfidfVectorizer

from parlai.agents.tfidf_retriever.tfidf_retriever import TfidfRetrieverAgent
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.loader import register_agent
from parlai.core.opt import Opt
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_internet.mutators import chunk_docs_in_message
import parlai.tasks.wizard_of_internet.constants as CONST
import parlai.utils.logging as logging
from parlai.utils.torch import padded_tensor
from parlai.utils.typing import TShared
from parlai.utils.io import PathManager

from parlai.agents.rag.dpr import DprQueryEncoder
from parlai.agents.rag.polyfaiss import RagDropoutPolyWrapper
from parlai.agents.rag.indexers import DenseHNSWFlatIndexer, indexer_factory
from parlai.agents.rag.args import (
    RetrieverType,
    WOW_INDEX_PATH,
    WOW_PASSAGES_PATH,
    POLYENCODER_OPT_KEYS,
    TRANSFORMER_RANKER_BASE_OPT,
    WOW_COMPRESSED_INDEX_PATH,
)
from parlai.agents.rag.retrieve_api import SearchEngineRetriever


def load_passage_reader(
    ctx_file: str, return_dict: bool = True
) -> Union[Dict[str, Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Load passages from file, corresponding to a FAISS index.

    We attempt to read the passages with a csv reader.

    If passage files are not saved correctly with a csv reader,
    reads can fail.

    :param ctxt_file:
        file to read

    :return reader:
        return a reader over the passages
    """
    logging.info(f'Reading data from: {ctx_file}')
    f_open = gzip.open if ctx_file.endswith(".gz") else open
    try:
        passages = {} if return_dict else []
        with f_open(ctx_file) as tsvfile:
            _reader = csv.reader(tsvfile, delimiter='\t')  # type: ignore
            ids = []
            for idx, row in tqdm(enumerate(_reader)):
                if idx == 0:
                    assert row[0] == 'id'
                    ids.append(-1)
                elif idx <= 1:
                    ids.append(row[0])
                    if return_dict:
                        passages[row[0]] = (row[1], row[2])  # type: ignore
                    else:
                        passages.append((row[0], row[1], row[2]))  # type: ignore
                    continue
                else:
                    assert int(row[0]) == int(ids[idx - 1]) + 1, "invalid load"
                    if return_dict:
                        passages[row[0]] = (row[1], row[2])  # type: ignore
                    else:
                        passages.append((row[0], row[1], row[2]))  # type: ignore
                    ids.append(row[0])

        del ids
    except (csv.Error, AssertionError) as e:
        passages = {} if return_dict else []
        logging.error(f'Exception: {e}')
        logging.warning('Error in loading csv; loading via readlines')
        with f_open(ctx_file) as tsvfile:
            for idx, l in tqdm(enumerate(tsvfile.readlines())):
                line = l.replace('\n', '').split('\t')  # type: ignore
                assert len(line) == 3
                if idx == 0:
                    assert line[0] == 'id'
                if line[0] != 'id':
                    if return_dict:
                        passages[line[0]] = (line[1], line[2])  # type: ignore
                    else:
                        passages.append((line[0], line[1], line[2]))  # type: ignore
    return passages


def load_passages_dict(ctx_file: str) -> Dict[str, Tuple[str, str]]:
    """
    Load passages as a dict.

    :param ctx_file:
        file to read

    :return passages_dict:
        return a dict mapping passage id to a tuple of (text, title)
    """
    psgs_dict = load_passage_reader(ctx_file, return_dict=True)
    assert isinstance(psgs_dict, dict)
    return psgs_dict


def load_passages_list(ctx_file: str) -> List[Tuple[str, str, str]]:
    """
    Load passages as a list.

    :param ctx_file:
        file to read

    :return passages_dict:
        return a list of 3-tuples (id, text, title)
    """
    psgs_list = load_passage_reader(ctx_file, return_dict=False)
    assert isinstance(psgs_list, list)
    return psgs_list


class Document:
    """
    A Document used in retrieval.
    """

    TITLE_DELIM = ' / '
    PASSAGE_DELIM = ' // '

    def __init__(self, title: str, text: str, docid: Union[int, str]):
        assert all(isinstance(t, str) for t in [title, text])
        self._title = title
        self._text = text
        self._id = str(docid)

    def get_title(self) -> str:
        return self._title

    def get_text(self) -> str:
        return self._text

    def get_id(self) -> str:
        return self._id

    def __repr__(self):
        return f"ID: {self._id}\nTitle: {self._title}\nText: {self._text}"

    def __str__(self):
        return f"{self._title} | {self._text}"

    def get_passage_str(self):
        return f"{self._title.strip()}{self.TITLE_DELIM}{self._text.strip()}{self.PASSAGE_DELIM}"

    def get_tokenization_str(self):
        return f"{self._title.strip()}{self.TITLE_DELIM}{self._text.strip()}"


BLANK_DOC = Document('', '', '')


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


def clean_vec(
    vec: torch.LongTensor, end_idx: int, special_toks: List[int] = None
) -> List[int]:
    """
    Remove special tokens from a tensor prior to text conversion.
    """
    new_vec = []
    for i in vec:
        if i == end_idx:
            break
        elif special_toks and i in special_toks:
            continue
        new_vec.append(i)
    return new_vec


class RagRetrieverTokenizer:
    """
    Wrapper for various tokenizers used by RAG Query Model.
    """

    VOCAB_PATH = 'vocab.txt'

    def __init__(
        self,
        datapath: str,
        query_model: str,
        dictionary: DictionaryAgent,
        max_length: int = 256,
        delimiter='\n',
    ):
        """
        :param query_model:
            query model type (e.g. bert)
        :param dictionary:
            ParlAI dictionary agent
        :param fast:
            whether to instantiate fast BertTokenizer
        :param max_length:
            maximum length of encoding.
        """
        self.datapath = datapath
        self.query_model = query_model
        self.tokenizer = self._init_tokenizer(dictionary)
        self.max_length = max_length
        self._delimiter = delimiter

    def _init_tokenizer(
        self, dictionary: DictionaryAgent
    ) -> Union[BertTokenizer, DictionaryAgent]:
        """
        If a regular parlai model, use the regular dictionary.

        Otherwise, build as necessary

        :param dictionary:
            ParlAI dictionary agent
        """
        if self.query_model in ['bert', 'bert_from_parlai_rag']:
            try:
                return BertTokenizer.from_pretrained('bert-base-uncased')
            except (ImportError, OSError):
                vocab_path = PathManager.get_local_path(
                    os.path.join(self.datapath, "bert_base_uncased", self.VOCAB_PATH)
                )
                return transformers.BertTokenizer.from_pretrained(vocab_path)
        else:
            return dictionary

    def get_pad_idx(self) -> int:
        """
        Return pad token idx.
        """
        if self.query_model in ['bert', 'bert_from_parlai_rag']:
            return self.tokenizer.pad_token_id
        else:
            return self.tokenizer[self.tokenizer.null_token]

    def get_delimiter(self) -> str:
        """
        Return delimiter.
        """
        return self._delimiter

    def get_bos_idx(self) -> int:
        """
        Return start token idx.
        """
        if self.query_model in ['bert', 'bert_from_parlai_rag']:
            return self.tokenizer.bos_token_id or 1
        else:
            return self.tokenizer[self.tokenizer.start_token]

    def get_eos_idx(self) -> int:
        """
        Return start token idx.
        """
        if self.query_model in ['bert', 'bert_from_parlai_rag']:
            return self.tokenizer.eos_token_id or 2
        else:
            return self.tokenizer[self.tokenizer.end_token]

    def encode(self, txt: str, txt_pair: Optional[str] = None) -> List[int]:
        """
        Encode text.

        :param txt:
            text to encode
        :param txt_pair:
            Optional additional text to encode.
            Useful if encoding two parts of a text, e.g. title & text.

        :return encoding:
            return encoded text.
        """
        if self.query_model in ['bert', 'bert_from_parlai_rag']:
            txt = txt.lower().strip()
            if txt_pair:
                txt_pair = txt_pair.lower().strip()
            return self.tokenizer.encode(
                txt,
                text_pair=txt_pair,
                add_special_tokens=True,
                max_length=self.max_length,
                pad_to_max_length=False,
                truncation='longest_first',
            )
        else:
            return self.tokenizer.txt2vec(txt)

    def decode(self, vec: torch.LongTensor) -> str:
        """
        Decode a token vector into a string.
        """
        if self.query_model in ['bert', 'bert_from_parlai_rag']:
            return self.tokenizer.decode(
                clean_vec(vec, self.get_eos_idx()), skip_special_tokens=True
            )
        else:
            return self.tokenizer.vec2txt(
                clean_vec(
                    vec,
                    self.get_eos_idx(),
                    special_toks=[
                        self.get_pad_idx(),
                        self.get_bos_idx(),
                        self.get_eos_idx(),
                    ],
                )
            )


class RagRetriever(torch.nn.Module, ABC):
    """
    RAG Retriever.

    Provides an interface to the RagModel for retrieving documents.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared = None):
        super().__init__()
        self.retriever_type = RetrieverType(opt['rag_retriever_type'])
        if not (
            (
                self.retriever_type
                in (
                    RetrieverType.SEARCH_ENGINE,
                    RetrieverType.OBSERVATION_ECHO_RETRIEVER,
                )
            )
            or (opt.get('retriever_debug_index') in [None, 'none'])
        ):
            if opt.get('retriever_debug_index') == 'exact':
                opt['path_to_index'] = WOW_INDEX_PATH
            else:
                opt['path_to_index'] = WOW_COMPRESSED_INDEX_PATH
            opt['path_to_dpr_passages'] = WOW_PASSAGES_PATH
        self.opt = opt
        self.print_docs = opt.get('print_docs', False)
        self.max_doc_len = opt['max_doc_token_length']
        self.max_query_len = opt['rag_query_truncate'] or 1024
        self.end_idx = dictionary[dictionary.end_token]
        self._tokenizer = RagRetrieverTokenizer(
            datapath=opt['datapath'],
            query_model=opt['query_model'],
            dictionary=dictionary,
            delimiter=opt.get('delimiter', '\n') or '\n',
        )
        self.fp16 = (
            not opt['no_cuda']
            and torch.cuda.is_available()
            and self.opt.get('fp16', False)
        )

    @final
    def retrieve(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieve documents, given a query vector.

        :param query:
            tokenized query

        :return (docs, scores):
            docs: list of Documents for each batch example.
            scores: [bsz, n_docs] document scores
        """
        docs, scores = self.retrieve_and_score(query)
        if self.print_docs:
            self.display_docs(docs)
        self.top_docs = [[str(d) for d in ds] for ds in docs]
        return docs, scores

    @abstractmethod
    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieve documents for a given query.

        :param query:
            tokenized query

        :return (docs, scores):
            docs: list of Documents for each batch example.
            scores: [bsz, n_docs] document scores
        """

    def tokenize_query(self, query: str) -> List[int]:
        """
        Tokenize the query.

        :param query:
            query to tokenize

        :return tokenized_query:
            return list of tokens
        """
        return self._tokenizer.encode(query)

    def vectorize_texts(
        self,
        input_text: List[str],
        tokenizer: RagRetrieverTokenizer,
        max_len: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Vectorize a set of input texts with an arbitrary RagRetrieverTokenizer.

        :param input_text:
            list of input strings
        :param tokenizer:
            tokenizer that encodes the input strings
        :param max_len:
            max length to tokenize

        :return vecs:
            returns a stacked padded tensor of tokens.
        """
        vecs = [tokenizer.encode(q) for q in input_text]
        if max_len:
            vecs = [v[:max_len] for v in vecs]
        vecs, _ = padded_tensor(
            vecs,
            fp16friendly=self.fp16,
            pad_idx=tokenizer.get_pad_idx(),
            max_len=max_len,
        )
        return vecs

    def get_delimiter(self) -> str:
        """
        Return the tokenizer's delimiter.
        """
        return self._tokenizer.get_delimiter()

    def display_docs(self, top_docs: List[List[Document]]):
        """
        Prints documents.

        :param top_docs:
            list of documents for each batch item
        """
        for docs in top_docs:
            for rank, doc in enumerate(docs):
                print(f"Rank: {rank}\n{doc}")

    def share(self) -> TShared:
        """
        Share retriever stuff.

        Share anything that can be handily used by other retrievers.

        This is primarily to share things that take up substantial RAM
        (indices, passages)
        """
        return {}


class RagRetrieverReranker(RagRetriever, ABC):
    """
    Trait that carries methods for Reranker-based retrievers.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared = None):
        super().__init__(opt, dictionary, shared=shared)
        self.n_final_docs = opt['n_docs']

    @final
    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Perform two-stage retrieval; rescore initial set of docs.

        :param query:
            query tokens

        :return (docs, scores):
            docs: list of Documents for each batch example
            scores: doc scores
        """
        # 1. Get Initial documents
        initial_docs, initial_scores = self._retrieve_initial(query)
        new_scores = self._rescore(query, initial_docs)

        # 2. Get new scores
        final_docs: List[List[Document]] = []
        final_scores: List[torch.Tensor] = []
        new_score_lambda = self._get_new_score_lambda()

        for i in range(len(initial_docs)):
            docs_i = initial_docs[i]
            initial_scores_i = initial_scores[i]
            scores_i = torch.mul(initial_scores_i, (1 - new_score_lambda)) + torch.mul(
                new_scores[i], new_score_lambda
            )
            docs_i, scores_i = argsort_scores_and_docs(
                scores_i, docs_i, self.n_final_docs
            )
            final_docs.append(docs_i)
            final_scores.append(scores_i)

        return final_docs, torch.stack(final_scores)

    @abstractmethod
    def _retrieve_initial(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Perform initial stage of retrieval.

        :param query:
            tokenized query

        :return (docs, scores):
            docs: list of Documents for each batch example
            scores: doc scores
        """

    @abstractmethod
    def _rescore(
        self, query: torch.LongTensor, docs: List[List[Document]]
    ) -> torch.Tensor:
        """
        Rescore retrieved documents.

        :param query:
            tokenized query
        :param docs:
            List of initially retrieved top docs for each batch example

        :return scores:
            return new doc scores.
        """

    @abstractmethod
    def _get_new_score_lambda(self) -> torch.nn.Parameter:
        """
        Return the lambda used for computing the new score.
        """


class DPRRetriever(RagRetriever):
    """
    DPR Retriever.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared=None):
        """
        Initialize DPR Retriever.
        """
        super().__init__(opt, dictionary, shared=shared)
        self.load_index(opt, shared)
        self.n_docs = opt['n_docs']
        self.query_encoder = DprQueryEncoder(
            opt, dpr_model=opt['query_model'], pretrained_path=opt['dpr_model_file']
        )

    def load_index(self, opt, shared):
        if not shared:
            self.indexer = indexer_factory(opt)
            index_path = modelzoo_path(opt['datapath'], opt['path_to_index'])
            passages_path = modelzoo_path(opt['datapath'], opt['path_to_dpr_passages'])
            embeddings_path = None
            if opt['path_to_dense_embeddings'] is not None:
                embeddings_path = modelzoo_path(
                    opt['datapath'], opt['path_to_dense_embeddings']
                )
            self.indexer.deserialize_from(index_path, embeddings_path)
            self.passages = load_passages_dict(passages_path)
        elif shared:
            self.indexer = shared['indexer']
            self.passages = shared['passages']

    def share(self) -> TShared:
        """
        Share FAISS retriever and passages.
        """
        shared = super().share()
        shared['indexer'] = self.indexer
        shared['passages'] = self.passages
        return shared

    def index_retrieve(
        self, query: torch.Tensor, n_docs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve over FAISS index.

        :param query:
            bsz x embed_dim query tensor
        :param n_docs:
            number of docs to retrieve

        :return (ids, scores):
            ids: [bsz, n_docs] tensor of document IDs
            scores: [bsz, n_docs] tensor of document scores
        """
        # retrieve docs and scores, reconstruct document embeddings & scores
        # NOTE: important that detach occurs _for retrieval only_, as we use the
        # query encodings to compute scores later in this function; if detached,
        # gradient will not flow to the query encoder.
        top_docs_and_scores = self.indexer.search(
            query.cpu().detach().to(torch.float32).numpy(), n_docs
        )
        ids, np_vectors = zip(*top_docs_and_scores)
        vectors = torch.tensor(np.array(np_vectors)).to(query)
        if isinstance(self.indexer, DenseHNSWFlatIndexer):
            vectors = vectors[:, :, :-1]
        # recompute exact FAISS scores
        scores = torch.bmm(query.unsqueeze(1), vectors.transpose(1, 2)).squeeze(1)
        if torch.isnan(scores).sum().item():
            raise RuntimeError(
                '\n[ Document scores are NaN; please look into the built index. ]\n'
                '[ This generally happens if FAISS cannot separate vectors appropriately. ]\n'
                '[ If using a compressed index, try building an exact index: ]\n'
                '[ $ python index_dense_embeddings --indexer-type exact... ]'
            )
        ids = torch.tensor([[int(s) for s in ss] for ss in ids])

        return ids, scores

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieve and score.

        For DPR, we encode query tokens and retrieve from FAISS index.

        :param query:
            query tokens

        :return (docs, scores):
            docs: list of (text, title) tuples for each batch example
            scores: doc scores
        """
        query_enc = self.query_encoder(query)
        top_doc_ids_tensor, top_doc_scores = self.index_retrieve(query_enc, self.n_docs)
        top_docs, top_doc_ids = [], []
        for i in range(query.size(0)):
            ids_i = []
            docs_i = []
            for int_id in top_doc_ids_tensor[i]:
                doc_id = str(int_id.item())
                passage = self.passages[doc_id]

                ids_i.append(doc_id)
                docs_i.append(Document(title=passage[1], text=passage[0], docid=doc_id))
            top_docs.append(docs_i)
            top_doc_ids.append(ids_i)
        return top_docs, top_doc_scores


class TFIDFRetriever(RagRetriever):
    """
    Use TFIDF to retrieve wikipedia documents.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared = None):
        """
        Init a TFIDFRetrieverAgent.
        """
        opt['query_model'] = 'tfidf'
        super().__init__(opt, dictionary, shared=shared)
        tfidf_opt = {
            'model': 'rag_tfidf_retriever',
            'model_file': (opt['tfidf_model_path']),
            'tfidf_model_path': opt['tfidf_model_path'],
            'retriever_num_retrieved': opt['n_docs'],
            'retriever_mode': 'keys',
            'override': {'model': 'rag_tfidf_retriever', 'remove_title': False},
        }
        self.n_docs = opt['n_docs']
        self.max_doc_paragraphs = opt['tfidf_max_doc_paragraphs']
        assert self.max_doc_paragraphs != 0
        if not shared:
            self.tfidf_retriever = create_agent(tfidf_opt)
            self.query_encoder = DprQueryEncoder(
                opt, dpr_model=opt['query_model'], pretrained_path=opt['dpr_model_file']
            )
        else:
            self.tfidf_retriever = shared['tfidf_retriever']
            self.query_encoder = shared['query_encoder']

    def share(self) -> TShared:
        shared = super().share()
        shared['tfidf_retriever'] = self.tfidf_retriever
        shared['query_encoder'] = self.query_encoder
        return shared

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Retrieve and score using TFIDF.

        :param query:
            query tokens

        :return (docs, scores):
            docs: list of (text, title) tuples for each batch example
            scores: doc scores
        """

        def _build_doc(idx, cand):
            title = cand.split('\n\n')[0]
            paragraphs = cand.split('\n\n')[1:]
            if self.max_doc_paragraphs > 0:
                paragraphs = paragraphs[: self.max_doc_paragraphs]
            return Document(title=title, text=' '.join(paragraphs), docid=ids_i[idx])

        docs = []
        scores = []

        for q in query:
            query_text = self._tokenizer.decode(q)
            self.tfidf_retriever.observe({'text': query_text, 'episode_done': True})
            act = self.tfidf_retriever.act()
            if 'candidate_scores' not in act:
                scores_i = [0] * self.n_docs
                docs_i = [BLANK_DOC] * self.n_docs
            else:
                scores_i = act['candidate_scores']
                candidate_docs = act['text_candidates']
                ids_i = act['candidate_ids']
                candidate_docs = [
                    _build_doc(j, c) for j, c in enumerate(act['text_candidates'])
                ]
                docs_i = candidate_docs[: self.n_docs]
                scores_i = scores_i[: self.n_docs]
                if len(docs_i) < self.n_docs:
                    # Something went wrong with TFIDF here; need to add null docs
                    logging.warning(
                        f'Ex has less than {self.n_docs} TFIDF docs: {len(docs_i)}'
                    )
                    num_null = self.n_docs - len(docs_i)
                    docs_i += [BLANK_DOC] * num_null
                    scores_i = np.append(scores_i, [0] * num_null)
            docs.append(docs_i)
            scores.append(torch.FloatTensor(scores_i).to(query.device))

        scores = torch.stack(scores)
        return docs, scores


class DPRThenTorchReranker(RagRetrieverReranker, DPRRetriever, ABC):
    """
    Base Class for DPR --> TorchRanker Retrievers.

    Handles some shared functionality.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared = None):
        """
        Initialize DPR model.

        It is up to subclasses to initialize rerankers.
        """
        RagRetrieverReranker.__init__(self, opt, dictionary, shared=shared)
        self.dpr_num_docs = opt['dpr_num_docs']
        assert self.dpr_num_docs
        dpr_opt = copy.deepcopy(opt)
        dpr_opt['n_docs'] = self.dpr_num_docs
        DPRRetriever.__init__(self, dpr_opt, dictionary, shared=shared)

    def get_reranker_opts(self, opt: Opt) -> Dict[str, Any]:
        """
        Provide options used when building the rerankers.

        Base class ensures that various optimizations (cuda, fp16, parallel)
        are accounted for.

        :param opt:
            base opt

        :return options_dict:
            return a dictionary mapping options to values.
        """
        return {
            'no_cuda': opt['no_cuda'],
            'fp16': opt['fp16'],
            'model_parallel': opt['model_parallel'],
            'data_parallel': opt['data_parallel'],
        }

    def _build_reranker(
        self, opt: Opt
    ) -> Tuple[torch.nn.Module, RagRetrieverTokenizer]:
        """
        Builds reranker.

        :param opt:
            original opt

        :return (module, dict)
            module: the model from the agent created via the options
            dict: A RagRetrieverTokenizer, dictionary for the created model.
        """
        rerank_opt = copy.deepcopy(opt)
        rerank_opt = {**TRANSFORMER_RANKER_BASE_OPT, **self.get_reranker_opts(opt)}
        logging.disable()
        agent = create_agent(rerank_opt)
        logging.enable()
        assert isinstance(agent, TorchRankerAgent)

        return (
            agent.model,
            RagRetrieverTokenizer(opt['datapath'], '', agent.dict, max_length=360),
        )

    def _retrieve_initial(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Initial DPR retrieval.

        Just call superclass to retrieve first stage.

        :param query:
            encoding of query
        :param mask:
            optional query mask

        :return (docs, scores):
            docs: list of (text, title) tuples for each batch example
            scores: doc scores
        """
        return DPRRetriever.retrieve_and_score(self, query)


class DPRThenPolyRetriever(DPRThenTorchReranker):
    """
    2 Stage Retrieval with DPR and Poly-encoder.

    1. Retrieve N Docs with DPR
    2. Rescore docs with polyencoder
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared = None):
        """
        Initialize a Poly-Encoder Agent.
        """
        # 1. Call super to init DPR
        super().__init__(opt, dictionary, shared=shared)

        # 2. Poly-encoder
        self.polyencoder, self.poly_tokenizer = self._build_reranker(opt)
        self.register_parameter(
            "poly_score_lambda",
            torch.nn.Parameter(torch.tensor([float(opt['poly_score_initial_lambda'])])),
        )

    def _get_new_score_lambda(self) -> torch.nn.Parameter:
        """
        Return the lambda used for computing the new score.
        """
        return self.poly_score_lambda  # type: ignore

    def get_reranker_opts(self, opt: Opt) -> Dict[str, Any]:
        """
        Provide options used when building the polyencoder.

        :param opt:
            base opt

        :return options_dict:
            return a dictionary mapping options to values.
        """
        from parlai.agents.rag.args import PRETRAINED_RANKER_TYPES

        init_path = opt['polyencoder_init_model']
        if init_path in PRETRAINED_RANKER_TYPES:
            init_model = f"zoo:pretrained_transformers/poly_model_huge_{opt['polyencoder_init_model']}/model"
            dict_file = f"zoo:pretrained_transformers/poly_model_huge_{opt['polyencoder_init_model']}/model.dict"
        else:
            assert os.path.exists(init_path)
            init_model = init_path
            dict_file = f'{init_path}.dict'

        return {
            'model': 'transformer/polyencoder',
            'init_model': init_model,
            'dict_file': dict_file,
            # necessary opt args
            'multitask_weights': [1],
            **{k: opt[k] for k in POLYENCODER_OPT_KEYS},
            **super().get_reranker_opts(opt),
        }

    def _rescore(
        self, query: torch.LongTensor, docs: List[List[Document]]
    ) -> torch.Tensor:
        """
        Compute Poly-encoder score with initial set of Documents.

        Scoring taken from PolyencoderAgent.score_candidates

        :param query:
            query tokens, used in DPR retrieval.
        :param docs:
            List of initially retrieved top docs for each batch example

        :return new_scores:
            return scored documents.
        """
        poly_query_vec = self.vectorize_texts(
            [self._tokenizer.decode(q) for q in query],
            self.poly_tokenizer,
            self.max_query_len,
        ).to(query.device)

        doc_vecs = torch.stack(
            [
                self.vectorize_texts(
                    [d.get_tokenization_str() for d in docs_i],
                    self.poly_tokenizer,
                    self.max_doc_len,
                )
                for docs_i in docs
            ]
        ).to(query.device)

        ctxt_rep, ctxt_rep_mask, _ = self.polyencoder(ctxt_tokens=poly_query_vec)
        _, _, cand_rep = self.polyencoder(cand_tokens=doc_vecs)
        scores = self.polyencoder(
            ctxt_rep=ctxt_rep, ctxt_rep_mask=ctxt_rep_mask, cand_rep=cand_rep
        )
        return scores


class PolyFaissRetriever(DPRThenPolyRetriever):
    """
    Poly-encoder Retriever, using FAISS.

    Performs FAISS retrieval to retrieve N initial docs; re-ranks according to Poly-
    encoder score to narrow down to K docs.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared = None):
        assert opt['query_model'] == 'dropout_poly'
        super().__init__(opt, dictionary, shared=shared)
        self.dropout_poly = RagDropoutPolyWrapper(opt)
        self.polyencoder = self.dropout_poly.model

        self.poly_tokenizer = RagRetrieverTokenizer(
            opt['datapath'], opt['query_model'], self.dropout_poly.dict, max_length=360
        )

        model = (
            self.polyencoder.module
            if hasattr(self.polyencoder, 'module')
            else self.polyencoder
        )
        for param in model.encoder_cand.parameters():  # type: ignore
            # freeze document encoding for PolyFAISS.
            param.requires_grad = False


@register_agent("rag_tfidf_retriever")
class RagTfidfRetrieverAgent(TfidfRetrieverAgent):
    """
    Wrapper around TFIDF Retriever to cache retrieved documents.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        super().__init__(opt, shared)
        if not shared:
            self.docid_to_text = {}
        else:
            self.docid_to_text = shared.get('docid_to_text', {})

    def share(self) -> TShared:
        shared = super().share()
        shared['docid_to_text'] = self.docid_to_text
        return shared

    def doc2txt(self, docid):
        """
        Cache document texts during train/eval.
        """
        if docid not in self.docid_to_text:
            text = super().doc2txt(docid)
            self.docid_to_text[docid] = text
        else:
            text = self.docid_to_text[docid]
        return text


BLANK_SEARCH_DOC = {'url': None, 'content': '', 'title': ''}
NO_SEARCH_QUERY = 'no_passages_used'


class SearchQueryRetriever(RagRetriever):
    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared):
        RagRetriever.__init__(self, opt, dictionary, shared=shared)
        opt['skip_retrieval_token'] = NO_SEARCH_QUERY
        self.n_docs = opt['n_docs']
        self.len_chunk = opt['splitted_chunk_length']
        self.doc_chunk_split_mode = opt['doc_chunk_split_mode']
        n_doc_chunks = opt['n_ranked_doc_chunks']
        chunk_ranker_type = opt['doc_chunks_ranker']
        if chunk_ranker_type == 'tfidf':
            self.chunk_reranker = TfidfChunkRanker(n_doc_chunks)
        elif chunk_ranker_type == 'head':
            self.chunk_reranker = HeadChunkRanker(n_doc_chunks)
        else:
            self.chunk_reranker = RetrievedChunkRanker(
                n_doc_chunks, opt['woi_doc_chunk_size']
            )

        if not shared:
            self.query_generator = self.init_search_query_generator(opt)
        else:
            self.query_generator = shared['query_generator']
        self.dict = dictionary
        self.init_query_encoder(opt)

    def share(self) -> TShared:
        shared = super().share()
        shared['query_generator'] = self.query_generator
        return shared

    def init_search_query_generator(self, opt) -> TorchGeneratorAgent:
        model_file = opt['search_query_generator_model_file']
        logging.info('Loading search generator model')
        logging.disable()
        search_query_gen_agent = create_agent_from_model_file(
            model_file,
            opt_overrides={
                'skip_generation': False,
                'inference': opt['search_query_generator_inference'],
                'beam_min_length': opt['search_query_generator_beam_min_length'],
                'beam_size': opt['search_query_generator_beam_size'],
                'text_truncate': opt['search_query_generator_text_truncate'],
            },
        )
        logging.enable()
        logging.info('Search query generator model loading completed!')
        return search_query_gen_agent

    def generate_search_query(self, query: torch.LongTensor) -> List[str]:
        """
        Generates a list of queries for the encoded query (context) tensor.
        """
        texts = [self._tokenizer.decode(q) for q in query]
        obs_list = []
        for t in texts:
            msg = Message({'text': t, 'episode_done': True})
            obs_list.append(self.query_generator.observe(msg))
            self.query_generator.reset()  # Erase the history
        search_quries = [r['text'] for r in self.query_generator.batch_act(obs_list)]
        logging.debug(f'Generated search queries {search_quries}')
        return search_quries

    def init_query_encoder(self, opt):
        if hasattr(self, 'query_encoder'):
            # It is already instantiated
            return
        self.query_encoder = DprQueryEncoder(
            opt, dpr_model=opt['query_model'], pretrained_path=opt['dpr_model_file']
        )

    def text2tokens(self, txt: str) -> Union[List[str], List[int]]:
        if self.doc_chunk_split_mode == 'word':
            return txt.split(' ')
        else:
            return self.dict.txt2vec(txt)

    def tokens2text(self, tokens: Union[List[int], List[str]]) -> str:
        if self.doc_chunk_split_mode == 'word':
            return ' '.join(tokens)
        else:
            return self.dict.vec2txt(tokens)

    def pick_chunk(self, query: str, doc_title: str, doc_text: str, doc_url: str):
        """
        Splits the document and returns the selected chunks.

        The number of returned chunks is controlled by `n_ranked_doc_chunks` in opt. The
        chunk selection is determined by `doc_chunks_ranker` in the opt.
        """
        if not doc_text:
            # When there is no search query for the context
            return [("", 0)]
        tokens = self.text2tokens(doc_text)
        if self.opt['doc_chunks_ranker'] != 'woi_chunk_retrieved_docs':
            doc_chunks = [
                self.tokens2text(tokens[i : i + self.len_chunk])
                for i in range(0, len(tokens), self.len_chunk)
            ]
        else:
            doc_chunks = self.tokens2text(tokens)
        return self.chunk_reranker.get_top_chunks(query, doc_title, doc_chunks, doc_url)


class SearchQuerySearchEngineRetriever(SearchQueryRetriever):
    """
    A retriever that uses a search engine server for retrieving documents.

    It instantiates a `SearchEngineRetriever` object that in turns send search queries
    to an external server for retrieving documents.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared):
        super().__init__(opt, dictionary, shared)
        if not shared:
            self.search_client = self.initiate_retriever_api(opt)
        else:
            self.search_client = shared['search_client']

    def share(self) -> TShared:
        shared = super().share()
        shared['search_client'] = self.search_client
        return shared

    def initiate_retriever_api(self, opt) -> SearchEngineRetriever:
        logging.info('Creating the search engine retriever.')
        return SearchEngineRetriever(opt)

    def _empty_docs(self, num: int):
        """
        Generates the requested number of empty documents.
        """
        return [BLANK_SEARCH_DOC for _ in range(num)]

    def rank_score(self, rank_id: int):
        """
        Scores the chunks of the retrieved document based on their rank.

        Note that this is the score for the retrieved document and applies to all its
        chunks.
        """
        return 1 / (1 + rank_id)

    def _display_urls(self, search_results):
        """
        Generates a string that lists retrieved URLs (document IDs).
        """
        return '\n'.join([d['url'] for d in search_results if d['url']])

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieves relevant documents for the query (the conversation context). This
        method conducts three main steps that are flagged in the main code as well.

        Step 1: generate search queries for the conversation context batch.This step
        uses the query generator model (self.query_generator).

        Step 2: use the search client to retrieve documents.This step uses retrieval
        API agent (self.search_client)

        Step 3: generate the list of Document objects from the
        retrieved content. Here if the documents too long, the code splits them and
        chooses a chunk based on the selected `doc_chunks_ranker` in the opt.
        """
        # step 1
        search_queries = self.generate_search_query(query)

        # step 2
        search_results_batch = self.search_client.retrieve(search_queries, self.n_docs)

        # step 3
        top_docs = []
        top_doc_scores = []
        max_n_docs: int = self.n_docs
        for sq, search_results in zip(search_queries, search_results_batch):
            if not search_results:
                search_results = self._empty_docs(self.n_docs)
            elif len(search_results) < self.n_docs:
                remain_docs = self.n_docs - len(search_results)
                search_results.extend(self._empty_docs(remain_docs))
            docs_i = []
            scors_i = []
            # Change this debug later
            logging.debug(f'URLS:\n{self._display_urls(search_results)}')
            for i, doc in enumerate(search_results):
                url = doc['url']
                title = doc['title']
                dcontent = doc['content']
                assert type(dcontent) in (
                    str,
                    list,
                ), f'Unrecognized retrieved doc: {dcontent}'
                full_text = (
                    dcontent if isinstance(dcontent, str) else '\n'.join(doc['content'])
                )
                doc_chunks = [
                    dc[0] for dc in self.pick_chunk(sq, title, full_text, url)
                ]
                for splt_id, splt_content in enumerate(doc_chunks):
                    docs_i.append(
                        Document(
                            docid=url, text=splt_content, title=f'{title}_{splt_id}'
                        )
                    )
                    scors_i.append(self.rank_score(i))
            max_n_docs = max(max_n_docs, len(docs_i))
            top_docs.append(docs_i)
            top_doc_scores.append(scors_i)
        # Pad with empty docs
        for i in range(len(top_docs)):
            n_empty = max_n_docs - len(top_docs[i])
            if n_empty:
                top_docs[i] = top_docs[i] + [BLANK_DOC] * n_empty
                top_doc_scores[i] = top_doc_scores[i] + [0] * n_empty
        self.top_docs = top_docs
        return top_docs, torch.Tensor(top_doc_scores).to(query.device)


class SearchQueryFAISSIndexRetriever(SearchQueryRetriever, DPRRetriever):
    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared):
        SearchQueryRetriever.__init__(self, opt, dictionary, shared=shared)
        self.load_index(opt, shared)

    def share(self) -> TShared:
        shared = SearchQueryRetriever.share(self)
        shared.update(DPRRetriever.share(self))
        return shared

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieves from the FAISS index using a search query.

        This methods relies on the `retrieve_and_score` method in `RagRetriever`
        ancestor class. It receive the query (conversation context) and generatess the
        search term queries based on them. Then uses those search quries (instead of the
        the query text itself) to retrieve from the FAISS index.
        """

        search_queries = self.generate_search_query(query)
        tokenized_search_queries, _ = padded_tensor(
            [self._tokenizer.encode(sq) for sq in search_queries]
        )
        top_docs, top_doc_scores = DPRRetriever.retrieve_and_score(
            self, tokenized_search_queries.to(query.device)
        )
        for query_id in range(len(top_docs)):
            if search_queries[query_id] == NO_SEARCH_QUERY:
                top_docs[query_id] = [BLANK_DOC for _ in range(self.n_docs)]
        return top_docs, top_doc_scores


class ObservationEchoRetriever(RagRetriever):
    """
    This retriever returns (echos) documents that are already passed to it to return.

    Use this only with GoldFiD agents. It relies on the retrieved docs being included in
    the observed example of the agent.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared: TShared = None):
        self._delimiter = '\n'
        self.n_docs = opt['n_docs']
        self._query_ids = dict()
        self._saved_docs = dict()
        self._largest_seen_idx = -1
        super().__init__(opt, dictionary, shared=shared)

    def add_retrieve_doc(self, query: str, retrieved_docs: List[Document]):
        self._largest_seen_idx += 1
        new_idx = self._largest_seen_idx
        if new_idx in self._query_ids.values() or new_idx in self._saved_docs:
            raise RuntimeError(
                "Nonunique new_idx created in add_retrieve_doc in ObservationEchoRetriever \n"
                "this might return the same set of docs for two distinct queries"
            )
        self._query_ids[query] = new_idx
        self._saved_docs[new_idx] = retrieved_docs or [
            BLANK_DOC for _ in range(self.n_docs)
        ]

    def tokenize_query(self, query: str) -> List[int]:
        return [self._query_ids[query]]

    def get_delimiter(self) -> str:
        return self._delimiter

    def clear_mapping(self):
        self._query_ids = dict()
        self._saved_docs = dict()
        self._largest_seen_idx = -1

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        batch_size = query.size(0)

        retrieved_docs = []
        for endoded_query in query.tolist():
            docs_retrieve_idx = endoded_query[0]
            retrieved_docs.append(self._saved_docs[docs_retrieve_idx])

        # Some arbitrary scoring of docs
        max_num_docs = max([len(rtds) for rtds in retrieved_docs])
        retrieved_doc_scores = torch.Tensor([1 / (1 + i) for i in range(max_num_docs)])
        retrieved_doc_scores = retrieved_doc_scores.repeat(batch_size, 1).to(
            query.device
        )

        return retrieved_docs, retrieved_doc_scores


class DocumentChunkRanker:
    """
    Base class for controlling splitting long documents and selecting relevant chunks.
    """

    def __init__(self, n_retrieved_chunks):
        self.n_ret_chunks = n_retrieved_chunks

    @abstractmethod
    def get_top_chunks(
        self,
        query: str,
        doc_title: str,
        doc_chunks: Union[List[str], str],
        doc_url: str,
    ):
        """
        Ranks documents (chunk) based on their relevance to `query`
        """


class HeadChunkRanker(DocumentChunkRanker):
    """
    Returns the head chunks only.
    """

    def get_top_chunks(
        self,
        query: str,
        doc_title: str,
        doc_chunks: Union[List[str], str],
        doc_url: str,
    ):
        """
        Return chunks in doc-present order.
        """
        return [(c,) for c in doc_chunks[: self.n_ret_chunks]]


class RetrievedChunkRanker(DocumentChunkRanker):
    """
    Utilize retrieved doc chunk mutator.
    """

    def __init__(self, n_retrieved_chunks, chunk_size: int = 500):
        super().__init__(n_retrieved_chunks)
        self.chunk_size = chunk_size

    def get_top_chunks(
        self,
        query: str,
        doc_title: str,
        doc_chunks: Union[List[str], str],
        doc_url: str,
    ):
        """
        Return chunks according to the woi_chunk_retrieved_docs_mutator.
        """
        if isinstance(doc_chunks, list):
            docs = ''.join(doc_chunks)
        else:
            assert isinstance(doc_chunks, str)
            docs = doc_chunks
        chunks = chunk_docs_in_message(
            Message(
                {
                    CONST.RETRIEVED_DOCS: [docs],
                    CONST.RETRIEVED_DOCS_TITLES: [doc_title],
                    CONST.RETRIEVED_DOCS_URLS: [doc_url],
                    CONST.SELECTED_SENTENCES: [CONST.NO_SELECTED_SENTENCES_TOKEN],
                }
            ),
            self.chunk_size,
        )[CONST.RETRIEVED_DOCS]
        return [(c,) for c in chunks[: self.n_ret_chunks]]


class TfidfChunkRanker(DocumentChunkRanker):
    """
    Uses TF-IDF to compare chunks to the original search query.
    """

    def __init__(self, n_retrieved_chunks):
        super().__init__(n_retrieved_chunks)
        self._vectorizer = TfidfVectorizer()

    def get_top_chunks(
        self,
        query: str,
        doc_title: str,
        doc_chunks: Union[List[str], str],
        doc_url: str,
    ):
        assert isinstance(doc_chunks, list)
        vectorized_corpus = self._vectorizer.fit_transform(doc_chunks + [query])
        docs_vec = vectorized_corpus[:-1, :]
        q_vec = vectorized_corpus[-1, :]
        scores = np.hstack((q_vec * docs_vec.transpose()).toarray())
        top_chunk_ids = np.argsort(-scores)[: self.n_ret_chunks]
        return [(doc_chunks[i], scores[i]) for i in top_chunk_ids]


def retriever_factory(
    opt: Opt, dictionary: DictionaryAgent, shared=None
) -> Optional[RagRetriever]:
    """
    Build retriever.

    :param opt:
        ParlAI Opt
    :param dictionary:
        dictionary agent
    :param shared:
        shared objects.

    :return retriever:
        return a retriever for RAG.
    """
    if opt.get('converting'):
        return None
    # only build retriever when not converting a BART model
    retriever = RetrieverType(opt['rag_retriever_type'])
    if retriever is RetrieverType.DPR:
        return DPRRetriever(opt, dictionary, shared=shared)
    elif retriever is RetrieverType.TFIDF:
        return TFIDFRetriever(opt, dictionary, shared=shared)
    elif retriever is RetrieverType.DPR_THEN_POLY:
        return DPRThenPolyRetriever(opt, dictionary, shared=shared)
    elif retriever is RetrieverType.POLY_FAISS:
        return PolyFaissRetriever(opt, dictionary, shared=shared)
    elif retriever is RetrieverType.SEARCH_ENGINE:
        return SearchQuerySearchEngineRetriever(opt, dictionary, shared=shared)
    elif retriever is RetrieverType.SEARCH_TERM_FAISS:
        return SearchQueryFAISSIndexRetriever(opt, dictionary, shared=shared)
    elif retriever is RetrieverType.OBSERVATION_ECHO_RETRIEVER:
        return ObservationEchoRetriever(opt, dictionary, shared=shared)
