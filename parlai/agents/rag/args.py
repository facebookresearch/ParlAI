#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RAG Args.

An area to maintain RAG-Related Arguments
"""
from enum import Enum
from parlai.core.params import ParlaiParser

QUERY_MODEL_TYPES = ['bert', 'bert_from_parlai_rag', 'dropout_poly']
PRETRAINED_RANKER_TYPES = ['wikito', 'reddit']
SMALL_INDEX_TYPES = [None, 'none', 'exact', 'compressed']

DPR_ZOO_MODEL = 'zoo:hallucination/multiset_dpr/hf_bert_base.cp'
TFIDF_ZOO_MODEL = 'zoo:wikipedia_full/tfidf_retriever/model'
WIKIPEDIA_ZOO_PASSAGES = 'zoo:hallucination/wiki_passages/psgs_w100.tsv'
WIKIPEDIA_COMPRESSED_INDEX = 'zoo:hallucination/wiki_index_compressed/compressed_pq'
WIKIPEDIA_EXACT_INDEX = 'zoo:hallucination/wiki_index_exact/exact'

WOW_PASSAGES_PATH = 'zoo:hallucination/wow_passages/wow_articles.paragraphs.tsv'
WOW_INDEX_PATH = 'zoo:hallucination/wow_passages/exact'
WOW_COMPRESSED_INDEX_PATH = 'zoo:hallucination/wow_passages/compressed'

POLYFAISS_ZOO_MODEL = 'zoo:hallucination/dropout_poly/model'
RAG_TOKEN_ZOO_MODEL = 'zoo:hallucination/bart_rag_token/model'
RAG_SEQUENCE_ZOO_MODEL = 'zoo:hallucination/bart_rag_sequence/model'
RAG_TURN_DTT_ZOO_MODEL = 'zoo:hallucination/bart_rag_turn_dtt/model'
RAG_TURN_DO_ZOO_MODEL = 'zoo:hallucination/bart_rag_turn_do/model'
RAG_DPR_POLY_ZOO_MODEL = 'zoo:hallucination/bart_rag_dpr_poly/model'
FID_DPR_ZOO_MODEL = 'zoo:hallucination/bart_fid_dpr/model'
FID_RAG_ZOO_MODEL = 'zoo:hallucination/bart_fid_rag/model'
FID_RAG_DPR_POLY_ZOO_MODEL = 'zoo:hallucination/bart_fid_rag_dpr_poly/model'

# the following args are copied from
# zoo:pretrained_transformers/poly_model_huge_wikito/model.opt
TRANSFORMER_RANKER_BASE_OPT = {
    'text_truncate': 360,
    'label_truncate': 360,
    'history_size': 20,
    'variant': 'xlm',
    'n_layers': 12,
    'n_heads': 12,
    'n_positions': 1024,
    'embedding_size': 768,
    'ffn_size': 3072,
    'activation': 'gelu',
    'attention_dropout': 0.1,
    'relu_dropout': 0.0,
    'dropout': 0.1,
    'n_segments': 2,
    'learn_embeddings': True,
    'dict_endtoken': '__start__',
    'reduction_type': 'mean',
    'dict_tokenizer': 'bpe',
    'dict_lower': True,
    'output_scaling': 0.06,
    'share_encoders': False,
    'learn_positional_embeddings': True,
    'embeddings_scale': False,
}


# Options for poly-encoders
POLYENCODER_OPT_KEYS = [
    'polyencoder_type',
    'poly_n_codes',
    'poly_attention_type',
    'poly_n_codes',
    'poly_attention_type',
    'poly_attention_num_heads',
    'codes_attention_type',
    'codes_attention_num_heads',
]


class RetrieverType(Enum):
    """
    Rag Retriever Types.
    """

    DPR = 'dpr'
    TFIDF = 'tfidf'
    DPR_THEN_POLY = 'dpr_then_poly'
    POLY_FAISS = 'poly_faiss'
    SEARCH_ENGINE = 'search_engine'
    SEARCH_TERM_FAISS = 'search_term_faiss'
    OBSERVATION_ECHO_RETRIEVER = 'observation_echo_retriever'


def setup_rag_args(parser: ParlaiParser) -> ParlaiParser:
    group = parser.add_argument_group('RAG Model Args')
    # Standard RAG Agent Arguments
    group.add_argument(
        '--generation-model',
        type=str,
        default='bart',
        help='which generation model to use',
        choices=['transformer/generator', 'bart', 't5'],
    )
    group.add_argument(
        '--query-model',
        type=str,
        default='bert',
        help='Which query model to use for DPR.',
        choices=QUERY_MODEL_TYPES,
    )
    group.add_argument(
        '--rag-model-type',
        type=str,
        default='token',
        help='which rag model decoding to use.',
        choices=['token', 'sequence', 'turn'],
    )
    group.add_argument(
        '--thorough',
        type='bool',
        default=False,
        help='whether to use thorough decoding for rag sequence. ',
    )
    modified_group = parser.add_argument_group('Modified RAG Args')
    modified_group.add_argument(
        '--n-extra-positions',
        type=int,
        default=0,
        help='Specify > 0 to include extra positions in the encoder, in which '
        'retrieved knowledge will go. In this setup, knowledge is _appended_ '
        'instead of prepended.',
    )
    modified_group.add_argument(
        '--gold-knowledge-passage-key',
        type=str,
        default='checked_sentence',
        help='key in the observation dict that indicates the gold knowledge passage. '
        'Specify, along with --debug, to compute passage retrieval metrics at train/test time.',
    )
    modified_group.add_argument(
        '--gold-knowledge-title-key',
        type=str,
        default='title',
        help='key in the observation dict that indicates the gold knowledge passage title. '
        'Specify, along with --debug, to compute passage retrieval metrics at train/test time.',
    )
    retriever_group = parser.add_argument_group('RAG Retriever Args')
    retriever_group.add_argument(
        '--rag-retriever-query',
        type=str,
        default='full_history',
        choices=['one_turn', 'full_history'],
        help='What to use as the query for retrieval. `one_turn` retrieves only on the last turn '
        'of dialogue; `full_history` retrieves based on the full dialogue history.',
    )
    retriever_group.add_argument(
        '--rag-retriever-type',
        type=str,
        default=RetrieverType.DPR.value,
        choices=[r.value for r in RetrieverType],
        help='Which retriever to use',
    )
    retriever_group.add_argument(
        '--retriever-debug-index',
        type=str,
        default=None,
        choices=SMALL_INDEX_TYPES,
        help='Load specified small index, for debugging.',
    )
    retriever_group.add_argument(
        '--n-docs', type=int, default=5, help='How many documents to retrieve'
    )
    retriever_group.add_argument(
        '--min-doc-token-length',
        type=int,
        default=64,
        help='minimum amount of information to retain from document. '
        'Useful to define if encoder does not use a lot of BPE token context.',
    )
    retriever_group.add_argument(
        '--max-doc-token-length',
        type=int,
        default=256,
        help='maximum amount of information to retain from document. ',
    )
    retriever_group.add_argument(
        '--rag-query-truncate',
        type=int,
        default=512,
        help='Max token length of query for retrieval.',
    )
    retriever_group.add_argument(
        '--print-docs',
        type='bool',
        default=False,
        help='Whether to print docs; usually useful during interactive mode.',
    )
    dense_retriever_group = parser.add_argument_group(
        'RAG Dense Passage Retriever Args'
    )
    dense_retriever_group.add_argument(
        '--path-to-index',
        type=str,
        default=WIKIPEDIA_COMPRESSED_INDEX,
        help='path to FAISS Index.',
    )
    dense_retriever_group.add_argument(
        '--path-to-dense-embeddings',
        type=str,
        default=None,
        help='path to dense embeddings directory used to build index. '
        'Default None will assume embeddings and index are in the same directory.',
    )
    dense_retriever_group.add_argument(
        '--dpr-model-file', type=str, default=DPR_ZOO_MODEL, help='path to DPR Model.'
    )
    dense_retriever_group.add_argument(
        '--path-to-dpr-passages',
        type=str,
        default=WIKIPEDIA_ZOO_PASSAGES,
        help='Path to DPR passages, used to build index.',
    )
    dense_retriever_group.add_argument(
        '--retriever-embedding-size',
        type=int,
        default=768,
        help='Embedding size of dense retriever',
    )
    tfidf_retriever_group = parser.add_argument_group('RAG TFIDF Retriever Args')
    tfidf_retriever_group.add_argument(
        '--tfidf-max-doc-paragraphs',
        type=int,
        default=-1,
        help='If > 0, limit documents to this many paragraphs',
    )
    tfidf_retriever_group.add_argument(
        '--tfidf-model-path',
        type=str,
        default=TFIDF_ZOO_MODEL,
        help='Optionally override TFIDF model.',
    )
    dpr_poly_retriever_group = parser.add_argument_group('RAG DPR-POLY Retriever Args')
    dpr_poly_retriever_group.add_argument(
        '--dpr-num-docs',
        type=int,
        default=25,
        help='In two stage retrieval, how many DPR documents to retrieve',
    )
    dpr_poly_retriever_group.add_argument(
        '--poly-score-initial-lambda',
        type=float,
        default=0.5,
        help='In two stage retrieval, how much weight to give to the poly scores. '
        'Note: Learned parameter. Specify initial value here',
    )
    dpr_poly_retriever_group.add_argument(
        '--polyencoder-init-model',
        type=str,
        default='wikito',
        help='Which init model to initialize polyencoder with. Specify wikito or reddit to use '
        'models from the ParlAI zoo; otherwise, provide a path to a trained polyencoder',
    )
    poly_faiss_group = parser.add_argument_group('RAG PolyFAISS retriever args')
    poly_faiss_group.add_argument(
        '--poly-faiss-model-file',
        type=str,
        default=None,
        help='path to poly-encoder for use in poly-faiss retrieval.',
    )
    regret_group = parser.add_argument_group("RAG ReGReT args")
    regret_group.add_argument(
        '--regret',
        type='bool',
        default=False,
        help='Retrieve, Generate, Retrieve, Tune. '
        'Retrieve, generate, then retrieve again, and finally tune (refine).',
    )
    regret_group.add_argument(
        '--regret-intermediate-maxlen',
        type=int,
        default=32,
        help='Maximum length in intermediate regret generation',
    )
    regret_group.add_argument(
        '--regret-model-file',
        type=str,
        default=None,
        help='Path to model for initial round of retrieval. ',
    )
    regret_group.add_argument(
        '--regret-dict-file',
        type=str,
        default=None,
        help='Path to dict file for model for initial round of retrieval. ',
    )
    regret_group.add_argument(
        '--regret-override-index',
        type='bool',
        default=False,
        help='Overrides the index used with the ReGReT model, if using separate models. '
        'I.e., the initial round of retrieval uses the same index as specified for the '
        'second round of retrieval',
    )
    indexer_group = parser.add_argument_group("RAG Indexer Args")
    indexer_group.add_argument(
        '--indexer-type',
        type=str,
        default='compressed',
        choices=['exact', 'compressed'],
        help='Granularity of RAG Indexer. Choose compressed to save on RAM costs, at the '
        'possible expense of accuracy.',
    )
    indexer_group.add_argument(
        '--indexer-buffer-size',
        type=int,
        default=65536,
        help='buffer size for adding vectors to the index',
    )
    indexer_group.add_argument(
        '--compressed-indexer-factory',
        type=str,
        default='IVF4096_HNSW128,PQ128',
        help='If specified, builds compressed indexer from a FAISS Index Factory. '
        'see https://github.com/facebookresearch/faiss/wiki/The-index-factory for details',
    )
    indexer_group.add_argument(
        '--compressed-indexer-gpu-train',
        type='bool',
        default=False,
        hidden=True,
        help='Set False to not train compressed indexer on the gpu.',
    )
    indexer_group.add_argument(
        '--compressed-indexer-nprobe',
        type=int,
        default=64,
        help='How many centroids to search in compressed indexer. See '
        'https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#cell-probe-methods-indexivf-indexes '
        'for details',
    )
    # See https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#indexhnsw-variants for details
    indexer_group.add_argument(
        '--hnsw-indexer-store-n',
        type=int,
        default=128,
        hidden=True,
        help='Granularity of  DenseHNSWIndexer. Higher == more accurate, more RAM',
    )
    indexer_group.add_argument(
        '--hnsw-ef-search',
        type=int,
        default=128,
        hidden=True,
        help='Depth of exploration of search for HNSW.',
    )
    indexer_group.add_argument(
        '--hnsw-ef-construction',
        type=int,
        default=200,
        hidden=True,
        help='Depth of exploration at add time for HNSW',
    )
    return parser
