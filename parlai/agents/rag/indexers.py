#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
FAISS-based Indexers.

Adapted from https://github.com/facebookresearch/DPR/blob/main/dpr/indexer/faiss_indexers.py
"""
from parlai.core.build_data import modelzoo_path
from parlai.core.opt import Opt
import parlai.utils.logging as logging

from abc import ABC, abstractmethod
import math
import numpy as np
import os
import random
import time
import torch
import torch.cuda
from typing import List, Tuple, Optional

from parlai.agents.rag.args import WIKIPEDIA_COMPRESSED_INDEX, WIKIPEDIA_EXACT_INDEX


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.

    From https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    from itertools import zip_longest

    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class BaseIndexer(ABC):
    """
    Top-Level Indexer.

    Provides an interface for interacting with FAISS Indexes.
    """

    def __init__(self, opt: Opt):
        self.buffer_size = opt['indexer_buffer_size']
        self.index_id_to_db_id = []
        self.index = None
        try:
            import faiss  # noqa: f401

            self.faiss = faiss
        except ImportError:
            raise ImportError(
                'Please install faiss: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md'
            )

    @abstractmethod
    def index_data(self, data: List[torch.Tensor]):
        """
        Index data.

        Given a list of tensors, construct a FAISS Index.

        :param data:
            list of torch.Tensors
        """

    def get_search_vectors(self, query_vectors: np.array) -> np.array:
        """
        Given array of query vectors, return the vectors used for search.

        Allows subclasses to modify the vectors at their discretion.

        :param query_vectors:
            q vecs to modify

        :return search vectors:
            return vectors used for FAISS.
        """
        return query_vectors

    def search(
        self, query_vectors: np.array, top_docs: int
    ) -> List[Tuple[List[int], List[np.array]]]:
        """
        Search FAISS index.

        :param query_vectors:
            query vectors into the index
        :param top_docs:
            number of docs to return

        :return top_docs:
            returns, for each query vector:
                a list of document ids (according to db),
                a list of reconstructed document vectors
        """
        query_vectors = self.get_search_vectors(query_vectors)
        logging.debug(f'query_vectors {query_vectors.shape}')
        _scores, indexes, vectors = self.index.search_and_reconstruct(
            query_vectors, top_docs
        )
        db_ids = [
            [self.index_id_to_db_id[i] for i in query_top_idxs]
            for query_top_idxs in indexes
        ]
        result = [(db_ids[i], vectors[i]) for i in range(len(db_ids))]
        return result

    def serialize(self, file: str):
        """
        Serialize index into file.

        :param file:
            output file.
        """
        logging.info(f'Serializing index to {file}')

        if os.path.isdir(file):
            index_file = os.path.join(file, "index")
            meta_file = os.path.join(file, "index_meta")
        else:
            index_file = f'{file}.index'
            meta_file = f'{file}.index_meta'

        self.faiss.write_index(self.index, index_file)
        if self.index_id_to_db_id:
            torch.save(self.index_id_to_db_id, meta_file)

    def deserialize_from(self, file: str, emb_path: Optional[str] = None):
        """
        Deserialize index from file.

        :param file:
            input file
        :param emb_path:
            optional path to embeddings
        """
        logging.info(f'Loading index from {file}')

        if os.path.isdir(file):
            index_file = os.path.join(file, "index")
            meta_file = os.path.join(file, "index_meta")
        elif not file.endswith('.index'):
            index_file = f'{file}.index'
            meta_file = f'{file}.index_meta'
        else:
            index_file = file
            meta_file = f'{index_file}_meta'

        self.index = self.faiss.read_index(index_file)
        logging.info(f'Loaded index of type {self.index} and size {self.index.ntotal}')

        if os.path.exists(meta_file):
            self.index_id_to_db_id = torch.load(meta_file)
        else:
            index_dir = os.path.split(file)[0] if emb_path is None else emb_path
            if not os.path.isdir(index_dir):
                # if emb_path has the embeddings name in there, need to split.
                index_dir = os.path.split(index_dir)[0]
            meta_files = [f for f in os.listdir(index_dir) if f.startswith('ids_')]
            meta_files = sorted(meta_files, key=lambda x: int(x.split('_')[-1]))
            for f in meta_files:
                ids = torch.load(os.path.join(index_dir, f))
                self.index_id_to_db_id.extend(ids)
            torch.save(self.index_id_to_db_id, meta_file)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), 'Deserialized index_id_to_db_id should match faiss index size '
        f'{len(self.index_id_to_db_id)} != {self.index.ntotal}'


class DenseHNSWFlatIndexer(BaseIndexer):
    """
    Indexer for Dense, HNSW FLAT Index.

    High retrieval accuracy comes at the cost of High RAM Usage.
    """

    def __init__(self, opt: Opt):
        super().__init__(opt)
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = self.faiss.IndexHNSWFlat(
            opt['retriever_embedding_size'] + 1, opt['hnsw_indexer_store_n']
        )
        index.hnsw.efSearch = opt['hnsw_ef_search']
        index.hnsw.efConstruction = opt['hnsw_ef_construction']
        self.index = index
        self.built = False

    def index_data(self, tensors: List[torch.Tensor]):
        """
        Index data.

        The HNSW Flat Indexer computes an auxiliary dimension that converts inner product
        similarity to L2 distance similarity.

        :param data:
            List of torch.Tensor
        """
        data = torch.cat(tensors).float()
        n = data.size(0)
        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.built:
            raise RuntimeError(
                'HNSW index needs to index all data at once, results will be unpredictable otherwise.'
            )
        phi = 0
        norms = (data**2).sum(dim=1)
        max_norms = norms.max().item()
        phi = max(phi, max_norms)
        logging.info(f'HNSWF DotProduct -> L2 space phi={phi}')
        start = time.time()

        for i in range(0, n, self.buffer_size):
            vectors_i = data[i : i + self.buffer_size]
            norms_i = norms[i : i + self.buffer_size]
            aux_dims = torch.sqrt(phi - norms_i)
            hnsw_vectors = torch.cat([vectors_i, aux_dims.unsqueeze(1)], dim=1)
            self.index.add(hnsw_vectors.numpy())
            logging.info(
                f'{time.time() - start}s Elapsed: data indexed {i + len(vectors_i)}'
            )

        logging.info(f'Total data indexed {n}')

    def get_search_vectors(self, query_vectors: np.array) -> np.array:
        """
        Add an additional dimension to the query vectors to account for cosine
        similarity.

        :param query_vectors:
            query vectors of dimension [n_search, q_dim]

        :return search_vectors:
            search vectors of dimension [n_search, q_dim + 1]
        """
        aux_dim = np.zeros(len(query_vectors), dtype='float32')
        query_hnsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        return query_hnsw_vectors

    def deserialize_from(self, file: str, emb_path: Optional[str] = None):
        super().deserialize_from(file, emb_path)
        # to trigger warning on subsequent indexing
        self.built = True


class CompressedIndexer(BaseIndexer):
    """
    Compressed Indexer.

    If a FAISS index factory is specified, we build that.
    (see https://github.com/facebookresearch/faiss/wiki/The-index-factory)

    The default is IVF4096_HNSW128,PQ128; this also translates directly to the
    default when index_factory is specified as ''.

    See https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#cell-probe-methods-indexivf-indexes
    for more details.
    """

    def __init__(self, opt: Opt):
        """
        Initialize IVFPQ FAISS Indexer.

        The IVFPQ Indexer is a great way to reduce memory footprint of dense embeddings.
        """
        super().__init__(opt)
        self.dim = opt['retriever_embedding_size']
        self.use_gpu_train = (
            not opt['no_cuda']
            and torch.cuda.is_available()
            and opt['compressed_indexer_gpu_train']
        )
        self.hnsw_ef_search = opt['hnsw_ef_search']

        self.index_factory = opt['compressed_indexer_factory']
        if self.index_factory:
            logging.warning(f'Creating Index from Index Factory: {self.index_factory}')
            self.is_ivf_index = 'IVF' in self.index_factory
            self.index = self.faiss.index_factory(
                self.dim, self.index_factory, self.faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.is_ivf_index = True
            quantizer = self.faiss.IndexHNSWFlat(
                self.dim, opt['hnsw_indexer_store_n'], self.faiss.METRIC_INNER_PRODUCT
            )
            quantizer.hnsw.efConstruction = opt['hnsw_ef_construction']
            quantizer.hnsw.efSearch = opt['hnsw_ef_search']
            ivf_index = self.faiss.IndexIVFPQ(
                quantizer, self.dim, 4096, 128, 8, self.faiss.METRIC_INNER_PRODUCT
            )
            ivf_index.nprobe = opt['compressed_indexer_nprobe']
            self.index = ivf_index

        if self.is_ivf_index:
            self.index_ivf = self.faiss.extract_index_ivf(self.index)
            self.index_ivf.metric_type = self.faiss.METRIC_INNER_PRODUCT
            self.nlist = self.index_ivf.nlist
            self.index_ivf.verbose = True
            self.downcast_quantizer = self.faiss.downcast_index(
                self.index_ivf.quantizer
            )
            self.downcast_quantizer.verbose = True
            self.downcast_quantizer.metric_type = self.faiss.METRIC_INNER_PRODUCT
            if hasattr(self.downcast_quantizer, 'hnsw'):
                self.downcast_quantizer.hnsw.efSearch = opt['hnsw_ef_search']
                self.downcast_quantizer.hnsw.efConstruction = opt[
                    'hnsw_ef_construction'
                ]
                self.downcast_quantizer.hnsw.metric_type = (
                    self.faiss.METRIC_INNER_PRODUCT
                )

            self.setup_gpu_train()
            self.index.nprobe = opt['compressed_indexer_nprobe']

        self.nprobe = opt['compressed_indexer_nprobe']
        self.span = 5  # arbitrarily chosen, from prior evidence
        self.random = random.Random(42)

    def setup_gpu_train(self):
        """
        Setup training on the gpu.
        """
        if self.use_gpu_train:
            logging.warning('Will train index on GPU')
            try:
                clustering_index = self.faiss.index_cpu_to_all_gpus(
                    self.faiss.IndexFlatIP(self.index_ivf.d)
                )
                self.index.clustering_index = clustering_index
            except NameError:
                logging.warning('GPU training not supported; switching to CPU.')

    def train(self, vectors: List[torch.Tensor]):
        """
        Train clustering index on list of vectors.

        :param vectors:
            a list of tensors to train on
        """
        start = time.time()
        for i, vecs in enumerate(grouper(vectors, self.span, None)):
            vec = torch.cat([v for v in vecs if v is not None])
            # sample
            num_samples = (
                min(50 * self.nlist, vec.size(0))
                if hasattr(self, 'n_list')
                else vec.size(0) // 2
            )
            vec = vec[torch.LongTensor(random.sample(range(vec.size(0)), num_samples))]
            logging.info(
                f'Training data {i+1}/{math.ceil(len(vectors) / self.span)} of shape {vec.shape}'
            )
            self.index.train(vec.float().numpy())
            logging.info(
                f'{time.time() - start:.2f}s Elapsed: training complete for {i+1}'
            )

    def add(self, vectors: List[torch.Tensor]):
        """
        Add vectors to index, using the CPU.

        :param vectors:
            vectors to add.
        """
        start = time.time()
        for i, vecs in enumerate(grouper(vectors, self.span, None)):
            vec = torch.cat([v for v in vecs if v is not None])
            logging.info(
                f'Adding data {(i+1)}/{math.ceil(len(vectors) / self.span)} of shape: {vec.shape}'
            )
            self.index.add(vec.float().numpy())
            logging.info(f'{time.time() - start}s Elapsed: adding complete for {i+1}')

    def index_data(self, data: List[torch.Tensor]):
        """
        Index data.

        :param data:
            list of (db_id, np.vector) tuples
        """
        start = time.time()
        assert isinstance(data, list)
        logging.info(f'Indexing {sum(v.size(0) for v in data)} vectors')
        # First, train
        self.train(data)

        # then, Add
        self.add(data)
        logging.info(f'Indexing complete; total time elapsed: {time.time() - start}')

    def deserialize_from(self, file: str, emb_path: Optional[str] = None):
        """
        If we're using an IVF index, we need to reset the nprobe parameter.

        `Make Direct Map` allows us to reconstruct vectors as well.
        """
        super().deserialize_from(file, emb_path)
        if self.is_ivf_index:
            self.index_ivf = self.faiss.extract_index_ivf(self.index)
            self.index_ivf.make_direct_map()
            self.index_ivf.nprobe = self.nprobe
            self.index.nprobe = self.nprobe


def indexer_factory(opt: Opt) -> BaseIndexer:
    """
    Build indexer.

    :param opt:
        Options

    :return indexer:
        return build indexer, according to options
    """
    if opt['indexer_type'] == 'compressed':
        if opt['path_to_index'] == WIKIPEDIA_EXACT_INDEX:
            logging.warning(
                f'Changing index path to compressed index: {WIKIPEDIA_COMPRESSED_INDEX}'
            )
            opt['path_to_index'] = modelzoo_path(
                opt['datapath'], WIKIPEDIA_COMPRESSED_INDEX
            )
        indexer = CompressedIndexer(opt)
    elif opt['indexer_type'] == 'exact':
        if opt['path_to_index'] == WIKIPEDIA_COMPRESSED_INDEX:
            logging.warning(
                f'Changing index path to exact index: {WIKIPEDIA_EXACT_INDEX}'
            )
            opt['path_to_index'] = modelzoo_path(opt['datapath'], WIKIPEDIA_EXACT_INDEX)
        indexer = DenseHNSWFlatIndexer(opt)
    else:
        raise ValueError(f"Unsupported indexer type: {opt['indexer_type']}")

    return indexer
