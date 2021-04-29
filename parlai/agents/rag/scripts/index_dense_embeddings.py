#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Create a FAISS Index with a series of dense embeddings.
"""
import os
import random
import torch
from typing import List

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
import parlai.utils.logging as logging

from parlai.agents.rag.rag import RagAgent
from parlai.agents.rag.indexers import indexer_factory, CompressedIndexer


class Indexer(ParlaiScript):
    """
    Index Dense Embeddings.
    """

    @classmethod
    def setup_args(cls):
        """
        Setup args.
        """
        parser = ParlaiParser(True, True, 'Index Dense Embs')
        parser.add_argument(
            '--embeddings-dir', type=str, help='directory of embeddings'
        )
        parser.add_argument(
            '--embeddings-name', type=str, default='', help='name of emb part'
        )
        parser.add_argument(
            '--partition-index',
            type='bool',
            default=False,
            help='specify True to partition indexing per file (useful when all files do not fit into memory)',
        )
        parser.add_argument(
            '--save-index-dir',
            type=str,
            help='directory in which to save index',
            default=None,
        )
        parser.add_argument(
            '--num-shards',
            type=int,
            default=1,
            help='how many workers to use to split up the work',
        )
        parser.add_argument(
            '--shard-id',
            type=int,
            help='shard id for this worker. should be between 0 and num_shards',
        )
        parser = RagAgent.add_cmdline_args(parser)
        parser.set_defaults(compressed_indexer_gpu_train=True)
        return parser

    def run(self):
        """
        Load dense embeddings and index with FAISS.
        """
        # create index
        index_dir = self.opt['embeddings_dir']
        embs_name = (
            f"{self.opt['embeddings_name']}_" if self.opt['embeddings_name'] else ''
        )
        num_parts = len(
            [
                f
                for f in os.listdir(index_dir)
                if f.endswith('.pt') and 'sample' not in f
            ]
        )
        input_files = [
            os.path.join(index_dir, f'{embs_name}{i}.pt') for i in range(num_parts)
        ]
        if self.opt['indexer_type'] == 'compressed':
            index_name = self.opt['compressed_indexer_factory'].replace(',', '__')
        elif self.opt['embeddings_name']:
            index_name = self.opt['embeddings_name']
        else:
            index_name = 'hnsw_flat'
        index_path = os.path.join(index_dir, index_name)

        if self.opt['save_index_dir']:
            index_path, index_name = os.path.split(index_path)
            index_path = os.path.join(self.opt['save_index_dir'], index_name)
            if not os.path.exists(self.opt['save_index_dir']):
                logging.info(f'Creating directory for file {index_path}')
                os.makedirs(self.opt['save_index_dir'])

        logging.info(f'index path: {index_path}')
        self.index_path = index_path

        self.index = indexer_factory(self.opt)
        if self.opt['indexer_type'] != 'exact':
            self.train_then_add(input_files)
        else:
            self.index_data(input_files)
        # save data
        self.index.serialize(index_path)

    def index_data(self, input_files: List[str], add_only: bool = False):
        """
        Index data.

        :param input_files:
            files to load.
        """
        all_docs = []
        for in_file in input_files:
            logging.info(f'Reading file {in_file}')
            docs = torch.load(in_file)
            if isinstance(docs, list):
                all_docs += docs
            else:
                all_docs.append(docs)

        self.index.index_data(all_docs)

    def train_then_add(self, input_files: List[str]):
        """
        First train data, then add it.

        If we're only training... then don't add!!
        """
        assert isinstance(self.index, CompressedIndexer)
        # Train
        random.seed(42)

        tensors = []
        for s in input_files:
            logging.info(f'Loading in file {s}')
            tensor = torch.load(s)
            if self.opt['partition_index']:
                self.index.train([tensor])
                self.index.add([tensor])
            else:
                tensors.append(tensor)

        if not self.opt['partition_index']:
            self.index.train([torch.cat(tensors)])
            self.index.add([torch.cat(tensors)])


if __name__ == "__main__":
    Indexer.main()
