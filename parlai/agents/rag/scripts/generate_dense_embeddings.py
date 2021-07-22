#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate Dense Embeddings for use in a FAISS Index.

Modified/Adapted from https://github.com/facebookresearch/DPR/blob/master/generate_dense_embeddings.py.

The input file must be a .tsv file with three elements per row: 1) id; 2) text; 3) title, in that order.

The output is saved in two parts:

1) outfile_<shard_idx>.pt -> the tensor of embeddings
2) ids_<shard_idx> -> The corresponding document IDs encoded in this shard.
"""
import os
import torch
import torch.cuda
from typing import List, Tuple

from parlai.agents.transformer.transformer import TransformerRankerAgent
from parlai.agents.transformer.polyencoder import PolyencoderAgent
from parlai.core.agents import create_agent_from_model_file, create_agent
from parlai.core.build_data import modelzoo_path
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from parlai.core.torch_ranker_agent import TorchRankerAgent
import parlai.utils.logging as logging

from parlai.agents.transformer.dropout_poly import DropoutPolyAgent
from parlai.agents.rag.retrievers import load_passages_list

# import the below to have the agent registered.
import parlai.agents.rag.dpr  # noqa: F401


class Generator(ParlaiScript):
    """
    Generate Dense Embeddings.
    """

    @classmethod
    def setup_args(cls):
        """
        File in/out args, and sharding args.
        """
        parser = ParlaiParser(True, True, 'Generate Dense Embs')
        parser.add_argument(
            '--passages-file',
            type=str,
            help='file containing passages to encode. file should be a tsv file.',
        )
        parser.add_argument(
            '--outfile', type=str, help='where to save the passage embeddings'
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
        parser.add_argument(
            '--dpr-model',
            type='bool',
            default=False,
            help='Specify True to indicate that the provided model file is a DPR Model',
        )
        return parser

    def run(self):
        """
        1) load model 2) generate embeddings 3) save embeddings.
        """
        self.use_cuda = not self.opt.get('no_cuda') and torch.cuda.is_available()
        overrides = {'interactive_mode': True, 'interactive_candidates': 'inline'}
        if self.opt['dpr_model']:
            overrides.update(
                {
                    'model': 'dpr_agent',
                    'model_file': self.opt['model_file'],
                    'share_encoders': False,
                    'override': {
                        'model': 'dpr_agent',
                        'interactive_candidates': 'inline',
                        'share_encoders': False,
                    },
                }
            )
            agent = create_agent(Opt(overrides))
        else:
            agent = create_agent_from_model_file(self.opt['model_file'], overrides)
        model = agent.model.module if hasattr(agent.model, 'module') else agent.model
        assert hasattr(model, 'encoder_cand') or hasattr(model, 'cand_encoder')
        assert isinstance(agent, TorchRankerAgent)
        passages = self.load_passages()
        data = self.encode_passages(agent, passages)
        self.save_data(data)

    def encode_passages(
        self, agent: TorchRankerAgent, passages: List[Tuple[str, str, str]]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Encode passages with model, using candidate encoder.

        :param agent:
            parlai agent
        :param passages:
            passages to encode

        :return encodings:
            return passage encodings
        """
        agent.model.eval()
        n = len(passages)
        bsz = self.opt['batchsize']
        total_enc = 0
        results: List[torch.Tensor] = []
        results_ids: List[str] = []

        for batch_start in range(0, n, bsz):
            batch_passages = passages[batch_start : batch_start + bsz]
            batch_msgs = [
                Message(
                    {
                        "text_vec": agent._check_truncate(
                            agent.dict.txt2vec(f'{title} {text}'), agent.text_truncate
                        )
                    }
                )
                for _, text, title in batch_passages
            ]
            # we call batchify here rather than _pad_tensor directly.
            batch = agent.batchify(batch_msgs)
            if self.use_cuda:
                batch = batch.to('cuda')
            with torch.no_grad():
                if isinstance(agent, TransformerRankerAgent):
                    _, encoding = agent.model(None, None, batch.text_vec)
                else:
                    assert isinstance(agent, DropoutPolyAgent) or isinstance(
                        agent, PolyencoderAgent
                    )
                    _, _, encoding = agent.model(
                        cand_tokens=batch.text_vec.unsqueeze(1)
                    )
                    encoding = encoding.squeeze(1)

            ids = [r[0] for r in batch_passages]
            assert len(ids) == encoding.size(0)
            results.append(encoding.cpu())
            results_ids += ids

            total_enc += len(ids)
            if total_enc % (10 * bsz) == 0:
                logging.info(f'Encoded {total_enc} out of {n} passages')

        return torch.cat(results).cpu(), results_ids

    def load_passages(self) -> List[Tuple[str, str, str]]:
        """
        Load passages from tsv file.

        Limit passages returned according to shard number.

        :return passages:
            return a list of (doc_id, doc_text, doc_title) tuples
        """
        logging.info(f"Loading {self.opt['passages_file']}")
        rows = load_passages_list(
            modelzoo_path(
                self.opt['datapath'], self.opt['passages_file']
            )  # type: ignore
        )
        shard_id, num_shards = self.opt['shard_id'], self.opt['num_shards']
        shard_size = int(len(rows) / num_shards)
        if shard_id < len(rows) % num_shards:
            # don't forget the remainder!
            shard_size += 1
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size
        logging.info(
            f'Shard {shard_id} of {num_shards} encoding psg index '
            f'{start_idx} to {end_idx}, out of {len(rows)}'
        )
        return rows[start_idx:end_idx]

    def save_data(self, data: Tuple[torch.Tensor, List[str]]):
        """
        Save data.

        :param data:
            encoded passages, and corresponding ids
        """
        encoding, ids = data
        assert len(ids) == encoding.size(0)
        embs_outfile = f"{self.opt['outfile']}_{self.opt['shard_id']}.pt"
        logging.info(f'Writing results to {embs_outfile}')
        torch.save(encoding, embs_outfile)
        outdir = os.path.split(self.opt['outfile'])[0]
        ids_outfile = os.path.join(outdir, f"ids_{self.opt['shard_id']}")
        logging.info(f'Writing ids to {ids_outfile}')
        torch.save(ids, ids_outfile)


if __name__ == "__main__":
    Generator.main()
