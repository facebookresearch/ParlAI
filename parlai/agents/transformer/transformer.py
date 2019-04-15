# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.utils import warn_once
from parlai.core.utils import padded_3d
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent

from .modules import TransformerMemNetModel
from .modules import TransformerGeneratorModel

import torch


warn_once(
    "Public release transformer models are currently in beta. The name of "
    "command line options may change or disappear before a stable release. We "
    "welcome your feedback. Please file feedback as issues at "
    "https://github.com/facebookresearch/ParlAI/issues/new"
)


def add_common_cmdline_args(argparser):
    argparser.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument('-hid', '--ffn-size', type=int, default=300,
                           help='Hidden size of the FFN layers')
    argparser.add_argument('--dropout', type=float, default=0.0,
                           help='Dropout used in Vaswani 2017.')
    argparser.add_argument('--attention-dropout', type=float, default=0.0,
                           help='Dropout used after attention softmax.')
    argparser.add_argument('--relu-dropout', type=float, default=0.0,
                           help='Dropout used after ReLU. From tensor2tensor.')
    argparser.add_argument('--n-heads', type=int, default=2,
                           help='Number of multihead attention heads')
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument('--n-positions', type=int, default=None, hidden=True,
                           help='Number of positional embeddings to learn. Defaults '
                                'to truncate or 1024 if not provided.')
    argparser.add_argument('--n-segments', type=int, default=0,
                           help='The number of segments that support the model. '
                                'If zero no segment and no langs_embedding.')
    argparser.add_argument('--variant', choices={'aiayn', 'xlm'}, default='aiayn',
                           help='Chooses locations of layer norms, etc.')
    argparser.add_argument('--activation', choices={'relu', 'gelu'}, default='relu',
                           help='Nonlinear activation to use. AIAYN uses relu, but '
                                'more recent papers prefer gelu.')


class Transformer(Agent):
    """
    Placeholder class, which just throws an error telling the user to specify
    whether they want the ranker or the generator.
    """
    def __init__(self, opt, shared=None):
        raise RuntimeError(
            "`--model transformer` is not a valid choice. Please select either "
            "`--model transformer/ranker` or `--model transformer/generator"
        )


class TransformerRankerAgent(TorchRankerAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(TransformerRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        # memory and knowledge arguments
        agent.add_argument('--use-memories', type='bool', default=False,
                           help='use memories: must implement the function '
                                '`_vectorize_memories` to use this')
        agent.add_argument('--wrap-memory-encoder', type='bool',
                           default=False,
                           help='wrap memory encoder with MLP')
        agent.add_argument('--memory-attention', type=str, default='sqrt',
                           choices=['cosine', 'dot', 'sqrt'],
                           help='similarity for basic attention mechanism'
                                'when using transformer to encode memories')
        # model specific arguments
        agent.add_argument('--normalize-sent-emb', type='bool', default=False)
        agent.add_argument('--share-encoders', type='bool', default=True)
        agent.add_argument('--learn-embeddings', type='bool', default=True,
                           help='learn embeddings')
        agent.add_argument('--data-parallel', type='bool', default=False,
                           help='use model in data parallel, requires '
                                'multiple gpus')
        agent.add_argument('--reduction-type', type=str, default='mean',
                           choices=['first', 'max', 'mean'],
                           help='Type of reduction at the end of transformer')

        argparser.set_defaults(
            learningrate=0.0001,
            optimizer='adamax',
            truncate=1024,
        )
        cls.dictionary_class().add_cmdline_args(argparser)

        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            from parlai.core.distributed_utils import is_distributed
            if is_distributed():
                raise ValueError(
                    'Cannot combine --data-parallel and distributed mode'
                )
            self.model = torch.nn.DataParallel(self.model)

    def _score(self, output, cands):
        if cands.dim() == 2:
            return torch.matmul(output, cands.t())
        elif cands.dim() == 3:
            return torch.bmm(output.unsqueeze(1),
                             cands.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError('Unexpected candidate dimensions {}'
                               ''.format(cands.dim()))

    def build_model(self, states=None):
        self.model = TransformerMemNetModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embeddings.weight, self.opt['embedding_type']
            )
        return self.model

    def batchify(self, obs_batch, sort=False):
        """Override so that we can add memories to the Batch object."""
        batch = super().batchify(obs_batch, sort)
        if self.opt['use_memories']:
            valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                         self.is_valid(ex)]
            valid_inds, exs = zip(*valid_obs)
            mems = None
            if any('memory_vecs' in ex for ex in exs):
                mems = [ex.get('memory_vecs', None) for ex in exs]
            batch.memory_vecs = mems
        return batch

    def _vectorize_memories(self, obs):
        # TODO: move this to Torch Ranker Agent
        raise NotImplementedError(
            'Abstract class: user must implement this function to use memories'
        )

    def vectorize(self, *args, **kwargs):
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        obs = super().vectorize(*args, **kwargs)
        if self.opt['use_memories']:
            obs = self._vectorize_memories(obs)
        return obs

    def encode_candidates(self, padded_cands):
        _, cands = self.model(
            xs=None,
            mems=None,
            cands=padded_cands,
        )

        return cands

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        # convoluted check that not all memories are empty
        if (self.opt['use_memories'] and batch.memory_vecs is not None and
                sum(len(m) for m in batch.memory_vecs)):
            mems = padded_3d(batch.memory_vecs, use_cuda=self.use_cuda,
                             pad_idx=self.NULL_IDX)
        else:
            mems = None

        if cand_encs is not None:
            # we pre-encoded the candidates, do not re-encode here
            cand_vecs = None

        context_h, cands_h = self.model(
            xs=batch.text_vec,
            mems=mems,
            cands=cand_vecs,
        )

        if cand_encs is not None:
            cands_h = cand_encs

        scores = self._score(context_h, cands_h)

        return scores


class TransformerGeneratorAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        self.model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model.cuda()
        return self.model
