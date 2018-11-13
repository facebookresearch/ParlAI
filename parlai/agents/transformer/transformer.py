# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.utils import padded_3d
from .modules import TransformerMemNetModel, TransformerGeneratorModel


class TransformerAgent(TorchRankerAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Transformer Arguments')
        # memory and knowledge arguments
        agent.add_argument('--wrap-memory-encoder', type='bool',
                           default=False,
                           help='wrap memory encoder with MLP')
        agent.add_argument('--memory-attention', type=str, default='sqrt',
                           choices=['cosine', 'dot', 'sqrt'],
                           help='similarity for basic attention mechanism'
                                'when using transformer to encode memories')

        # model specific arguments
        agent.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
        agent.add_argument('--n-layers', type=int, default=2)
        agent.add_argument('--normalize-sent-emb', type='bool', default=False)
        agent.add_argument('--share-encoders', type='bool', default=False)
        agent.add_argument('--ffn-size', type=int, default=300,
                           help='Hidden size of the FFN layers')
        agent.add_argument('--attention-dropout', type=float, default=0.0)
        agent.add_argument('--relu-dropout', type=float, default=0.0)
        agent.add_argument('--n-heads', type=int, default=3)
        agent.add_argument('--learn-positional-embeddings', type='bool', default=False)
        agent.add_argument('--has-memories', type='bool', default=False,
                           help='If true, text contains newline separated memories '
                                'before the actual text')
        agent.add_argument('--use-memories', type='bool', default=False,
                           help='If true, use the memories to help with predictions')
        agent.add_argument('--embeddings-scale', type='bool', default=True)
        agent.add_argument('--scores-norm', choices={'dot', 'sqrt', 'dim'})

        TransformerAgent.dictionary_class().add_cmdline_args(argparser)

        super(cls, TransformerAgent).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        # check all the options
        if opt.get('num_threads', 1) != 1:
            raise ValueError('Transformer does not work with hogwild.')

        super().__init__(opt, shared)

    def build_model(self, states=None):
        self.model = TransformerMemNetModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embeddings.weight, self.opt['embedding_type']
            )
        return self.model

    def vectorize(self, *args, **kwargs):
        """Override options in vectorize from parent."""
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        kwargs['split_lines'] = self.opt['has_memories']
        return super().vectorize(*args, **kwargs)

    def get_dialog_history(self, *args, **kwargs):
        """Override options in get_dialog_history from parent."""
        kwargs['add_p1_after_newln'] = True  # will only happen if -pt True
        return super().get_dialog_history(*args, **kwargs)

    def score_candidates(self, batch, cand_vecs):
        # convoluted check that not all memories are empty
        if (self.opt['use_memories'] and batch.memory_vecs is not None and
                sum(len(m) for m in batch.memory_vecs)):
            mems = padded_3d(batch.memory_vecs, use_cuda=self.use_cuda)
        else:
            mems = None

        return self.model(
            xs=batch.text_vec,
            mems=mems,
            cands=cand_vecs,
        )


class GenerativeTransformerAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Transformer Arguments')
        # model specific arguments
        agent.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
        agent.add_argument('--n-layers', type=int, default=2)
        agent.add_argument('--normalize-sent-emb', type='bool', default=False)
        agent.add_argument('--ffn-size', type=int, default=300,
                           help='Hidden size of the FFN layers')
        agent.add_argument('--attention-dropout', type=float, default=0.0)
        agent.add_argument('--relu-dropout', type=float, default=0.0)
        agent.add_argument('--n-heads', type=int, default=3)
        agent.add_argument('--learn-positional-embeddings', type='bool', default=False)
        agent.add_argument('--embeddings-scale', type='bool', default=True)

        TransformerAgent.dictionary_class().add_cmdline_args(argparser)

        super(cls, GenerativeTransformerAgent).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        # check all the options
        if opt.get('num_threads', 1) != 1:
            raise ValueError('Transformer does not work with hogwild.')

        super().__init__(opt, shared)

    def build_model(self, states=None):
        self.model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model.cuda()
        return self.model
