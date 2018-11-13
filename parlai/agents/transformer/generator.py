# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from .modules import TransformerGeneratorModel
from .transformer import add_common_cmdline_args


class GeneratorAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(cls, GeneratorAgent).add_cmdline_args(argparser)
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
