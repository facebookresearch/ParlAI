#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.jit
import torch.nn as nn
from packaging import version

from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.io import PathManager


def export_model(opt: Opt):
    """
    Export a model to TorchScript so that inference can be run outside of ParlAI.

    Currently, only CPU greedy-search inference on BART models is supported.
    """

    if version.parse(torch.__version__) < version.parse('1.7.0'):
        raise NotImplementedError(
            'TorchScript export is only supported for Torch 1.7 and higher!'
        )
    else:
        # Only load TorchScriptGreedySearch now, because this will trigger scripting of
        # associated modules
        from parlai.torchscript.modules import TorchScriptGreedySearch

    overrides = {
        'no_cuda': True,  # TorchScripting is CPU only
        'model_parallel': False,  # model_parallel is not currently supported when TorchScripting
    }
    if 'override' not in opt:
        opt['override'] = {}
    for k, v in overrides.items():
        opt[k] = v
        opt['override'][k] = v

    print('\nLoading original model.')
    agent = create_agent(opt, requireModelExists=True)

    print('\nCreating greedy-search module for the original unscripted model.')
    original_module = TorchScriptGreedySearch(
        agent=agent,
        embedding_weights=agent.model.embeddings.weight,
        encoder=agent.model.encoder,
    )

    print('\nDeleting extra copies of the token embedding layer, which won\'t be used.')
    del agent.model.encoder.embeddings
    del agent.model.decoder.embeddings

    class DummyEmbeddingWeightsModule(nn.Module):
        def __init__(self, embedding_weights: torch.Tensor):
            self.weights = embedding_weights

        def forward(self, tensor: torch.Tensor):
            return torch.index_select(self.weights, dim=0, index=input[0]).unsqueeze(0)

    print('\nCreating a dummy module containing the embedding weights.')
    dummy_module = DummyEmbeddingWeightsModule(
        embedding_weights=agent.model.embeddings.weight
    )

    # Optionally quantize the model
    if opt['quantize']:

        print('\nPerforming dynamic quantization of linear layers.')
        agent.model = torch.quantization.quantize_dynamic(
            model=agent.model,
            qconfig_spec={
                torch.nn.Linear: torch.quantization.per_channel_dynamic_qconfig
            },
            dtype=torch.qint8,
            inplace=False,
        )

        print('\nPerforming quantization of embeddings.')
        dummy_module = torch.quantization.quantize_dynamic(
            model=dummy_module,
            qconfig_spec={
                torch.nn.Linear: torch.quantization.per_channel_dynamic_qconfig
            },
            dtype=torch.qint8,
            inplace=False,
        )

        print('\nCreating module for quantized model.')
        quantized_module = TorchScriptGreedySearch(
            agent=agent,
            embedding_weights=dummy_module.weights,
            encoder=agent.model.encoder,
        )

    else:

        quantized_module = None

    print(
        '\nDeleting extra objects in the model so that they don\'t get duplicated when tracing.'
    )
    # It's not as important to delete the decoder because that typically has fewer layers, and the model's `reorder_decoder_incremental_state()` relies on it so it's harder to remove
    encoder = agent.model.encoder
    del agent.model.embeddings
    del agent.model.encoder

    print('\nScripting the module.')
    scripted_module = torch.jit.script(
        TorchScriptGreedySearch(
            agent=agent, embedding_weights=dummy_module.weights, encoder=encoder
        )
    )

    print('\nSaving the scripted module.')
    with PathManager.open(opt['scripted_model_file'], 'wb') as f:
        torch.jit.save(scripted_module, f)

    # Compare the original module to the scripted module against the test inputs
    if len(opt['input']) > 0:
        inputs = opt['input'].split('|')
        print('\nGenerating given the original unscripted module:')
        _run_conversation(module=original_module, inputs=inputs)
        if quantized_module is not None:
            print('\nGenerating given the quantized module:')
            _run_conversation(module=quantized_module, inputs=inputs)
        print('\nGenerating given the scripted module:')
        _run_conversation(module=scripted_module, inputs=inputs)


def setup_args() -> ParlaiParser:
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.add_argument(
        '-smf',
        '--scripted-model-file',
        type=str,
        default='_scripted.pt',
        help='Where the scripted model checkpoint will be saved',
    )
    parser.add_argument(
        "--quantize",
        type='bool',
        default=False,
        help="Run dynamic quantization (int8) on the model before TorchScripting",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default='',
        help="Input string to pass into the encoder of the scripted model, to test it against the unscripted version. Separate lines with a pipe",
    )
    return parser


def _run_conversation(module: nn.Module, inputs: List[str]):
    """
    Run a conversation with the given module given the input strings.
    """
    context = []
    for input_ in inputs:
        print(' TEXT: ' + input_)
        context.append(input_)
        label = module('\n'.join(context))
        print("LABEL: " + label)
        context.append(label)


@register_script('torchscript', hidden=True)
class TorchScript(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return export_model(self.opt)


if __name__ == '__main__':
    TorchScript.main()
