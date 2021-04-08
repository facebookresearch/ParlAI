#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.jit
import torch.nn as nn
from torch.quantization import convert, float_qparams_weight_only_qconfig, prepare
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

    # Create the unscripted greedy-search module
    agent = create_agent(opt, requireModelExists=True)
    original_module = TorchScriptGreedySearch(agent)

    # Optionally quantize the model
    if opt['quantize']:

        print('Performing dynamic quantization of linear layers.')
        model = torch.quantization.quantize_dynamic(
            model=agent.model,
            qconfig_spec={
                torch.nn.Linear: torch.quantization.per_channel_dynamic_qconfig
            },
            dtype=torch.qint8,
            inplace=False,
        )

        print('Performing quantization of embeddings.')
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                module.qconfig = float_qparams_weight_only_qconfig
        prepare(model, inplace=True)
        convert(model, inplace=True)

        agent.model = model

        agent.save(opt['scripted_model_file'] + '._quantized_unscripted')

        quantized_module = TorchScriptGreedySearch(agent)

    else:

        quantized_module = None

    # Script the module and save
    scripted_module = torch.jit.script(maybe_quantized_module)
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
