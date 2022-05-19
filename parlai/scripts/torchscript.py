#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from typing import List

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
    """

    if version.parse(torch.__version__) < version.parse("1.7.0"):
        raise NotImplementedError(
            "TorchScript export is only supported for Torch 1.7 and higher!"
        )
    else:
        # Only load TorchScriptGreedySearch now, because this will trigger scripting of
        # associated modules
        from parlai.torchscript.modules import TorchScriptGreedySearch

    overrides = {
        "model_parallel": False  # model_parallel is not currently supported when TorchScripting,
    }
    if opt.get("script_module"):
        script_module_name, script_class_name = opt["script_module"].split(":", 1)
        script_module = importlib.import_module(script_module_name)
        script_class = getattr(script_module, script_class_name)
    else:
        script_class = TorchScriptGreedySearch
    if "override" not in opt:
        opt["override"] = {}
    for k, v in overrides.items():
        opt[k] = v
        opt["override"][k] = v

    # Create the unscripted greedy-search module
    agent = create_agent(opt, requireModelExists=True)
    original_module = script_class(agent)

    # Script the module and save
    instantiated = script_class(agent)
    if not opt["no_cuda"]:
        instantiated = instantiated.cuda()
    if opt.get("enable_inference_optimizations"):
        scripted_model = torch.jit.optimize_for_inference(
            torch.jit.script(instantiated.eval())
        )
    else:
        scripted_model = torch.jit.script(instantiated)

    with PathManager.open(opt["scripted_model_file"], "wb") as f:
        torch.jit.save(scripted_model, f)

    # Compare the original module to the scripted module against the test inputs
    if len(opt["input"]) > 0:
        inputs = opt["input"].split("|")
        print("\nGenerating given the original unscripted module:")
        _run_conversation(module=original_module, inputs=inputs)
        print("\nGenerating given the scripted module:")
        _run_conversation(module=scripted_model, inputs=inputs)


def setup_args() -> ParlaiParser:
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.add_argument(
        "-smf",
        "--scripted-model-file",
        type=str,
        default="_scripted.pt",
        help="Where the scripted model checkpoint will be saved",
    )
    parser.add_argument(
        "-in",
        "--input",
        type=str,
        default="",
        help="Input string to pass into the encoder of the scripted model, to test it against the unscripted version. Separate lines with a pipe",
    )
    parser.add_argument(
        "-sm",
        "--script-module",
        type=str,
        default="parlai.torchscript.modules:TorchScriptGreedySearch",
        help="module to TorchScript. Example: parlai.torchscript.modules:TorchScriptGreedySearch",
    )
    parser.add_argument(
        "-eio",
        "--enable-inference-optimization",
        type=bool,
        default=False,
        help="Enable inference optimizations on the scripted model.",
    )
    return parser


def _run_conversation(module: nn.Module, inputs: List[str]):
    """
    Run a conversation with the given module given the input strings.
    """
    context = []
    for input_ in inputs:
        print(" TEXT: " + input_)
        context.append(input_)
        label = module("\n".join(context))
        print("LABEL: " + label)
        context.append(label)


@register_script("torchscript", hidden=True)
class TorchScript(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return export_model(self.opt)


if __name__ == "__main__":
    TorchScript.main()
