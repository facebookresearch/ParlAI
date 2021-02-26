#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.jit
import torch.nn.functional as F

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import TorchGeneratorModel


def test_jit():

    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    args_ = f"""\
    --model-file zoo:blender/blender_90M/model \
    --no-cuda \
    """
    opt = parser.parse_args(args_.split())
    agent = create_agent(opt, requireModelExists=True)
    # Using create_agent() instead of create_agent_from_model_file() because I couldn't get
    # --no-cuda to be recognized with the latter
    # get the tokenization
    obs = agent.observe({'text': 'hello world', 'episode_done': True})
    batch = agent.batchify([obs])
    tokens = batch.text_vec

    # Trace all subcomponents of the model separately
    bsz = tokens.size(0)
    encoder_states = agent.model.encoder(tokens)
    generations = agent.model._get_initial_decoder_input(bsz, 1).to(tokens.device)
    # latent, incr_state = agent.model.decoder(generations, encoder_states, incr_state)
    latent = agent.model.decoder(generations, encoder_states)
    traced_encoder = torch.jit.trace(agent.model.encoder, tokens)
    # partially_traced_model = torch.jit.trace_module(
    #     agent.model, {'_get_initial_decoder_input': (bsz, 1)}
    # )
    # traced_decoder = torch.jit.trace(agent.model.decoder, (generations, encoder_states))
    # traced_output = torch.jit.trace(agent.model.output, (latent[:, -1:, :]))

    # Run the original greedy search and trace the result
    scripted_function = jit_greedy_search(
        traced_encoder=traced_encoder,
        # partially_traced_model=partially_traced_model,
        # traced_decoder=traced_decoder,
        # traced_output=traced_output,
        x=tokens,
        end_idx=agent.model.end_idx,
        null_idx=agent.model.NULL_IDX,
    )
    print(scripted_function)

    # # print(result)
    # print(agent._v2t(result[0].tolist()))

    # # Trace and save the module
    # traced_module = torch.jit.trace_module(agent.model, {'jit_greedy_search': tokens})
    # print('Finished tracing.')
    # traced_module.save('_traced_blender_90M.pt')

    # # Run greedy search on the traced module
    # traced_result = traced_module.jit_greedy_search(tokens)
    # print(agent._v2t(traced_result[0].tolist()))


@torch.jit.script
def jit_greedy_search(
    traced_encoder,
    # partially_traced_model,
    # traced_decoder,
    # traced_output,
    x: torch.Tensor,
    end_idx: int,
    null_idx: int,
    max_len: int = 128,
):
    """
    A helper function for exporting simple greedy-search models via
    TorchScript.

    Models with extra inputs will need to override to include more
    variables.

    Utilize with:

    >>> TODO: write this
    """
    # incr_state: Optional[Dict[int, Dict[str, Dict[str, torch.Tensor]]]] = None
    bsz = x.size(0)
    encoder_states = traced_encoder(x)
    # generations = partially_traced_model._get_initial_decoder_input(bsz, 1).to(x.device)
    # # keep track of early stopping if all generations finish
    # seen_end = torch.zeros(x.size(0), device=x.device, dtype=torch.bool)
    # for _ in range(max_len):
    #     # latent, incr_state = self.decoder(generations, encoder_states, incr_state)
    #     latent = traced_decoder(generations, encoder_states)
    #     logits = traced_output(latent[:, -1:, :])
    #     _, preds = logits.max(dim=2)
    #     seen_end = seen_end + (preds == end_idx).squeeze(1)
    #     generations = torch.cat([generations, preds], dim=1)
    #     if torch.all(seen_end):
    #         break
    # padded_generations = F.pad(
    #     generations, pad=(0, max_len - generations.size(1)), value=null_idx
    # )
    # Just pad the generation to max_len so that the generation will be the same
    # size before and after tracing, which is needed when the tracer checks the
    # similarity of the outputs after tracing
    # return padded_generations
    return 0


if __name__ == '__main__':
    test_jit()
