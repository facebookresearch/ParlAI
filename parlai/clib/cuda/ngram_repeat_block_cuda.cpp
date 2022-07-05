/*
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Microsoft Corporation.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
Code adapted from https://github.com/microsoft/fastseq/blob/main/fastseq/clib/cuda/ngram_repeat_block_cuda.cpp.
*/

#include <torch/extension.h>
#include <vector>

/*
CPP Binding for CUDA OP
*/

// CUDA forward declarations
torch::Tensor ngram_repeat_block_cuda_forward(const torch::Tensor hypothesis, const torch::Tensor context,
                                              torch::Tensor lprobs, int bsz,
                                              int step, int beam_size,
                                              int no_repeat_ngram_size,
                                              bool if_context_blocking);

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// Input check and call to CUDA OP
// Backward method not required
torch::Tensor ngram_repeat_block_forward(const torch::Tensor hypothesis, const torch::Tensor context,
                                         torch::Tensor lprobs, int bsz,
                                         int step, int beam_size,
                                         int no_repeat_ngram_size,
                                         bool if_context_blocking)
{
  CHECK_INPUT(hypothesis);
  CHECK_INPUT(lprobs);
  assert(bsz > 0);
  assert(step >= 0);
  assert(beam_size > 0);
  assert(no_repeat_ngram_size > 0);

  return ngram_repeat_block_cuda_forward(hypothesis, context, lprobs, bsz, step, beam_size,
                                         no_repeat_ngram_size, if_context_blocking);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &ngram_repeat_block_forward,
        "No Repeat Ngram Block forward (CUDA)");
}
