/*
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Microsoft Corporation.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
Code adapted from https://github.com/microsoft/fastseq/blob/main/fastseq/clib/cuda/ngram_repeat_block_cuda_kernel.cu.
*/

/*
Kernel implementation for blocking repeated n-grams.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>
#include <vector>

// Ban repeated ngrams of length = 'no_repeat_ngram_size'
__global__ void banRepeatedTokens(long *__restrict__ hypothesis_ptr,
                                  long *__restrict__ context_ptr,
                                  float *__restrict__ lprobs,
                                  int current_seq_length,
                                  int vocab_size,
                                  int no_repeat_ngram_size,
                                  bool if_context_blocking)
{
  auto row = blockIdx.x;
  auto col = threadIdx.x;
  // start of context ngram on current thread
  auto index = row * current_seq_length + col;
  // start of last ngram of hypothesis
  auto start_of_ngram = current_seq_length - no_repeat_ngram_size + 1;
  long *__restrict__ previous_ngram_ptr;

  if (if_context_blocking)
  {
    previous_ngram_ptr = &context_ptr[col];
  }
  else
  {
    previous_ngram_ptr = &hypothesis_ptr[index];
  }

  auto lprob_start = row * vocab_size;
  extern __shared__ long tokens_shm[];
  // each thread writes to shared array
  tokens_shm[col] = *previous_ngram_ptr;

  // final thread writes the end of previous ngram array to tokens_shm
  if (col == blockDim.x - 1)
  {
    for (int i = 1; i < no_repeat_ngram_size; i++)
    {
      tokens_shm[col + i] = previous_ngram_ptr[i];
    }
  }
  __syncthreads();

  // Each thread compares ngram starting from
  // thread index with final ngram starting
  for (int k = 0; k < no_repeat_ngram_size - 1; k++)
  {
    if (tokens_shm[col + k] != hypothesis_ptr[row * current_seq_length + start_of_ngram + k])
    {
      return;
    }
  }

  // reach here means ban
  auto token_to_be_banned = tokens_shm[col + no_repeat_ngram_size - 1];
  lprobs[lprob_start + token_to_be_banned] = -INFINITY;
}

// Allocate blocks and threads based on
// batch size and sequence length and launch
// kernel
torch::Tensor ngram_repeat_block_cuda_forward(const torch::Tensor hypothesis,
                                              const torch::Tensor context,
                                              torch::Tensor lprobs,
                                              int bsz,
                                              int step,
                                              int beam_size,
                                              int no_repeat_ngram_size,
                                              bool if_context_blocking)
{

  auto hypothesis_ptr = hypothesis.data_ptr<long>();
  auto context_ptr = context.data_ptr<long>();
  auto lprob_ptr = lprobs.data_ptr<float>();

  int context_length;
  if (if_context_blocking)
  {
    context_length = context.size(0);
  }
  else
  {
    // context is previously generated word sequences for self-blocking
    context_length = hypothesis.size(1);
  }

  int threads = context_length - no_repeat_ngram_size + 1;
  if (step - no_repeat_ngram_size + 2 <= 0)
    return lprobs;
  int vocab_size = lprobs.size(1);
  int blocks = bsz * beam_size;
  int current_seq_length = hypothesis.size(1);
  int shared_mem_size = context_length * sizeof(long);

  // Launching N blocks where N is number of samples in a batch (beams*bsz)
  // Launching T threads where T is number of previous ngrams in a sample
  // Allocating shared mem per block for fastser access of input tokens since
  // each token will be accessed N times to compare with current Ngram where
  // N is Ngram size.

  banRepeatedTokens<<<blocks, threads, shared_mem_size>>>(
      hypothesis_ptr,
      context_ptr,
      lprob_ptr,
      current_seq_length,
      vocab_size,
      no_repeat_ngram_size,
      if_context_blocking);
  return lprobs;
}
