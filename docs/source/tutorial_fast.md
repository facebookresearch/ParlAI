# Speeding up training

Author: Stephen Roller

This tutorial walks you through a few ways to massively speed up your training runs
in ParlAI. These tricks tend to work best with generative models, but some can also be
used with others.

A summary of the speedups is in this table:

| Method                  | Train | Eval | Total | Speedup |
| ----------------------- | ----: | ---: | ----: | ------: |
| Baseline                |  504s |  48s |  552s |    1.0x |
| Skip generation         |  504s |  16s |  520s |    1.1x |
| Dynamic batching        |  254s |  11s |  265s |    2.1x |
| FP16                    |  197s |   8s |  205s |    2.7x |
| Larger batchsize (FP16) |  151s |   7s |  158s |    3.5x |
| Using 4 GPUs            |   47s |   3s |   50s |   11.0x |

## Setting a baseline

We'll start with an example training command, which trains a
transformer/generator on ConvAI2 for one epoch, using a batchsize of 64, with a
roughly 20M parameter model. We'll train using ADAM optimizer, with a learning
rate of 1e-3. We'll build the dictionary ahead of time to ensure it's kept the
same.

    mkdir fastmodels
    parlai build_dict -t convai2 -df dictfile
    parlai train -df dictfile -m transformer/generator -t convai2 -eps 1.0 -bs 64 \
        --embedding-size 250 --ffn-size 1000 --n-layers 8 -opt adam -lr 1e-3

On my computer using a 16gb V100, this takes about 550s. 500s is spend in training,
and another 50s is spent in evaluation.

We will continuously modify this training command in this tutorial, but you are
free to mix and match options.

## Skip generation

You may notice your model is taking a long time to evaluate, even though the
evaluation dataset is much smaller. This is because we are doing full
generation through the model, including beam search. You can get a massive
speedup by turning off this generation step, with `--skip-generation true`.

    parlai train -df dictfile -m transformer/generator -t convai2 -eps 1.0 -bs 64 \
        --embedding-size 250 --ffn-size 1000 --n-layers 8 -opt adam -lr 1e-3 \
        --skip-generation true

This brings evaluation time down to 16s, but doesn't affect training time.  Just
remember you will need to _turn `--skip-generation` back off_ if you want
statistics like BLEU or F1. Also, `--skip-generation` is only an option in
generative models. Ranking models have similar options like `-cands batch`.

:::{warning} Persistence
Models trained with `--skip-generation True` will remember this option when
loading back up. You will need to manually set it back to false whenever you
need evaluations with generations.
:::

## Dynamic batching

Dynamic batching groups conversations of the same length at the same time, to
minimize the amount of unnecessary padding in the tensors. Furthermore, dynamic
batching actually _increases_ the batch size to use the maximum amount of
memory available on your GPU.

Add `--dynamic-batching full` (`-dynb full`) to your training command. Note
that in order to use dynamic batching, we must also set a `--truncate` option.
We'll use 256, since that is longer than almost all the conversations in our
data.

    parlai train -df dictfile -m transformer/generator -t convai2 -eps 1.0 -bs 64 \
        --embedding-size 250 --ffn-size 1000 --n-layers 8 -opt adam -lr 1e-3 \
        --skip-generation true \
        --eval-batchsize 128 \
        --dynamic-batching full --truncate 256

You should notice that your memory utilization is much higher in this mode.
This actually has an advantage, since you can more easily find your maximum
batchsize.  To get the full benefits of dynamic batching, make sure to use the
largest batchsize you can.

Overall, this results in a large increase in speed: about 2x, bringing training
down to 250s and evaluation to 11s.

:::{warning} WARNING
You may find perplexity is quite a bit worse than without dynamic
batching. This is because we use larger batches, and take fewer steps.  You can
usually increase your learning rate pretty substantially when using dynamic
batching, to compensate for the fewer steps.
:::

## FP16

If you have access to an NVIDIA GPU with FP16 CUDA Cores (V100, GTX 2080, etc),
then you can get large speedups by switching on the option `--fp16 true`. The
default version of FP16 requires that you install
[APEX](https://github.com/NVIDIA/apex), but you can use a simplified version
(which doesn't depend on APEX) with `--fp16 true --fp16-impl mem_efficient`.

Note that in order to get the full benefit of fp16, we need to make sure all
our hidden dimensions are _multiples of 8_, otherwise the hardware won't use
CUDA cores. We will slightly adjust the size of the network to support this.
We'll slightly adjust the network parameters (--embedding-size and --ffn-size)
to conform to this.

    parlai train -df dictfile -m transformer/generator -t convai2 -eps 1.0 -bs 64 \
        --embedding-size 256 --ffn-size 1024 --n-layers 8 -opt adam -lr 1e-3 \
        --skip-generation true \
        --eval-batchsize 128 \
        --dynamic-batching full \
        --fp16 true --fp16-impl mem_efficient

Further notice that FP16 often significantly lowers the memory size of your model
and activations (almost by a factor of 2). This means you can usually get away with
significantly increasing the batchsize (and eval batchsize).

    parlai train -df dictfile -m transformer/generator -t convai2 -eps 1.0 \
        --embedding-size 256 --ffn-size 1024 --n-layers 8 -opt adam -lr 1e-3 \
        --skip-generation true \
        --eval-batchsize 256 \
        --dynamic-batching full \
        --fp16 true -bs 128

In this example, we see about a 25% speedup. Generally you can expect a larger
speedup with larger models, with models of >300M often getting a ~50% speedup.
With the increased batch size, this can often be brought to 2.5x faster.

:::{warning} FP16 requires modern GPUs
Without a GPU with FP16 CUDA cores, you may find that FP16 actually slows
your program. You may still see a benefit from the reduced memory usage though.
:::

## Use multiple GPUs

If you have multiple GPUs, you can utilize them by switching from `train` to
`multiprocessing_train`. If you have 4 GPUs, you'll find training should be
roughly 3.5x faster. The arguments for the training are left otherwise the same.

    parlai multiprocessing_train \
        -df dictfile -m transformer/generator -t convai2 -eps 1.0 -bs 64 \
        --embedding-size 256 --ffn-size 1024 --n-layers 8 -opt adam -lr 1e-3 \
        --skip-generation true \
        --eval-batchsize 128 \
        --dynamic-batching full \
        --fp16 true

Note that we leave batchsize the same: we use the batchsize PER GPU. In my
system, I have 4 GPUs, so things are a little under 4x faster.

Similarly, we also have the `multiprocessing_eval` command, for using multiple
GPUs in evaluation.

:::{danger}
This should never be mixed with options like `--model-parallel true`
or `--data-parallel true`, as those options use different GPUs without
multiprocessing.  The BlenderBot3B and BlenderBot9B models both use those
options, so this should be used with care.
:::
