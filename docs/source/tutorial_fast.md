# Speeding up training

Author: Stephen Roller

This tutorial walks you through a few ways to massively speed up your training
runs in ParlAI. These tricks tend to work best with generative models, but some
can also be used with others.

:::{warning} Retuning-hyperparameters
Many of these options are effectively tricks for changing the batchsize.
Note that you will want to freshly tune your learning rate when using
these tricks, or simply use these options in the first place.

This tutorial is only for illustration purposes of how to speed up training,
and may not get you the best _performing_ model. You will need to tune
other hyperparameters like Learning Rate.
:::

A summary of the speedups, for an 8 layer Transformer, is in this table:

| Method                   | Train | Eval | Total | Speedup |
| ------------------------ | ----: | ---: | ----: | ------: |
| Baseline                 |  579s |  60s |  639s |   1.00x |
| Skip generation          |  579s |  20s |  599s |   1.07x |
| Dynamic batching         |  289s |  18s |  307s |   2.08x |
| FP16                     |  212s |  11s |  223s |   2.60x |
| Larger batchsize (FP16)  |  168s |  10s |  178s |   3.59x |
| Background preprocessing |  144s |  10s |  154s |   4.15x |
| Using 4 GPUs             |   63s |   4s |   67s |   8.64x |

## Setting a baseline

We'll start with an example training command, which trains a
transformer/generator on ConvAI2 for 1 epoch, using a batchsize of 64, with a
roughly 20M parameter model. We'll train using ADAM optimizer, with a learning
rate of 1e-3. We'll build the dictionary ahead of time to ensure it's kept the
same. *This will be our baseline*.

    parlai build_dict --task convai2 --dict-file dictfile
    parlai train --dict-file dictfile --model transformer/generator \
        --task convai2 --num-epochs 1.0 --batchsize 64 --n-layers 8  \
        --embedding-size 250 --ffn-size 1000 --optimizer adam --learningrate 1e-3

On my computer using a 16gb V100, this takes about 550s. 500s is spend in training,
and another 50s is spent in evaluation.

We will continuously modify this training command in this tutorial, but you are
free to mix and match options.

## Skip generation & larger eval batchsize

You may notice your model is taking a long time to evaluate, even though the
evaluation dataset is much smaller. This is because we are doing full
generation through the model, including beam search. You can get a massive
speedup by turning off this generation step, with `--skip-generation true`.

Note that during evaluation, we don't need to keep activations around for
backpropagation, so we can also afford to increase the batchsize. About 2x
commonly works, but 1.5x is more conservative.

    parlai train --dict-file dictfile --model transformer/generator \
        --task convai2 --num-epochs 1.0 --batchsize 64 --n-layers 8  \
        --embedding-size 250 --ffn-size 1000 --optimizer adam --learningrate 1e-3 \
        --skip-generation true --eval-batchsize 128

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
data. Note you can use the `clen` metric to help you set a good truncation.

    parlai train --dict-file dictfile --model transformer/generator \
        --task convai2 --num-epochs 1.0 --batchsize 64 --n-layers 8  \
        --embedding-size 250 --ffn-size 1000 --optimizer adam --learningrate 1e-3 \
        --skip-generation true --eval-batchsize 128 \
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
then you can get large speedups by switching on the option `--fp16 true`. There
are two version of FP16 you may use: `--fp16-impl safe` and
`--fp16-impl mem_efficient`. The latter trades some numerical stability in exhange
for lower memory consumption, and typically works well in practice.

Note that in order to get the full benefit of fp16, we need to make sure all
our hidden dimensions are _multiples of 8_, otherwise the hardware won't use
CUDA cores. We will slightly adjust the size of the network to support this.
We'll slightly adjust the network parameters (--embedding-size and --ffn-size)
to conform to this.

    parlai train --dict-file dictfile --model transformer/generator \
        --task convai2 --num-epochs 1.0 --batchsize 64 --n-layers 8  \
        --embedding-size 256 --ffn-size 1024 --optimizer adam --learningrate 1e-3 \
        --skip-generation true --eval-batchsize 128 \
        --dynamic-batching full --truncate 256
        --fp16 true --fp16-impl mem_efficient

Further notice that FP16 often significantly lowers the memory size of your model
and activations (almost by a factor of 2). This means you can usually get away with
significantly increasing the batchsize (and eval batchsize).

    parlai train --dict-file dictfile --model transformer/generator \
        --task convai2 --num-epochs 1.0 --batchsize 64 --n-layers 8  \
        --embedding-size 256 --ffn-size 1024 --optimizer adam --learningrate 1e-3 \
        --skip-generation true \
        --dynamic-batching full \
        --fp16 true --fp16-impl mem_efficient --batchsize 128 --eval-batchsize 256

In this example, we see about a 25% speedup. Generally you can expect a larger
speedup with larger models, with models of >300M often getting a ~50% speedup.
With the increased batch size, this can often be brought to 2.5x faster.

:::{warning} FP16 requires modern GPUs
Without a GPU with FP16 CUDA cores, you may find that FP16 actually slows
your program. You may still see a benefit from the reduced memory usage though.
:::

## Background preprocessing

We can further speed things up by doing some preprocessing, such as
tokenization and dialogue state tracking, in the background thread. This can be
enabled by setting `--num-workers` to a value greater than 0. A good rule of
thumb is to set `--num-workers` to the number of CPU cores you have PER GPU.
On my server, there are 8 cores per GPU, so I will set it to 8.

    parlai train --dict-file dictfile --model transformer/generator \
        --task convai2 --num-epochs 1.0 --batchsize 64 --n-layers 8  \
        --embedding-size 256 --ffn-size 1024 --optimizer adam --learningrate 1e-3 \
        --skip-generation true \
        --dynamic-batching full \
        --fp16 true --fp16-impl mem_efficient --batchsize 128 --eval-batchsize 256 \
        --num-workers 8

## Use multiple GPUs

If you have multiple GPUs, you can utilize them by switching from `train` to
`multiprocessing_train`. If you have 4 GPUs, you'll find training should be
roughly 3.5x faster. The arguments for the training are left otherwise the same.

    parlai multiprocessing_train \
        --dict-file dictfile --model transformer/generator \
        --task convai2 --num-epochs 1.0 --batchsize 64 --n-layers 8  \
        --embedding-size 256 --ffn-size 1024 --optimizer adam --learningrate 1e-3 \
        --skip-generation true \
        --dynamic-batching full \
        --fp16 true --fp16-impl mem_efficient --batchsize 128 --eval-batchsize 256 \
        --num-workers 8

Note that we leave batchsize the same: we use the batchsize PER GPU. We expect
the batchsize will be effectively 4x'd by using 4 GPUs.

Note that launching a multiprocessing train can take time to "warm up" and it
may take time for you to realize speed improvements when using small, toy
training runs like in this document. If we train for 4 epochs instead, our FP16
large-batch run takes 614 seconds and the multi-GPU training takes 222 seconds.
Also, if you have access to a SLURM cluster, `distributed_train` is sometimes
faster than `multiprocessing_train`. With SLURM, multi-GPU training takes 167 seconds
for 4 epochs, or about a 3.7x speedup compared to single GPU.

Similarly, we also have the `multiprocessing_eval` command, for using multiple
GPUs in evaluation.

:::{danger}
This should never be mixed with options like `--model-parallel true`
or `--data-parallel true`, as those options use different GPUs without
multiprocessing.  The BlenderBot3B and BlenderBot9B models both use those
options, so this should be used with care.
:::
