# Frequently Asked Questions

This document contains a number of questions that are regularly asked on GitHub Issues.

## Why is my model not generating a response?

For a generative model, check that `--skip-generation` is set to `False`.

## Why can't I reproduce the results of an evaluation on a task with a pretrained model?

One common culprit for this is that the flags for that task may not be
correctly set. When loading a pretrained checkpoint, all of the parameters for
the model itself will be loaded from the model's `.opt` file, but all
task-specific parameters will need to be re-specified.

If results differ by a few small decimal places, this can often be attributed
to differences in hardware or software environment.

## I want to generate a lot of responses to fixed utterances

The easiest way to do this is to [create a
teacher](tutorial_task) in ParlAI Dialog Format. Then, use
`eval_model` with world logging to store all the responses:

```
parlai eval_model -t fromfile:parlaiformat --fromfile-datapath yourtextfile.txt \
    -mf some_model_file --world-logs outputfile
```

## Why is my generative model's perplexity so high (>1000) when evaluating?

One first thing to check is whether there is a problem with your dictionary or
token embeddings, because this high perplexity implies that the model is very
bad at predicting the next token in a string of text.

## I changed my teacher and now its tests won't pass.

Take a careful look at the diff outputs that those tests produce. If the results
look expected, then you can update the regression fixtures (stored, expected
results) with:

`pytest --force-regen parlai/tasks/TASK_NAME_HERE/test.py`

