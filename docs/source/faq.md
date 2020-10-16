# Frequently Asked Questions

This document contains a number of questions that are regularly asked on GitHub Issues.


**Why is my model not generating a response?**

For a generative model, check that `--skip-generation` is set to `False`.


**Why can't I reproduce the results of an evaluation on a task with a pretrained model?**

One common culprit for this is that the flags for that task may not be correctly set. When loading a pretrained checkpoint, all of the parameters for the model itself will be loaded from the model's `.opt` file, but all task-specific parameters will need to be re-specified.


**Why is my generative model's perplexity so high (>1000) when evaluating?**

One first thing to check is whether there is a problem with your dictionary or token embeddings, because this high perplexity implies that the model is very bad at predicting the next token in a string of text.
