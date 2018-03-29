# ConvAI2 Competition for NIPS 2018

This directory contains code and examples for the ConvAI2 competition.
See the [website](http://convai.io/) for more details.

The ConvAI2 dataset is in ParlAI in parlai/agents/tasks/convai2 and is accessible by using the command line parameter `--task convai2`.

For example, to see the training data, try:
```
python ~/ParlAI/examples/display_data.py -t convai2 -dt train
```


## Examples

You can run examples of training on this task in the `baselines` folder in this directory.

For example, you can download and evaluate a pre-trained seq2seq model (based on ParlAI's implementation in parlai/agents/seq2seq) using `baselines/seq2seq/eval.py`, or train your own with the same parameters using the `train.py` file in that directory.


## Submitting models for evaluation

Submission guidelines will be released soon


## Remarks on dataset versions

The several versions of the dataset can be accessed with `convai2:self`, `convai2:self_revised` and `convai2:none`.

These correspond to "original self persona", "revised self persona" and "no persona" in the original [PersonaChat](https://arxiv.org/pdf/1801.07243.pdf) paper.
However, in contrast to that dataset, we have modified the preprocessing and are generating a new hidden test set. We have also added training examples from the perspective of both speakers and additional candidates to support training / evaluating using a ranking loss.

You can use the `none` mode to contrast your model's ability to take advantage of its persona with `self`, but `self` is the default setting for the task, and you will be evaluate on 'self' and 'self_revised'.

You can see examples of models prepared for the original (non-competition) PersonaChat task [here](https://github.com/facebookresearch/ParlAI/tree/master/projects/personachat), which should generally transfer pretty easily to this task. However, we will continue to add baseline models to this project folder as we run experiments.
