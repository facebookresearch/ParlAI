# ConvAI2 Competition for NIPS 2018

This directory contains code and examples for the ConvAI2 competition.
See the [website](http://convai.io/) for more details.

The ConvAI2 dataset is already in ParlAI in parlai/agents/tasks/convai2 and is accessible by using the command line parameter `--task convai2`.

To see this data, try:
```
python ~/ParlAI/examples/display_data.py -t convai2 -dt train
```


## Examples

You can run examples of training on this task in the `baselines` folder in this directory.

For example, you can download and evaluate a pre-trained seq2seq model (based on ParlAI's implementation in parlai/agents/seq2seq) using `baselines/seq2seq/eval.sh`, or train your own with the same parameters using the `train.sh` file in that directory.


## Contributing for the final evaluation

You can submit your model by creating a PR with a folder for your model in the `submissions` folder in this directory.
You do not have to submit your competition model before the evaluation deadlines, but feel free to submit additional simple baselines for others to compare to or use to get started.
While you don't *have* to train your model in ParlAI, we encourage you to provide full model code and a similar `train.sh` file so that others can reproduce your results.

In order to have your trained model evaluated for the competition, however, you will have to submit code similar to the `eval.sh` script in the `baselines/seq2seq` folder.
We will use the eval_model command in the same way, using the `valid` datatype for the validation leaderboard and a hidden test set for final evaluation (you don't need to support batching or CUDA with your model, though).
This will require at least writing an inference-time wrapper according to the ParlAI format.
If you need help hosting your trained model file, contact @alexholdenmiller (email `ahm at fb`).


## Remarks on dataset versions

The several versions of the dataset can be accessed with `convai2:none`, `convai2:self`, and `convai2:self_revised`.

These correspond to "no persona", "original self persona", and "revised self persona" in the original [PersonaChat](https://arxiv.org/pdf/1801.07243.pdf) paper.
However, in contrast to that dataset, we have modified the preprocessing, merged the valid and test set, and are generating a new hidden test set. We have also added training examples from the perspective of both speakers and additional candidates to support training / evaluating using a ranking loss.

You can use the `none` mode to contrast your model's ability to take advantage of its persona with `self`, but `self` is the default setting for the task.

You can see examples of models prepared for the original (non-competition) PersonaChat task [here](https://github.com/facebookresearch/ParlAI/tree/master/projects/personachat), which should generally transfer pretty easily to this task. However, we will continue to add baseline models to this project folder as we run experiments.
