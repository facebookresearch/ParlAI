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

For example, you can download and interact with a pre-trained seq2seq model (based on ParlAI's implementation in parlai/agents/seq2seq) using `baselines/seq2seq/interact.py`, or train your own with the same parameters using the `train.py` file in that directory.


## Submitting models for evaluation

To submit an entry, create a private repo with your model that works with our evaluation code, and share it with the following github accounts: edinan@fb.com, jaseweston@gmail.com, jju@fb.com, kshuster@fb.com.

More specifically, create a repo with a directory similar to the baseline models directory, with a `eval_XXX.py` file with each metric you would like to be evaluated against. These files should import the appropriate eval function from this directory. For example, the `eval_f1.py` file in `baselines/seq2seq/` sets up the right parameters for the model and then imports the evaluation function from the base `eval_f1.py` file in this directory and runs it.

We will then run the automatic evaluations against the hidden test set (which is in the same format as the validation set you can access) and update the leaderboard.
You can submit a maximum of once per month.
We will use the same submitted code for the top performing models for computing human evaluations when the submission system is locked on September 30th.


## Remarks on dataset versions

The several versions of the dataset can be accessed with `convai2:self`, `convai2:self_revised` and `convai2:none`.

These correspond to "original self persona", "revised self persona" and "no persona" in the original [PersonaChat](https://arxiv.org/pdf/1801.07243.pdf) paper.
However, in contrast to that dataset, we have modified the preprocessing and are generating a new hidden test set. We have also added training examples from the perspective of both speakers and additional candidates to support training / evaluating using a ranking loss.

You can use the `none` mode to contrast your model's ability to take advantage of its persona with `self`, but `self` is the default setting for the task, and you will be evaluated on 'self_orignal' for the leaderboard.

You can see examples of models prepared for the original (non-competition) PersonaChat task [here](https://github.com/facebookresearch/ParlAI/tree/master/projects/personachat), which should generally transfer pretty easily to this task. However, we will continue to add baseline models to this project folder as we run experiments.
