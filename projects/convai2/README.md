# ConvAI2 Competition for NIPS 2018

This directory contains code and examples for the ConvAI2 competition.

The ConvAI2 dataset is already in ParlAI in parlai/agents/tasks/convai2 and is accesible by using the command line parameter `--task convai2`.
The several versions of the dataset can be access with `convai2:none`, `convai2:self`, and `convai2:self_revised`.

These correspond to "no persona", "original self persona", and "revised self persona" in the original [PersonaChat](https://arxiv.org/pdf/1801.07243.pdf) paper.
However, in contrast to that dataset, we have modified the preprocessing, merged the valid and test set, and are generating a new hidden test set.

We have also generated a version of the dataset with candidates available from the perspective of each speaker. This supports training / evaluating using a ranking loss to a greater degree.


## Examples

You can see examples of models prepared for the PersonaChat task [here](https://github.com/facebookresearch/ParlAI/tree/master/projects/personachat), which should generally transfer pretty easily to this task.

You can run examples of training on this task in the `baselines` folder in this directory.

For example, you can download and evaluate a pre-trained seq2seq model (based on ParlAI's implementation in parlai/agents/seq2seq) using `baselines/seq2seq/eval.sh`, or train your own with the same parameters using the `train.sh` file in that directory.


## Contributing

You can submit your model by creating a PR with a folder for your model in the `submissions` folder in this directory.
While you don't *have* to train your model in ParlAI, we encourage you to provide full model code and a similar `train.sh` file so that others can reproduce your results.

In order to have your trained model evaluated for the competition, however, you will have to submit code similar to the `eval.sh` script in the `baselines/seq2seq` folder.
We will use the eval_model command in the same way, using the `valid` datatype for the validation leaderboard and a hidden test set for final evaluation (you don't need to support batching or CUDA with your model, though).
This will require at least writing an inference-time wrapper according to the ParlAI format.
If you need help hosting your trained model file, contact @alexholdenmiller (email `ahm at fb`).
