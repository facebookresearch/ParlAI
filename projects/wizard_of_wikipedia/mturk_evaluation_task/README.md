# Wizard of Wikipedia MTurk evaluation

ParlAI is currently in the process of upgrading MTurk to
[Mephisto](https://github.com/facebookresearch/Mephisto). As part of this
process we have archived a number of older tasks. If you require need to run
this evaluation, you may rewind back to the
[`mturk_archive`](https://github.com/facebookresearch/ParlAI/tree/mturk_archive)
tag:

```bash
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
cd ~/ParlAI
git checkout mturk_archive
```

If you just need to read the code, for reference, you may browse it
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/projects/wizard_of_wikipedia/mturk_evaluation_task).

# Old README

This is the Mechanical Turk task for evaluating models trained on the Wizard of Wikipedia task.

As an example, we have one of the pre-trained models loaded inside the task. Please edit `config` in `run.py` to swap out the model for one of yours.

In order to run the task with two humans speaking to each other, run with the flag `--human-eval True`.
