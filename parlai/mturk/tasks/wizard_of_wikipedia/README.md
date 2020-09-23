# Wizard of Wikipedia Data Collection Task

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
[here](https://github.com/facebookresearch/ParlAI/tree/mturk_archive/parlai/mturk/tasks/wizard_of_wikipedia).

# Old README

The task involves two people holding a conversation. One dialog partner
chooses a topic to discuss, and then dialog proceeds.

One partner is the Wizard, who has access to retrieved external
information conditioned on the last two utterances, as well as
information regarding the chosen topic.

The other partner is the Apprentice, who assumes the role of someone
eager to learn about the chosen topic.

This task was used to collect the dataset for the wizard_of_wikipedia task
contained in the tasks folder. A detailed description of the project may
be found in [Dinan et al. (ICLR 2019)](https://arxiv.org/abs/1811.01241).
For more details and pre-trained models, please see the
[Wizard of Wikipedia project page](https://github.com/facebookresearch/ParlAI/tree/master/projects/wizard_of_wikipedia).
The project page contains the Mechanical Turk task used to evaluate
pre-trained models in this paper.
