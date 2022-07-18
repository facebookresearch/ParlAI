# Crowdsourcing task for single-model per-turn and per-dialogue evaluations

Code to run human crowdworker evaluations on a single model, one of the types of evaluation techniques explored in [Smith, et al. "Human Evaluation of Conversations is an Open Problem: comparing the sensitivity of various methods for evaluating dialogue agents" (2022)](https://arxiv.org/abs/2201.04723). To cite:
```
@misc{smith2022human,
      title={Human Evaluation of Conversations is an Open Problem: comparing the sensitivity of various methods for evaluating dialogue agents}, 
      author={Eric Michael Smith and Orion Hsu and Rebecca Qian and Stephen Roller and Y-Lan Boureau and Jason Weston},
      year={2022},
      eprint={2201.04723},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

This crowdsourcing task consists of a conversation between a Turker and a model. The task will collect evaluations of engagingness, humanness, and interestingness after every model response (SM-Turn in the paper), as well as final ratings of 1-to-5 Likert scores of those same metrics at the end of the conversation (SM-Dialog in the paper).

## Collecting evaluations

To launch HITs, run `python run.py`. All Hydra flags are as in the base human/model crowdsourcing task in [`parlai/crowdsourcing/tasks/model_chat/`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/model_chat), which this crowdsourcing task is a custom version of.

To specify the set of models that you want to evaluate, pass in a custom YAML file with the `mephisto.blueprint.model_opt_path` flag. The example `task_config/model_opts.yaml` file specifies the set of models evaluated in this paper:
- `blender_3B`: (**BlenderBot3B** in the paper) The 2.7-billion parameter variant of the [BlenderBot 1.0 model](https://parl.ai/projects/recipes/)
- `blender_3B_beam_min_length_0`: (**BlenderBot3B-M0**) BlenderBot3B is typically used with a minimum generation length of 20 tokens: this variant removes the minimum generation length.
- `blender_90M`: (**BlenderBot90M**) The variant of BlenderBot 1.0 with 90 million parameters, trained on the same datasets as BlenderBot3B.
- `reddit_3B`: (**Reddit3B**) Pretraining-only BlenderBot3B, without any fine-tuning on dialogue datasets.

## Running analysis

Call `python analysis/compile_results.py` to analyze single-model evaluations collected with this crowdsourcing task. Required flags for this script are:
- `--task-name`: The Mephisto task name used when collecting evaluations
- `--output-folder`: The folder to save analysis output files to

Set `--filter-uniform-hits` to `True` to filter out any HITs for which the Turker's annotations were the exact same on each turn of the conversation, as a quality check.

Features of this script include:
- Filtering out HITs with acceptability violations, and saving a file of all Turkers who violated acceptability checks
- Saving a file of all per-turn ratings (SM-Turn scores) and per-dialogue ratings (SM-Dialog scores) across all conversations
- Saving a file of the aggregate rates of selecting each annotation bucket across all turns (i.e. SM-Turn)
- Saving statistics about the distribution of Likert scores for each question asked at the end of each conversation (i.e. SM-Dialog)
