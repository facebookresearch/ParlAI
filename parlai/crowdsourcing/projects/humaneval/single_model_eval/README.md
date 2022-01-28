# Crowdsourcing task for single-model per-turn and per-dialogue evaluations

Code to run the human crowdworker evaluations on a single model at a time from [Smith, et al. "Human Evaluation of Conversations is an Open Problem: comparing the sensitivity of various methods for evaluating dialogue agents" (2022)](https://arxiv.org/abs/2201.04723). This crowdsourcing task will collect both evaluations of engagingness, humanness, and interestingness after every model response (SM-Turn in the paper) and a final evaluation of 1-to-5 Likert scores of those same metrics at the end of the conversation (SM-Dialog in the paper).

## Collecting evaluations

To launch HITs, run `python run.py` in this folder. All Hydra flags are as in the human/model crowdsourcing task in [`parlai/crowdsourcing/tasks/model_chat/`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/model_chat), which this crowdsourcing task is a custom version of.

To specify the set of models that you want to evaluate, pass in a custom YAML file with the `mephisto.blueprint.model_opt_path` flag. The example `task_config/model_opts.yaml` file specifies the set of models evaluated in the paper:
- `blender_3B`: (**BlenderBot3B** in the paper) The 2.7-billion parameter variant of the [BlenderBot 1.0 model](https://parl.ai/projects/recipes/)
- `blender_3B_beam_min_length_0`: (**BlenderBot3B-M0**) BlenderBot3B is typically used with a minimum generation length of 20 tokens: this variant removes this minimum length.
- `blender_90M`: (**BlenderBot90M**) The variant of BlenderBot with 90 million parameters, trained on the same datasets as BlenderBot3B.
- `reddit_3B`: (**Reddit3B**) Pretraining-only BlenderBot3B, without any fine-tuning on dialogue datasets.

## Running analysis

Call `python analysis/compile_results.py` to analyze these single-model evaluation results. Required flags for this script are:
- `--task-name`: The Mephisto task name used when collecting evaluations
- `--output-folder`: Folder to save analysis output files to

Set `--filter-uniform-hits` to `True` to filter out any HITs in which the worker's annotations were the exact same on each turn of the conversation, as a quality check.

Features of this script include:
- Filtering out HITs with acceptability violations, and saving a file of all Turkers violating acceptability checks
- Saving a file of all per-turn ratings (SM-Turn scores) and per-dialogue ratings (SM-Dialog scores) across all conversations
- Saving a file of the aggregate rates of selecting each annotation bucket
- Saving statistics about the distribution of Likert scores for each question asked at the end of each conversation
