# Per-turn Evaluation Crowdsourcing Task
Code to run human crowdworker evaluations on a pair of conversational models, one of the types of evaluation techniques explored in [Smith, et al. "Human Evaluation of Conversations is an Open Problem: comparing the sensitivity of various methods for evaluating dialogue agents" (2022)](https://arxiv.org/abs/2201.04723). To cite:
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

In this task, Turkers compare two models on every turn of a conversation by choosing the response based on a certain criteria. Each time the Turker's partner responds, the Turker will see two possible responses, one from each of two models. The Turker will choose which of the two responses is better according to some metric, provide a justification of why the Turker chose that response, and then that choice will be accepted as the bot’s response and the conversation will continue. 

## Launching

Call `run.py` to run this task with the default parameters, as set by `hydra_configs/conf/example_model_comparison.yaml`. You can also manually adjust the parameters by setting that flag name as part of the run command. For example, `python run.py mephisto.blueprint.conversations_needed_string="blender_90M:blender_3B:10"` to compare the `blender_90M` model with the `blender_3B` model.

Set `mephisto.blueprint.model_opt_path` to specify a path to a YAML file listing all models to be chatted with, as well as the ParlAI flags for running each one. See `task_config/model_opts.yaml` for an example.

Set `mephisto.blueprint.chat_data_folder` to the root folder that you want all results of HITs to be saved in: all results will be saved to a date folder (of format 2021_01_15) within that root folder.

## Passing in task config files

The following flags can be passed in to specify filepaths for overriding the text shown to the workers and the settings of the annotation categories. If they are not specified, the defaults in the `task_config/` folder will be used.
- `mephisto.blueprint.left_pane_text_path`: HTML to show on the left-hand pane of the chat window.
- `mephisto.blueprint.onboard_task_data_path`: JSON specifying parameters for testing workers during onboarding. Onboarding is only run if model responses will be annotated.
- `mephisto.blueprint.task_description_file`: HTML to show on the initial task-description page shown to the worker.

## Onboarding

We provide a simple onboarding task (see `task_config/onboard_task_data__engaging.json`) to act as a "qualification" for first-time users. For each turn, users must select the correct response out of two given responses, in order to pass the onboarding and move onto the actual task. Multiple attempts are allowed, but if all of these attempts fails, they become blocked and can no longer do the task.

To change the worker selection criteria for onboarding, see `handleOnboardingSubmit` in `frontend/components/onboarding_components.jsx`.

## Analysis
Run `analysis/compile_results.py` to compile and save statistics about collected human+model chats. Set `--results-folders` to the value of `mephisto.blueprint.chat_data_folder` used when running HITs. Specifically, the analysis file:
- Has most of the features from `parlai/crowdsourcing/tasks/model_chat`'s analysis script (doesn't include analysis of annotation buckets, since it isn't used here)

## Reproducing Paper Results
The following section contains instructions to reproduce the results in our paper. In our paper, we run 3 sets of model comparisons:

### Model Comparisons
To run a pairwise model comparison annotation task, create a `.yaml` config using the template provided by `hydra_configs/conf/example_model_comparison.yaml`. Set the models and number of conversations to collect ratings for in the `mephisto.blueprint.conversations_needed_string` field, following the format `${model_A}:${model_B}:${num_conversations}`. For example, `"blender_90M:blender_3B:10"` compares the `blender_90M` model with the `blender_3B` model and collects 10 conversations.

Here are the model comparisons ran in the paper, and corresponding values for `conversations_needed_string`:
- Size (BlenderBot3B vs. BlenderBot90M): `"blender_90M:blender_3B:60"`
- Generation Length (BlenderBot3B vs. BlenderBot3B-M0): `"blender_3B:blender_3B_beam_min_length_0:60"`
- Fine-tuning (BlenderBot3B vs. Reddit3B): `"blender_3B:reddit_3B:60"`

To run a crowdsourcing task, run the following with your modified parameters:
```
CONF=example_model_comparison && # Replace with your conf
REQUESTER_NAME=mturk_sandbox && # Replace with your Mephisto requester
python run.py \
conf=${CONF} \
mephisto.provider.requester_name=${REQUESTER_NAME}
```

As described above, you can also set config fields directly in the command line.

### Evaluation Metric
To change the metric that annotators use to select the better conversational response, change the `mephisto.blueprint.task_question` field to best reflect the evaluation metric you want to use. These are the metrics we used in our paper, and the corresponding task questions:
- Engagingness: “Which next response from your partner would you prefer in a long conversation?”
- Humanness: "Which next response from your partner sounds more human?”
- Interestingness: “If you had to say one of these responses is interesting and one is boring, which would you say is more interesting?”

You can also change the onboarding task to better reflect your evaluation metric. To do this, create a `.json` file with onboarding task questions and correct responses, and set `mephisto.blueprint.onboard_task_data_path` in the config to that filepath. We provide examples for all 3 eval metrics described above in the `task_config/` folder. The example provided in `task_config/onboard_task_data__engaging.json` requires users to select the most engaging response. To change the question asked during onboarding, set `mephisto.blueprint.annotation_question`.

We recommend modifying `mephisto.task.task_name` to describe the run parameters, such as the models being compared, and the evaluation metric.