# Per-turn Evaluation Crowdsourcing Task
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

We provide a simple onboarding task (see `task_config/onboard_task_data.json`) to act as a "qualification" for first-time users. Users must select the correct response out of two given responses, in order to pass the onboarding and move onto the actual task. Multiple attempts are allowed, but if all of these attempts fails, they become blocked and can no longer do the task.

To change the worker selection criteria for onboarding, see `handleOnboardingSubmit` in `frontend/components/onboarding_components.jsx`.

## Analysis
Run `analysis/compile_results.py` to compile and save statistics about collected human+model chats. Set `--results-folders` to the value of `mephisto.blueprint.chat_data_folder` used when running HITs. Specifically, the analysis file:
- Has most of the features from `model_chat`'s analysis file (doesn't include analysis of annotation buckets, since it isn't used here) 
- Based on the collected data, produces a `scipy.stats.binom_test` in order to test the statistical significance of the disparity in how often the user selects one model vs. the other
- Produces a Labor vs Proportion of Significant Results graph, similar to that of Figure 5 of the ACUTE-EVAL paper.