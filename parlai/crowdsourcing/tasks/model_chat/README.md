# Model chat crowdsourcing task

This task will collect conversations between a human and a model. After each response by the model, the human will optionally be prompted to annotate the model's response by selecting checkboxes that represent customizable attributes of that response.

**NOTE**: See [parlai/crowdsourcing/README.md](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/README.md) for general tips on running `parlai.crowdsourcing` tasks, such as how to specify your own YAML file of configuration settings, how to run tasks live, how to set parameters on the command line, etc.

## Launching

Call `run.py` to run this task with the default parameters, as set by `hydra_configs/conf/example.yaml`. Some parameters that you can adjust include where to save data, lists of workers to soft-block, the maximum response time, etc.

Set `mephisto.blueprint.model_opt_path` to specify a path to a YAML file listing all models to be chatted with, as well as the ParlAI flags for running each one. See `task_config/model_opts.yaml` for an example.

Set `mephisto.blueprint.chat_data_folder` to the root folder that you want all results of HITs to be saved in: all results will be saved to a date folder (of format `2021_01_15`) within that root folder.

Set `mephisto.blueprint.final_rating_question` to specify the question to ask the worker at the end of the task, for which the worker will respond with a 1-to-5 Likert score. Separate multiple questions with a `|`.

## Passing in task config files

The following flags can be passed in to specify filepaths for overriding the text shown to the workers and the settings of the annotation categories. If they are not specified, the defaults in the `task_config/` folder will be used.
- `mephisto.blueprint.annotations_config_path`: JSON file configuring annotation categories.
  - Set this flag to `""` to disable annotation of model responses.
  - If the text of the annotation categories is very long, you can set the `"show_line_breaks"` field (see `task_config/annotations_config.json`) to `true` in order to put a line break between the checkbox and text string of each category.
- `mephisto.blueprint.left_pane_text_path`: HTML to show on the left-hand pane of the chat window.
- `mephisto.blueprint.onboard_task_data_path`: JSON specifying parameters for testing workers during onboarding. Onboarding is only run if model responses will be annotated.
- `mephisto.blueprint.task_description_file`: HTML to show on the initial task-description page shown to the worker.

## Onboarding

Set the `"min_correct"`, `"max_incorrect"`, and `"max_failures_allowed"` fields in the JSON file passed to `mephisto.blueprint.onboard_task_data_path` in order to specify how many onboarding questions workers can pass/fail on while still passing onboarding, as well as how many times they are allowed to re-take the onboarding before being soft-blocked. (See `task_config/onboard_task_data.json` for an example.)

You can further modify the worker selection criteria in `handleOnboardingSubmit` in `frontend/components/onboarding_components.jsx`.

## Human+model image chat

Call `run.py conf=example_image_chat` to chat with a model about an image: each conversation will begin with a selected image, and then the human and model will chat about it. This task is run with the parameters defined in `hydra_configs/conf/example_image_chat.yaml`.

This code replaces the old `parlai/mturk/tasks/image_chat/` and `parlai/mturk/tasks/personality_captions/` tasks, which are deprecated and can be accessed with `git checkout v0.10.0`. Those tasks also featured the ability to compare two possible captions to an image and rate which one is more engaging: this functionality has now been replaced by the [ACUTE-Eval](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/acute_eval) task.

### Setup

Before running image chat HITs, you need to save a list of images that the humans and models will chat about. Do this by running `scripts/save_image_contexts.py`, which will loop over an image-based dataset and save information corresponding to a certain number of unique images. This script accepts any arguments used by `parlai display_data`, for instance:
```bash
python parlai/crowdsourcing/tasks/model_chat/scripts/save_image_contexts.py \
--task image_chat \
--datatype test \
--num-examples 10
```

### Options

Some options for running human+model image chat are as follows:
- `mephisto.blueprint.num_conversations`: the total number of conversations to collect.
- `mephisto.blueprint.image_context_path`: the path to the file saved by `scripts/save_image_contexts.py` during setup.
- `mephisto.blueprint.stack_folder`: a folder in which to store a stack file that will keep track of which crowdsource workers have chatted with which models about which images. The stack will ensure that no worker chats about the same image more than once and that conversations about images are collected uniformly among all models.
- `mephisto.blueprint.evals_per_image_model_combo`: the maximum number of conversations collected for each combination of image and model. For instance, if this is set to 3 and your 2 models are `model_1` and `model_2`, each image will have 6 conversations collected about it, 3 with `model_1` and 3 with `model_2`.
- `mephisto.blueprint.world_file`: the path to the Python module containing the class definition for the chat World, used for setting the logic for each turn of the conversation, when to end the conversation, actions upon shutdown, etc. (The onboarding World, if it exists, will be defined in this module as well.) Modify this value if you would like to write your own World class without having to create a new Blueprint class.

Note that onboarding is not currently supported with human+model image chat: use `ModelChatOnboardWorld` in `worlds.py` as a guide for how to set up onboarding for your specific task.

## Analysis

Run `analysis/compile_results.py` to compile and save statistics about collected human+model chats. The `ModelChatResultsCompiler` in that script uses dummy annotation buckets by default; set `--problem-buckets` in order to define your own. Set `--task-name` to specify a specific task to compile.
