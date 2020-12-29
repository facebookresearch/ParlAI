# Model chat crowdsourcing task

This task will collect conversations between a human and a model. After each response by the model, the human will optionally be prompted to annotate the model's response by selecting checkboxes that represent customizable attributes of that response.

**NOTE**: See [parlai/crowdsourcing/README.md](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/README.md) for general tips on running `parlai.crowdsourcing` tasks, such as how to specify your own YAML file of configuration settings, how to run tasks live, how to set parameters on the command line, etc.

## Launching

Call `run.py` to run this task with the default parameters, as set by `conf/example.yaml`. Some parameters that you can adjust include where to save data, lists of workers to soft-block, the maximum response time, etc.

The models used for chat should be placed in a single base folder, specified by the `mephisto.blueprint.base_model_folder` flag. Each model file should be named `model` and should be located in its own subfolder, and all options used when running the model should be listed in a JSON file named `model.opt` in that subfolder. For example, suppose your base model folder is `~/ParlAI/data/models/`, and you wish to run 10 conversations of the `blender_90M` model. Suppose that the contents of `~/ParlAI/data/models/` looks like this:
```
blender/blender_3B:
model
model.dict
model.dict.opt
model.opt

blender/blender_90M:
model
model.dict
model.dict.opt
model.opt
```
You might then run the following command:
```
python parlai/crowdsourcing/tasks/model_chat/run.py \
mephisto.blueprint.base_model_folder=~/ParlAI/data/models \
mephisto.blueprint.conversations_needed_string=\"blender/blender_90M:10\"
```

## Passing in task config files

The following flags can be passed in to specify filepaths for overriding the text shown to the workers and the settings of the annotation categories. If they are not specified, the defaults in the `task_config/` folder will be used.
- `mephisto.blueprint.annotations_config_path`: JSON file configuring annotation categories. Set this flag to `""` to disable annotation of model responses.
- `mephisto.blueprint.left_pane_text_path`: HTML to show on the left-hand pane of the chat window.
- `mephisto.blueprint.onboard_task_data_path`: JSON specifying parameters for testing workers during onboarding. Onboarding is only run if model responses will be annotated.
- `mephisto.blueprint.task_description_file`: HTML to show on the initial task-description page shown to the worker.

## Onboarding

In `worlds.py`, modify `ModelChatOnboardWorld.check_onboarding_answers()` to change the worker selection criteria.

## Human+model image chat

`run_image_chat.py` can be run to chat with a model about an image: each conversation will begin with a selected image, and then the human and model will chat about it.

This code replaces the old `parlai/mturk/tasks/image_chat/` and `parlai/mturk/tasks/personality_captions/` tasks, which are deprecated and can be accessed with `$ git checkout v0.10.0`. Those tasks featured the ability to compare two possible captions to an image and rate which one is more engaging: this functionality has now been replaced by the [ACUTE-Eval](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing/tasks/acute_eval) task. 

### Setup

Before running image chat HITs, you need to save a list of images that the humans and models will chat about. Do this by running `scripts/save_image_contexts.py`, which will loop over an image-based dataset and save information corresponding to a certain number of unique images. This script accepts any arguments used by `parlai display_data`, for instance:
```
python parlai/crowdsourcing/tasks/model_chat/scripts/save_image_contexts.py \
--task image_chat \
--datatype test \
--num-examples 10
```

### Options

Some options for running human+model image chat are as follows:
- `mephisto.blueprint.model_opt_path`: path to a YAML file listing all models to be chatted with, as well as the ParlAI flags for running each one. See `task_config/image_model_opts.yaml` for an example.
- `mephisto.blueprint.num_conversations`: the total number of conversations to collect.
- `mephisto.blueprint.image_context_path`: the path to the file saved by `scripts/save_image_contexts.py` during setup.
- `mephisto.blueprint.stack_folder`: a folder in which to store a stack file that will keep track of which crowdsource workers have chatted with which models about which images. The stack will ensure that no worker chats about the same image more than once and that conversations about images are collected uniformly among all models.
- `mephisto.blueprint.evals_per_image_model_combo`: the maximum number of conversations collected for each combination of image and model. For instance, if this is set to 3 and your 2 models are `model_1` and `model_2`, each image will have 6 conversations collected about it, 3 with `model_1` and 3 with `model_2`.

Note that onboarding is not currently supported with human+model image chat: use `ModelChatOnboardWorld` in `worlds.py` as a guide for how to set up onboarding for your specific task.
