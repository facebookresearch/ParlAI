# Model chat crowdsourcing task

This task will collect conversations between a human and a model. After each response by the model, the human will optionally be prompted to annotate the model's response by selecting checkboxes that represent customizable attributes of that response.

**NOTE**: See [parlai/crowdsourcing/README.md](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/README.md) for general tips on running `parlai.crowdsourcing` tasks, such as how to specify your own YAML file of configuration settings, how to run tasks live, how to set parameters on the command line, etc.

## Launching

Call `run.py` to run this task with the default parameters, as set by `conf/example.yaml`.

Some parameters that you can adjust include where to save data, lists of workers to soft-block, the maximum response time, etc.

## Passing in task config files

The following flags can be passed in to specify filepaths for overriding the text shown to the workers and the settings of the annotation categories. If they are not specified, the defaults in the `task_config/` folder will be used.
- `mephisto.blueprint.annotations_config_path`: JSON file configuring annotation categories. Set this flag to "" to disable annotation of model responses.
- `mephisto.blueprint.left_pane_text_path`: HTML to show on the left-hand pane of the chat window
- `mephisto.blueprint.onboard_task_data_path`: JSON specifying parameters for testing workers during onboarding. Onboarding is only run if model responses will be annotated
- `mephisto.blueprint.task_description_file`: HTML to show on the initial task-description page shown to the worker

## Onboarding

In `worlds.py`, modify `ModelChatOnboardWorld.check_onboarding_answers()` to change the worker selection criteria.

## Human+model image chat

`run_image_chat.py` can be run to chat with a model about an image: each conversation will begin with a selected image, and then the human and model will chat about it.

This code replaces the old `parlai/mturk/tasks/image_chat/` and `parlai/mturk/tasks/personality_captions/` tasks; 

{{{TODO: mention removed features}}}

### Setup

Before running image chat HITs, you need to save a list of images that the humans and models will chat about. Do this by running `scripts/save_image_contexts.py`, which will loop over a dataset containing images and save information corresponding to a certain number of unique images to a file. This script accepts any arguments used by `parlai display_data`, for instance:
```
python parlai/crowdsourcing/tasks/model_chat/scripts/save_image_contexts.py \
--task image_chat \
--datatype test \
--num-examples 10
```

### Options

{{{TODO: say that there is no onboarding for this variant}}}
