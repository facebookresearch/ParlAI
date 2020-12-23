# ACUTE-Eval

# TODO: revise all!

## Paper information

Margaret Li, Jason Weston, Stephen Roller.
_[ACUTE-EVAL: Improved Dialogue Evaluation with Optimized Questions and Multi-turn Comparisons](https://arxiv.org/abs/1909.03087)_.

## Citation

If you use this evaluation method in your own work, please cite with the
following BibTex entry:

    @misc{li2019acuteeval,
      title={ACUTE-EVAL: Improved Dialogue Evaluation with Optimized Questions and Multi-turn Comparisons},
      author={Margaret Li and Jason Weston and Stephen Roller},
      year={2019},
      journal={Advances in Neural Information Processing Systems, Conversational AI Workshop},
      url={https://arxiv.org/abs/1909.03087}
    }

# Code Instructions

Once you have installed [ParlAI](https://github.com/facebookresearch/ParlAI/#installing-parlai) and [Mephisto](https://github.com/facebookresearch/mephisto/blob/master/docs/quickstart.md), follow the instructions below.

The `run.py` script is designed to allow you to run this entire task from command line with an invocation like

    python parlai/crowdsourcing/tasks/acute_eval/run.py \
    mephisto.blueprint.pairings_filepath=${REPO_FOLDER}/parlai/crowdsourcing/tasks/acute_eval/pairings.jsonl

**NOTE**: See [parlai/crowdsourcing/README.md](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/README.md) for general tips on running `parlai.crowdsourcing` tasks, such as how to specify your own YAML file of configuration settings, how to run tasks live, how to set parameters on the command line, etc.

## Formatting conversation data

This task code assumes that you've parsed and saved your collected conversations in a simple .jsonl format. The path to this file should be passed in as `mephisto.blueprint.pairings_filepath=${PATH_TO_FILE}`.

This is a template of the expected format with the minimal expected fields:

    {
      "is_onboarding": false,
      "speakers_to_eval": ["first_modelname", "second_modelname"],
      "dialogue_ids": [dialogue_1_id, dialogue_2_id],
      "dialogue_dicts": [
        {
          "speakers": ["first_modelname", "other_speaker"],
          "dialogue": [
            {"id": "model1", "text": "Hi"},
            {"id": "other_speaker", "text": "Hi back"},
            ...
          ]
        },
        {
          "speakers": ["other_speaker", "second_modelname"],
          "dialogue": [
            {"id": "model1", "text": "Hi"},
            {"id": "other_speaker", "text": "Hi back"},
            ...
          ]
        }
      ]
    }

You can add an `"image_src"` key to an entry of `"dialogue"` to append an image to a chat message. The value of the key should be a serialized image, starting with a string such `data:image/jpeg;base64,`.

For onboarding tasks (tasks used to filter workers, see below for more details) you must additionally set a `correct_answer` field:

    {
      "is_onboarding": true,
      "speakers_to_eval": ["first_modelname", "second_modelname"],
      "correct_answer": "correct_modelname",
      "dialogue_dicts": [
        # as above
      ]
    }

Note that we assume that "dialogue" consists of strictly alternating turns (e.g. speakers a, b, a, b, a...). Additionally, `speakers_to_eval` must be in the same order as the dialogue_dicts. See `pairings.jsonl` for examples of the format required.

## Question phrasing

In our paper, we address the problem of wording the questions and binary choices in order to elicit the highest signal responses. The default question and choices correspond to our highest signal 'engagingness' phrasing, but it's very easy to customize this by changing `eval_question`, `s1_choice`, and `s2_choice` in `conf/example.yaml`. The special strings `<Speaker 1>` and `<Speaker 2>` are replaced when showing these questions to the user, and the Speaker's utterances in each conversation will be colored to identify that Speaker.


## Onboarding tasks

As discussed in the paper, we found that we had better annotation quality if we screened Turkers with an 'onboarding' comparison, consisting of a weak baseline conversation and a human-human conversation. Our code is set up so that this is optional.

By default, `block_on_onboarding_fail` in `conf/example.yaml` is set to `true`, which means that workers who fail onboarding will be soft-blocked. In other words, they won't be able to see or complete any more HITs from you, but won't receive any notification that they've been blocked. The Mechanical Turk qualification name used to soft block must be set with `block_qualification`.

By setting `onboarding_threshold`, you can also adjust the minimum proportion of onboarding tasks (if you have multiple) that must be answered correctly to pass onboarding.


## YAML and CLI arguments

A comprehensive list of settings specific to ACUTE-Eval can be found in `AcuteEvalBlueprintArgs` in [`acute_eval_blueprint.py`](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/acute_eval/acute_eval_blueprint.py). For examples of how these arguments can be set in practice, see [`conf/example.yaml`](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/acute_eval/conf/example.yaml). For instance, `additional_task_description` gives additional text to show in the left-hand pane of the chat window.

# Fast ACUTE

**NOTE**: this code is a nearly feature-complete version of the code in [`parlai.mturk.tasks.acute_eval`](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/acute_eval), which will be deprecated soon. The only missing features in this version are the ability to run ACUTE-Evals on ParlAI tasks (datasets), as well as minor differences with rendering conversations in HTML using the analysis script. Use the old version of this task if those features are needed.

The scripts in this directory will allow you to run all the steps of [ACUTE-Eval](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing/tasks/acute_eval) with one simple command. Two types of Fast ACUTE can be run:
1. The base version (`run.py`), which includes having models chat with each other (known as "self-chats")
1. A variant that skips self-chats (`run_no_self_chat.py`)

Both types are discussed below.

## How to run Fast ACUTE if you need to produce model self-chats

### 1. Choose the self-chat task

First, determine which ParlAI task you will use to run model self-chat on. This task must have been set up for self-chat, i.e. it must have the appropriate worlds (typically called with the `parlai self_chat` command) used for conducting self-chat.

### 2. Create a file that specifies model configurations

Create a JSON file of the ParlAI parameters used for running self-chat on all models: see `task_config/model_config.json` for an example file. The parameters are any that you would need to specify to run self-chat on the command line.

### 3. Define settings for running onboarding

Create a JSON file of onboarding settings used to make sure that crowdsourcing workers perform necessary quality checks. See `task_config/onboarding.json` for an example file, and see the [ACUTE-Eval README](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/acute_eval/README.md) for more details.

### 4. Run Fast ACUTEs

Now that you've set up everything, launch Fast ACUTEs in the sandbox with a command like the following:
```
python parlai/crowdsourcing/tasks/fast_acute/run.py \
mephisto.blueprint.config_path=${PATH_TO_MODEL_CONFIG_JSON} \
mephisto.blueprint.models=\'model1,model2,model3\' \
mephisto.blueprint.num_self_chats=100 \
mephisto.blueprint.root_dir=${DIR_TO_SAVE_IN} \
mephisto.blueprint.onboarding_path=${PATH_TO_ONBOARDING_JSON} \
mephisto.blueprint.task=${SELF_CHAT_TASK}
```

You can also specify running Fast ACUTEs between only specific model pairs, with a syntax like `mephisto.blueprint.model_pairs=model1:model2`. In this case, the `mephisto.blueprint.models` flag is not used.

When you are ready to run a **live** ACUTE-Eval, add `mephisto.provider.requester_name=${REQUESTER_NAME} mephisto/architect=heroku` to this command, where `${REQUESTER_NAME}` is the MTurk requester name that you specified when setting up Mephisto.

Fast ACUTE operates in three phases:

#### 4a. Self-chat

First, the script attempts to run self-chat with all models; each models' self-chats are saved in a path in `${ROOT_DIR}/self_chats/` that is unique according to the model and self-chat task. This saves the trouble of re-running self-chat if a corresponding self-chat file already exists.

#### 4b. ACUTE-Eval

The script will then prepare each conversation-pairs file and save it in `${ROOT_DIR}/pairings_files/`, with a unique string according to which two self-chat files were used to create it. It will then run ACUTE-Eval with appropriate arguments.

#### 4c. Analysis

After finishing ACUTE-Eval, the script will analyze the results, and save files with information such as the win/loss rate and significance scores. Tables of results are saved to `${ROOT_DIR}/acute_results/<date>/`.

Generated result files include the following:
1. A CSV file of the win rates of all model pairs, as is typically shown when displaying ACUTE-Eval results in papers. These can be viewed by running a command like `cat acute_eval_<timestamp>.grid.csv | column -t -s, | less -S`.
2. A CSV file of the statistical significances of results, given by the *p*-values of the win rates of model pairs. View these with `cat acute_eval_<timestamp>.significance.csv | column -t -s, | less -S`.
3. HTML files of nicely visualized conversations.

**NOTE**: Analysis can be run on its own by calling `analysis.py`, specifying the ACUTE-Eval `run_id` and the `root_dir` that you used when running Fast ACUTE:
```
python parlai/crowdsourcing/tasks/fast_acute/analysis.py \
--root-dir ${FAST_ACUTE_ROOT_DIR} \
--run-id ${RUN_ID}
```
Use `--outdir` to save analysis results in a custom folder.


## How to run Fast ACUTE if you already have model self-chats

### 1. Create a file that specifies model configurations

Save a JSON file containing paths to all model self-chat files. The file should have the following structure:
```
{
    "model1": {
        "log_path": "/path/to/model1/selfchats.jsonl",
        "is_selfchat": true
    },
    "model2": {
        "log_path": "/path/to/model2/selfchats.jsonl",
        "is_selfchat": true
    }
}
```

See `task_config/self_chats/` for examples of what these self-chat files should look like.

### 2. Build ACUTE-Eval pairs and run

Launch Fast ACUTEs with a command like the following:
```
python parlai/crowdsourcing/tasks/fast_acute/run_no_self_chat.py \
mephisto.blueprint.config_path=${PATH_TO_MODEL_SELFCHAT_JSON} \
mephisto.blueprint.models=\'model1,model2,model3\' \
mephisto.blueprint.num_self_chats=100 \
mephisto.blueprint.root_dir=${DIR_TO_SAVE_IN} \
mephisto.blueprint.onboarding_path=${PATH_TO_ONBOARDING_JSON}
```
Here, the `mephisto.blueprint.task` parameter is not needed because we are not running self-chats.
