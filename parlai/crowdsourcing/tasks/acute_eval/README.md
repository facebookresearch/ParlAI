# ACUTE-Eval

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

Once you have installed [ParlAI](https://github.com/facebookresearch/ParlAI/#installing-parlai) and [Mephisto](https://github.com/facebookresearch/mephisto/blob/main/docs/quickstart.md), follow the instructions below.

The `run.py` script is designed to allow you to run this entire task from command line with an invocation like

    python parlai/crowdsourcing/tasks/acute_eval/run.py \
    mephisto.blueprint.pairings_filepath=${REPO_FOLDER}/parlai/crowdsourcing/tasks/acute_eval/task_config/pairings.jsonl

Make a note of the run ID printed to the command line upon running, because this will be useful for analyzing results later on.

**NOTE**: See [parlai/crowdsourcing/README.md](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/README.md) for general tips on running `parlai.crowdsourcing` tasks, such as how to specify your own YAML file of configuration settings, how to run tasks live, how to set parameters on the command line, etc.


## Formatting conversation data

This task code assumes that you've parsed and saved your collected dialogues in a simple .jsonl format. The path to this file should be passed in as `mephisto.blueprint.pairings_filepath=${PATH_TO_FILE}`.

Note that this format is slightly different than that of chat logs from `eval_model` scripts. See information on Fast-ACUTE below for scripts to compile chat logs of the Conversations format into the ACUTE format.

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

Note that we assume that "dialogue" consists of strictly alternating turns (e.g. speakers a, b, a, b, a...). Speakers that we would like to evaluate in the dialogues of `dialogue_dicts` should appear in the same order as `speakers_to_eval`. (Consequently, the number of dialogues in `dialogues_dicts` will be the same as the number of speakers in `speakers_to_eval`.) See `task_config/pairings.jsonl` for examples of the format required.

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

## Question phrasing

In our paper, we address the problem of wording the questions and binary choices in order to elicit the highest signal responses. The default question and choices correspond to our highest signal 'engagingness' phrasing, but it's very easy to customize this by changing `eval_question`, `s1_choice`, and `s2_choice` in `conf/example.yaml`. The special strings `<Speaker 1>` and `<Speaker 2>` are replaced when showing these questions to the user, and the Speaker's utterances in each conversation will be colored to identify that Speaker.


## Onboarding tasks

As discussed in the paper, we found that we had better annotation quality if we screened Turkers with an 'onboarding' comparison, consisting of a weak baseline conversation and a human-human conversation. Our code is set up so that this is optional.

By default, `block_on_onboarding_fail` in `conf/example.yaml` is set to `true`, which means that workers who fail onboarding will be soft-blocked. In other words, they won't be able to see or complete any more HITs from you, but won't receive any notification that they've been blocked. The Mechanical Turk qualification name used to soft block must be set with `mephisto.blueprint.block_qualification`.

By setting `mephisto.blueprint.onboarding_threshold`, you can also adjust the minimum fraction of onboarding tasks (if you have multiple) that must be answered correctly to pass onboarding.


## YAML and CLI arguments

A comprehensive list of settings specific to ACUTE-Eval can be found in `AcuteEvalBlueprintArgs` in [`acute_eval_blueprint.py`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/tasks/acute_eval/acute_eval_blueprint.py). For examples of how these arguments can be set in practice, see [`conf/example.yaml`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/tasks/acute_eval/conf/example.yaml). For instance, `additional_task_description` gives additional text to show in the left-hand pane of the chat window.


# ACUTE-Eval analysis

Once you have successfully completed a run of ACUTE-Eval, it's time to analyze your results. We provide a handy script that does everything for you!

To analyze results, run the following command, specifying the ACUTE-Eval run ID (printed to the command line upon launching the ACUTE-Evals) and the path to the ACUTE-Eval pairings file that you specified with `mephisto.blueprint.pairings_filepath`:
```
python parlai/crowdsourcing/tasks/acute_eval/analysis.py \
--run-ids ${RUN_ID} \
--pairings-filepath ${PATH_TO_PAIRINGS_FILE} \
--outdir ${OUTPUT_FOLDER}
```
For analyzing results from a Fast ACUTE run (see below), use the `--root-dir` flag to specify the Fast ACUTE root directory (`mephisto.blueprint.root_dir`) instead of specifying the `--pairings-filepath` and `--outdir` flags.

The script will analyze the results and save files with information such as the win/loss rate and significance scores.

Generated result files include the following:
1. A CSV file of the win rates of all model pairs, as is typically shown when displaying ACUTE-Eval results in papers. These can be viewed by running a command like `cat acute_eval_<timestamp>.grid.csv | column -t -s, | less -S`.
2. A CSV file of the statistical significances of results, given by the *p*-values of the win rates of model pairs. View these with `cat acute_eval_<timestamp>.significance.csv | column -t -s, | less -S`.
3. HTML files of nicely visualized conversations.


# Fast-ACUTE

We provide an all-in-one script to run ACUTE-Eval in the smoothest experience possible.

The script combines three major steps of ACUTE-Eval into one simple command:

1. Generation (or compilation) of chat logs for given models into the ACUTE format
2. Execution of ACUTE-Eval
3. Analysis of ACUTE-Eval results.

**NOTE**: this code was adapted from the code formerly in [`parlai/mturk/tasks/acute_eval/`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/mturk), which has now been deprecated. A few minor options existed in the analysis script of that old version which have not been ported to the original:

- Specifying the minimum dialogue length to be counted as valid for analysis
- Specifying the maximum number of matchups per model pair visualized in HTML
- Optionally including a checkbox column for annotating the convo pairs in HTML

If you would like to make use of these old features, please open a [ParlAI issue](https://github.com/facebookresearch/ParlAI/issues) or restore the original version by switching to the `acute_eval` tag of ParlAI:

```bash
$ git checkout acute_eval
```

## Setup

### 1. Determine what you will be evaluating and create a config file

This is an important step - do you have conversation logs between a model and a human? Would you like to evaluate model self-chat? Do you want to evaluate dataset logs?

Each of these options involves _slightly_ different preparation. However, each involves specifying a config file, which is specified on the command line by `mephisto.blueprint.config_path` when launching Fast ACUTEs.

In the `task_config/` folder, you will find several `model_config_*.json` files that map a _unique_ identifier to appropriate configuration arguments; these arguments differ depending on what you will be evaluating.

A few of these options are enumerated below.

#### Model self-chat

If you would like to evaluate a model chatting to itself, you simply specify the appropriate model parameters in the config. The parameters are any that you would need to specify on the command line, and include things like the model-file, fixed candidates file, etc. You can see an example in `task_config/model_config_self_chat.json`. You will need to determine the ParlAI task that you will use to help generate the self-chats: this must be a task that is set up for self-chat, i.e., a task that has the appropriate worlds for conducting self-chat with the models.

#### JSONL logs

If you have logs in the appropriate JSONL format, as would be generated by the self-chat script, then all you need to specify is the `log_path` and whether the logs are model self-chats. In the case of self-chats, you can also specify whether to evaluate the conversation turns of the first or second speaker (i.e. `speaker_idx=0` or `1`). You can see an example in `task_config/model_config_logs.json`.

The appropriate JSONL format is one that can be read by ParlAI's [Conversations](https://github.com/facebookresearch/ParlAI/blob/main/parlai/utils/conversations.py) class. Note that the identifier in the config should match **EXACTLY** the `id` of the model in the conversations.

#### Dataset

If you'd like to evaluate examples from a dataset available in ParlAI directly, simply specify the `task` in the config. You can see an example in `task_config/model_config_dataset.json`. Optionally, you may want to introduce context, e.g. as in `convai2` or `blended_skill_talk`: this can be specified in the config file with the `'prepended_context'` parameter.

### 2. Run `fast_eval.py`


Now that you've set up everything, launch Fast ACUTEs in the sandbox with a command like the following:
```
python parlai/crowdsourcing/tasks/acute_eval/fast_eval.py \
mephisto.blueprint.config_path=${PATH_TO_MODEL_CONFIG_JSON} \
mephisto.blueprint.models=\'model1,model2,model3\' \
mephisto.blueprint.num_self_chats=100 \
mephisto.blueprint.root_dir=${ROOT_SAVE_DIR}
```

You can also specify running Fast ACUTEs between only specific model pairs, with a syntax like `mephisto.blueprint.model_pairs=model1:model2`. In this case, the `mephisto.blueprint.models` flag is not used.

If you are running self-chat, you can optionally specify a seed task to use for self-chat with `mephisto.blueprint.task=${SELF_CHAT_TASK}`.

When you are ready to run a **live** ACUTE-Eval, add `mephisto.provider.requester_name=${REQUESTER_NAME} mephisto/architect=ec2 mephisto.architect.profile_name=mephisto-router-iam` to this command, where `${REQUESTER_NAME}` is the MTurk requester name that you specified when setting up Mephisto.

#### Onboarding

The default onboaring dialogue pair is in `task_config/onboarding.json`. We recommend you use a different onboarding example, because the one provided is quite easy.

To use a custom onboarding path, specify the `mephisto.blueprint.onboarding_path` when running `fast_eval.py`. The onboarding file should be a jsonl file, where each line is a json dict consisting of a pair of dialogues to evaluate, and where `is_onboarding` is set to True.

## Script Execution

The script operates in three phases:

### Phase 1: Compile Chat Logs

The script will first compile the chat logs for each model specified on the command line.

For each model, the code will read the appropriate settings for it in the specified configuration file. If `'log_path'` is specified in the settings, the script will simply load the log from disk; if `'model'` is specified, the script will run self-chat (if a self-chat log does not already exist); and if `'task'` is specified, the script will convert the task into the appropriate format.

Self-chats are saved to `${ROOT_SAVE_DIR}/self_chats/`.

### Phase 2: ACUTE-Eval

The script will then prepare each conversation-pairs file and save it in `${ROOT_SAVE_DIR}/pairings_files/`, with a unique string according to which two self-chat files were used to create it. It will then run ACUTE-Eval with appropriate arguments.

Upon subsequent runs with the same configuration of `mephisto.blueprint.models` or `mephisto.blueprint.model_pairs`, you will have the option to re-use a pairings file or to regenerate it.

### Phase 3: Analysis

After finishing ACUTE-Eval, the script will analyze and save relevant results to `${ROOT_SAVE_DIR}/acute_results/<date>/<pairings_file>/`.
