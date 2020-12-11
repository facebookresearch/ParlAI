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
