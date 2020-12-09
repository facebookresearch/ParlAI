# Fast ACUTE

The scripts in this directory will allow you to run all the steps of [ACUTE-Eval](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing/tasks/acute_eval) with one simple command.

## Setup

### 1. Add the model to the model config

# *** TODO: rewrite all this! ***

In the `model_configs.py` file, you will find a `CONFIG` dictionary that maps a _unique_ model identifier to the appropriate configuration arguments. The parameters are any that you would need to specify on the command line, and include things like the model-file, fixed candidates file, etc.

**NOTE**: `CONFIG` is _append-only_, and all models must have a *unique* identifier.

### 2. Determine the self-chat task that you will use

This only requires you to identify a task that has been set up for self-chat, i.e. a task that has the appropriate worlds used for conducting self-chat with the models.

### 3. Run `fast_eval.py`

Now that you've set up everything, all you need to do is run the following command:
```
python ${TODO FIX ME BEFORE PR}/fast_acute/fast_eval.py \
mephisto.blueprint.model_pairs=\'<comma-separated list of model pairs>\' \
mephisto.blueprint.task=<self-chat task>
```
where `mephisto.blueprint.model_pairs` specifies a comma-separated list of model pairs, with each model taken from the `CONFIG`, and `mephisto.blueprint.task` specifies a task that allows for self-chat.

An example command is as follows:
```
python ${TODO FIX ME BEFORE PR}/fast_acute/fast_eval.py \
mephisto.blueprint.model_pairs=\'generative32.6B_bst_selfchat:generative2.7B_bst_0331_selfchat\' \
mephisto.blueprint.task=blended_skill_talk
```

When you are ready to run a **live** ACUTE-Eval, add `mephisto.provider.requester_name=${REQUESTER_NAME} mephisto/architect=heroku` to this command, where `${REQUESTER_NAME}` is the MTurk requester name that you specified when setting up Mephisto.

## So what does this script do?

The script operates in three phases:

##### Phase 1: Self-chat

[TODO I think a lot of this logic has changed] First, the script attempts to run self-chat with both models; self-chats are logged to `${TODO FIX ME BEFORE PR}` by default and are unique according to the model and self-chat task. This saves the trouble of re-running self-chat if a corresponding self-chat file already exists. The default root save folder, `${TODO FIX ME BEFORE PR}`, can be changed by modifying the `mephisto.blueprint.root_dir` parameter.

##### Phase 2: ACUTE-Eval

The script will then prepare the conversation-pairs file and save it to `${TODO FIX ME BEFORE PR}` by default, with a unique string according to which two self-chat files were used to create it. It will then run ACUTE-Eval with appropriate arguments.

##### Phase 3: Analysis

After finishing ACUTE-Eval, the script will analyze the results and print out the win/loss table with significance scores. Tables of results are saved to `${TODO FIX ME BEFORE PR}/acute_results/<date>/` by default.

Generated result files include the following:
1. A CSV file of the win rates of all model pairs, as is typically shown when displaying ACUTE-Eval results in papers. These can be viewed by running a command like `cat acute_eval_1601988419.grid.csv | column -t -s, | less -S`.
2. A CSV file of the statistical significances of results, given by the *p*-values of the win rates of model pairs. View these with `cat acute_eval_1601988419.significance.csv | column -t -s, | less -S`.
3. HTML files of nicely visualized conversations. To view these, `scp` them to your local machine.

**NOTE**: Analysis can be run on its own by running `analysis.py`, as long as you specify the ACUTE-Eval `run_id` and whether the run is a Q-function evaluation or not.

## How to run Fast ACUTE for the Q-function project

### Build ACUTE-Eval pairs and run

Run a command like `python ${TODO FIX ME BEFORE PR}/fast_acute_q_function/fast_eval.py mephisto.blueprint.model_pairs=\'generative32.6B_bst_selfchat:generative2.7B_bst_0331_selfchat\'`

By default, the pairing file is saved at `${TODO FIX ME BEFORE PR}` with a hashed name. The default root save folder, `${TODO FIX ME BEFORE PR}`, can be changed by modifying the `mephisto.blueprint.root_dir` parameter.

### Analyze results

Your ACUTE-Eval results should be analyzed automatically at the end of HIT collection. To launch analysis manually, run a command like `python ${TODO FIX ME BEFORE PR}/fast_acute/analysis.py --run-id acute_eval_1584577685 --is-qfunction-eval True`. See the README for the base Fast ACUTE task {TODO add link} for more details.