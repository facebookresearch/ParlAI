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
Once you have [installed ParlAI](https://github.com/facebookresearch/ParlAI/#installing-parlai),
follow the instructions below.

The `run.py` script is designed to allow you to run this entire task from command line with an invocation like

    python parlai/mturk/tasks/acute_eval/run.py --pairings-filepath parlai/mturk/tasks/acute_eval/example/pairings.jsonl

However, you can also choose to set the command line arguments with a script. You can find an example run script in `example_script.py`


## Formatting conversation data

This task code assumes that you've parsed and saved your collected conversations in a simple .jsonl format. The path to this file should be passed in as `--pairings-filepath`.

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

For onboarding tasks - tasks used to filter workers, see below for more details - you must additionally set a `correct_answer` field:

    {
      "is_onboarding": true,
      "speakers_to_eval": ["first_modelname", "second_modelname"],
      "correct_answer": "correct_modelname",
      "dialogue_dicts": [
        # as above
      ]
    }


Note that we assume "dialogue" consists of strictly alternating turns (e.g. speakers a, b, a, b, a...). Additionally, `speakers_to_eval` must be in the same order as the dialogue_dicts. See `example/pairings.jsonl` for examples of the format required.

## Question phrasing

In our paper, we address the problem of wording the questions and binary choices in order to elicit the highest signal responses. The default question and choices correspond to our highest signal 'engagingness' phrasing, but it's very easy to customize this. Simply set `args['question'], args['s1_choice'], args['s2_choice']`. The special strings `<Speaker 1>` and `<Speaker 2>` are replaced in the choices to be colored in with the color corresponding with the speaker.


## Onboarding tasks

As discussed in the paper, we found that we had better annotation quality if we screened turkers with an 'onboarding' comparison, consisting of a weak baseline conversation and a human-human conversation. Our code is set up so that this is optional.

By default `--block-on-onboarding-fail` is set to `True`, which means that workers who fail onboarding will be soft blocked - they won't be able to see or complete any more hits from you but won't receive any notification that they've been blocked. The Mechanical Turk qualification name used to soft block must be set with `--block-qualification`.

By setting `--onboarding-threshold`, you can also adjust the minimum proportion of onboarding tasks (if you have multiple) which must be answered correctly to pass onboarding.


## Other settings

### Task configuration on MTurk

The title, description, and keywords of the task as shown on MTurk default to values in DEFAULT_TASK_CONFIG shown at the top of `run.py`. If you would like to change any of these values, pass a dict to the as the `--task-config` argument with the following keys:

    """A short and descriptive title about the kind of task the HIT contains.
    On the Amazon Mechanical Turk web site, the HIT title appears in search results,
    and everywhere the HIT is mentioned.
    """
    task_config['hit_title'] = 'Which Conversational Partner is Better?'


    """A description includes detailed information about the kind of task the HIT contains.
    On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
    view of search results, and in the HIT and assignment screens.
    """
    task_config['hit_description'] = 'Evaluate quality of conversations through comparison.'


    """One or more words or phrases that describe the HIT, separated by commas.
    On MTurk website, these words are used in searches to find HITs.
    """
    task_config['hit_keywords'] = 'chat,evaluation,comparison,conversation'



### CLI arguments

A comprehensive list of settings specific to ACUTE-Eval can be found in `add_args()` in `run.py`. ParlAI MTurk arguments can be found in `~/ParlAI/parlai/core/params.py` under `add_mturk_args()`. For the arguments most likely to be useful for running ACUTE-Eval, see `example_script.py`:

** **

# ACUTE-Eval Analysis

Once you have successfully completed a run of ACUTE-Eval, it's time to analyze your results. We provide a handy script that does everything for you!

## Matchup Grid & Signficance
To generate a multitude of important analysis files, simply run the following command:

    python parlai/mturk/tasks/acute_eval/analysis.py -id <run_id> --is-sandbox <True/False>

This will generate the following two tables for your perusal:

1. A **winner/loser grid** for each model pairing in the ACUTE run, indicating the win/loss percentage for each model pairing.
2. A **matchup table**, where each row is a model comparison, and which includes the statistical significance of the wins/losses.

The script automatically saves these two dataframes as `.csv` files in `ParlAI/data/acute_eval/<run-id>-results/`. To change this, simply set the `--outdir` accordingly.

## Visualize Conversations

To visualize what conversations were chosen, and for what reasons, you can run the same command as above with the following additional parameter:

    python parlai/mturk/tasks/acute_eval/analysis.py -id <run_id> --is-sandbox <True/False> \
    --pairings-filepath </path/to/pairs/file>

Where `</path/to/pairs/file>` is your pairings file from the ACUTE Eval run. Running the command above will yield two additional HTML files saved to the same `--outdir` directory:

1. **all.html** - List of all conversations, indicating which was chosen as the winner by a turker.
2. **reason.html** - List of all conversations where reasons are provided by the turkers for why they chose a winner.

# Fast-ACUTE

We provide an all-in-one script to run ACUTE-Eval in the smoothest experience possible.

The script combines three major steps of ACUTE-Eval into one simple command:

1. Generation (or compilation) of chat logs for given models;
2. Execution of ACUTE-Eval
3. Analysis of ACUTE-Eval results.

## Setup Steps

### 1. Determine What You Will Be Evaluating; Populate Config.

This is an important step - do you have conversation logs between a model and a human? Would you like evaluate model self-chat? Do you want to evaluate dataset logs?

Each of these options involves _slightly_ different preparation. However, each involves specifying a config.

In the `configs.py` file in this directory, you will find a `CONFIG` dictionary that maps a _unique_ identifier to appropriate configuration arguments; these arguments differ depending on what you will be evaluating.

*NOTE*: the `CONFIG` is _append only_, and all configs must have a *unique* identifier.

I will enumerate a few of these options below.

#### Model self-chat

If you would like to evaluate a model chatting to itself, you simply specify the appropriate model parameters in the config. The parameters are any that you would need to specify on the command line, and include things like the model-file, fixed candidates file, etc. You can see an example in the `example_model` config.

#### JSONL Logs

If you have logs in the appropriate JSONL format, as would be generated by the self-chat script, then all you need to specify is the `log_path`. You can see an example in the `example_model_log` config.

The appropriate JSONL format is one that can be read by ParlAI's [Conversations](https://github.com/facebookresearch/ParlAI/blob/master/parlai/utils/conversations.py) class. Note that the identifier in the config should match **EXACTLY** the `id` of the model in the conversations.

#### Dataset

If you'd like to evaluate examples from a dataset available in ParlAI directly, simply specify the `task` in the config. You can see an example in the `example_dataset` config.

### 1b. (Optional) Determine the Self-Chat Task You Will Use

If you will be evaluating models via self-chat, you will need to determine the self-chat task you will use to help generate the self-chats. This is not so much any work on your part other than identfying a task that is setup for self-chat, i.e., a task that has the appropriate worlds used for conducting self-chat with the models. This is not strictly necessary, but you may want to introduce context, e.g. as in `convai2` or `blended_skill_talk`.

### 2. Run `fast_eval.py`

Now that you've setup everything, all you need to do is run one of the following commands.

If you want to compare a set of models in round-robin fashion, you would run:

    python parlai/mturk/tasks/acute_eval/fast_eval.py --ids <comma-separated list of config identifiers>

If you want multiple model comparisons, but do not want to compare ALL models with eachother, you would run:

    python parlai/mturk/tasks/acute_eval/fast_eval.py --id-pairs <comma-separated, colon-delimited list of config identifiers>

The ids specified for each of those flags corresponds to the entry in the `CONFIG`.

If you are running self-chat, you can optionally specify a seed task to use for self-chat with `-t <self_chat_task>`.

A few examples are as follows:

    python parlai/mturk/tasks/acute_eval/fast_eval.py --ids  example_model_1,example_model_2,example_model_log,example_dataset -t blended_skill_talk

    python parlai/mturk/tasks/acute_eval/fast_eval.py --id-pairs  example_model_1:example_model_2,example_model_1:example_model_log,example_dataset:example_model_2 -t blended_skill_talk

When you are ready to run a **LIVE** ACUTE-Eval, please specify `--live-acute true`.

#### Onboarding

The default onboaring dialogue pair is in `example/onboarding.json`. We recommend you use a different onboarding example as the one provided is quite easy.

To use a custom onboarding path, specify the `--onboarding-path` when running `fast_eval.py`. The onboarding file should be a jsonl file, where each line is a json dict consisting of a pair of dialogues to evaluate, and where `is_onboarding` is set to True.

## Script Execution

The script operates in three phases:

#### Phase 1: Compile Chat Logs

The script will first compile the chat logs for each identifier specified on the command line.

For `model`s, the script will run self-chat (if a self-chat log does not already exist); for `log`s, the script will simply load the log from disk; and for `task`s, the script will convert the task into the appropriate format.

Self-chats are saved to `PARLAI_PATH/data/acute_evals/self_chats/`

### Phase 2: ACUTE-Eval

The script will then prepare the conversation-pairs file (and save to `PARLAI_PATH/data/pairings_files/`, unique according to which chat files were used to create it) and run ACUTE-Eval with appropriate arguments.

Upon subsequent runs with the same configuration of `--ids` or `--id-pairs`, you will have the option to re-use a pairings file or to regenerate it.

### Phase 3: Analysis

After finishing ACUTE-Eval, the script will analyze and save relevat results to `PARLAI_PATH/data/acute_evals/acute_results/<date>/<pairings_file>/`

4 Results will be generated:

1. A csv file of significance result, which shows the win rates of model pairs with p value
2. A csv file of grid result, where the model comparisons and win rates are laid out in a nice grid (as seen in the ACUTE-Eval paper).
3. A html file of nicely visualized conversations with reason annotated only
4. A html file of ALL nicely visualized conversations

**NOTE** the the `analysis.py` file can be run by itself as long as you specify the ACUTE-Eval `run_id`, whether a sandbox run, and whether it is qfunction eval.

