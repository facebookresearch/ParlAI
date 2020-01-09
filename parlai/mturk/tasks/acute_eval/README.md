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
    eprint={1909.03087},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
  }

# Code Instructions
Once you have [installed ParlAI](https://github.com/facebookresearch/ParlAI/#installing-parlai),
follow the instructions below.

The `run.py` script is designed to allow you to run this entire task from command line with an invocation like

    python parlai/mturk/tasks/acute_eval/run.py --task_folder parlai/mturk/tasks/acute_eval/example/

However, you can also choose to set the command line arguments with a script. You can find an example run script in `example_script.py`


## Formatting conversation data

This task code assumes that you've parsed and saved your collected conversations in a simple .jsonl format within your `args['task_folder']` folder as `pairings.jsonl`. You can change the name of the file by setting `--pairings_file`.

This is a template of the expected format with the minimal expected fields:

    {
      "is_onboarding": false,
      "speakers_to_eval": ["first_modelname", "second_modelname"],
      "dialogue_dicts": [
        {
          "speakers": ["first_modelname", "other_speaker"],
          "conversation_id": "id",
          "dialogue": [
            {"speaker": "model1", "text": "Hi"},
            {"speaker": "other_speaker", "text": "Hi back"},
            ...
          ]
        },
        {
          "speakers": ["other_speaker", "second_modelname"],
          "conversation_id": "id",
          "dialogue": [
            {"speaker": "model1", "text": "Hi"},
            {"speaker": "other_speaker", "text": "Hi back"},
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


Note that we assume "dialogue" consists of strictly alternating turns (e.g. speakers a, b, a, b, a...). Additionally, `speakers_to_eval` must be in the same order as the dialogue_dicts. See `example/example_pairings.jsonl` for examples of the format required.

## Question phrasing

In our paper, we address the problem of wording the questions and binary choices in order to elicit the highest signal responses. The default question and choices correspond to our highest signal 'engagingness' phrasing, but it's very easy to customize this. Simply set `args['question'], args['s1_choice'], args['s2_choice']`. The special strings `<Speaker 1>` and `<Speaker 2>` are replaced in the choices to be colored in with the color corresponding with the speaker.


## Onboarding tasks

As discussed in the paper, we found that we had better annotation quality if we screened turkers with an 'onboarding' comparison, consisting of a weak baseline conversation and a human-human conversation. Our code is set up so that this is optional.

By default `args['block_on_onboarding']` is set to `True`, which means that workers who fail onboarding will be soft blocked - they won't be able to see or complete any more hits from you but won't receive any notification that they've been blocked. The Mechanical Turk qualification name used to soft block must be set with `args['block_qualification']`.

By setting `args['onboarding_threshold']`, you can also adjust the minimum proportion of onboarding tasks (if you have multiple) which must be answered correctly to pass onboarding.


## Other settings

### Task configuration on MTurk

The title, description, and keywords of the task as shown on MTurk default to values in DEFAULT_TASK_CONFIG shown at the top of `run.py`. If you would like to change any of these values, create a file called `task_config.py` in your `opt['task_folder']`. See the `example/task_config.py` for more guidance.


### CLI arguments

A comprehensive list of settings specific to ACUTE-Eval can be found in `add_args()` in `run.py`. ParlAI MTurk arguments can be found in `~/ParlAI/parlai/core/params.py` under `add_mturk_args()`. For the arguments most likely to be useful for running ACUTE-Eval, see `example_script.py`:



## Retrieving and analyzing results

Coming soon.


## Creating the pairings file

Coming soon.


** **
