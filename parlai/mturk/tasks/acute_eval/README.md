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

You can find an example run script in `example_script.py` - opening this and following along the instructions with it will be easiest.


## Formatting conversation data

This task code assumes that you've parsed and saved your collected conversations in a simple .jsonl format within a folder specified by `args['dialogs_path']`. This is a template of the format with the minimal expected fields:

    {
      "speakers": ["model", "human_evaluator"],
      "conversation_id": "id",
      "dialog": [
        {"speaker": "model", "text": "Hi"},
        {"speaker": "human_evaluator", "text": "Hi back"},
        ...
      ]
    }

Note that we assume "dialog" consists of strictly alternating turns (e.g. a, b, a, b, a...). See `example/model*.jsonl` for examples of the format required.

## Question phrasing

In our paper, we address the problem of wording the questions and binary choices in order to elicit the highest signal responses. The default question and choices correspond to our highest signal 'engagingness' phrasing, but it's very easy to customize this. Simply set `args['question'], args['s1_choice'], args['s2_choice']`. The special strings `<Speaker 1>` and `<Speaker 2>` are replaced in the choices to be colored in with the color corresponding with the speaker.


## Onboarding tasks

As discussed in the paper, we found that we had better annotation quality if we screened turkers with an 'onboarding' comparison, consisting of a weak baseline conversation and a human-human conversation. Our code is set up so that this is optional.

To use onboarding tasks, set `args['onboarding_tasks']` as shown in `example_script.py` to have this format:

    [
      (wrong_choice_id, right_choice_id, matchup_name),
      ...
    ]

where `right_choice_id` is the id of the conversation which should be chosen by the turker to successfully pass onboarding filtering.

By default `args['block_on_onboarding']` is set to `True`, which means that workers who fail onboarding will be soft blocked - they won't be able to see or complete any more hits from you but won't receive any notification that they've been blocked. The Mechanical Turk qualification name used to soft block must be set with `args['block_qualification']`.

By setting `args['onboarding_threshold']`, you can also adjust the minimum proportion of onboarding tasks (if you have multiple) which must be answered correctly to pass onboarding.


## Model comparisons

To compare models, you can either choose specific pairs of conversation to show workers, or you can have `run.py` draw pairs of conversations at random.


## Other settings

A comprehensive list of settings specific to ACUTE-Eval can be found in `add_args()` in `run.py`. ParlAI MTurk arguments can be found in `~/ParlAI/parlai/core/params.py` under `add_mturk_args()`. For the arguments most likely to be useful for running ACUTE-Eval, see `example_script.py`:




## Retrieving and analyzing results

Coming soon.




** **
