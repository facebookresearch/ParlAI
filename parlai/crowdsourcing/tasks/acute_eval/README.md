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
Once you have installed [ParlAI](https://github.com/facebookresearch/ParlAI/#installing-parlai) and [Mephisto](https://github.com/facebookresearch/mephisto/blob/master/docs/quickstart.md), follow the instructions below.

The `example_script.py` script is designed to allow you to run this entire task from command line with an invocation like

    python parlai/crowdsourcing/tasks/acute_eval/example_script.py \
    --pairings-filepath parlai/crowdsourcing/tasks/acute_eval/pairings.jsonl

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

In our paper, we address the problem of wording the questions and binary choices in order to elicit the highest signal responses. The default question and choices correspond to our highest signal 'engagingness' phrasing, but it's very easy to customize this by changing `extra_args['eval_question'], extra_args['s1_choice'], extra_args['s2_choice']` in `example_script.py`. The special strings `<Speaker 1>` and `<Speaker 2>` are replaced when showing these questions to the user, and the Speaker's utterances in each conversation will be colored to identify that Speaker.


## Onboarding tasks

As discussed in the paper, we found that we had better annotation quality if we screened Turkers with an 'onboarding' comparison, consisting of a weak baseline conversation and a human-human conversation. Our code is set up so that this is optional.

By default, `extra_args['block_on_onboarding_fail']` is set to `True`, which means that workers who fail onboarding will be soft-blocked. In other words, they won't be able to see or complete any more hits from you, but won't receive any notification that they've been blocked. The Mechanical Turk qualification name used to soft block must be set with `extra_args['block_qualification']`.

By setting `extra_args['onboarding_threshold']`, you can also adjust the minimum proportion of onboarding tasks (if you have multiple) that must be answered correctly to pass onboarding.


## Other settings

### Task configuration on MTurk

The title, description, and keywords of the task as shown on MTurk default to values in `ARG_STRING` in `example_script.py`. These values are used as follows:
- `--task-title`: A short and descriptive title about the kind of task that the HIT contains. On the Amazon Mechanical Turk web site, the HIT title appears in search results and everywhere that the HIT is mentioned.
- `--task-description`: Includes detailed information about the kind of task that the HIT contains. On the Amazon Mechanical Turk web site, the HIT description appears in the expanded view of search results, and in the HIT and assignment screens.
- `--task-tags`: One or more words or phrases that describe the HIT, separated by commas. On MTurk website, these words are used in searches to find HITs.
- `--additional-task-description`: Additional text to show in the left-hand pane of the chat window.


### CLI arguments

A comprehensive list of settings specific to ACUTE-Eval can be found in `add_args_to_group()` in `acute_eval_blueprint.py`. For the arguments most likely to be useful for running ACUTE-Eval, see `example_script.py`.
