# Human Evaluation of Conversations is an Open Problem: comparing the sensitivity of various methods for evaluating dialogue agents

Eric Michael Smith, Orion Hsu, Rebecca Qian, Stephen Roller, Y-Lan Boureau, Jason Weston

## Abstract

At the heart of improving conversational AI is the open problem of how to evaluate conversations. Issues with automatic metrics are well known ([Liu et al., 2016](https://arxiv.org/abs/1603.08023)), with human evaluations still considered the gold standard. Unfortunately, how to perform human evaluations is also an open problem: differing data collection methods have varying levels of human agreement and statistical sensitivity, resulting in differing amounts of human annotation hours and labor costs. In this work we compare five different crowdworker-based human evaluation methods and find that different methods are best depending on the types of models compared, with no clear winner across the board. While this highlights the open problems in the area, our analysis leads to advice of when to use which one, and possible future directions.

## Paper

[Link](https://arxiv.org/abs/2201.04723)

## Performing evaluations

### Pairwise per-turn evaluations (PW-Turn)

See the [PW-Turn README](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/tasks/pairwise_per_turn_eval/README.md) for running and analyzing pairwise per-turn evaluations.

### Pairwise per-dialogue evaluations (PW-Dialog)

See the "Fast-ACUTE" section of the [Acute-Eval README](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/tasks/acute_eval/README.md) for running and analyzing pairwise per-dialogue evaluations, referred to as PW-Dialog evaluations in this paper. Specify the evaluation metric with `mephisto.blueprint.acute_eval_type={engaging,human,interesting}`.

To specify the models to compare, pass in the arguments
```
mephisto.blueprint.config_path=${CONFIG_PATH} \
mephisto.blueprint.model_pairs=\'${MODEL_NAME_1}:${MODEL_NAME_2}\'
```
where `${CONFIG_PATH}` points to a JSON file that defines the configurations of the two models `${MODEL_NAME_1}` and `${MODEL_NAME_2}`. (See the [Acute-Eval `task_config/` folder](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/acute_eval/task_config) for examples of such JSON files for self-chats and for existing human+model chat logs.) For PW-Dialog self-chats, the following settings were used in the JSON config file:
```
{
    "model_file": MODEL_FILE_STRING,
    "model": "transformer/generator",
    "beam_min_length": 20,  # Or 0 for BlenderBot3B-M0
    "batchsize": 1,
    "skip_generation": false,
    "interactive_mode": false,
    "beam_size": 10,
    "inference": "beam",
    "beam_block_ngram": 3,
    "beam_context_block_ngram": 3,
    "beam_block_full_context": false
},
```
`MODEL_FILE_STRING` was set to the following value for each model evaluated:
- **BlenderBot3B** and **BlenderBot3B-M0**: `"zoo:blender/blender_3B/model"`
- **BlenderBot90M**: `"zoo:blender/blender_90M/model"`
- **Reddit3B**: `"zoo:blender/reddit_3B/model"`

### Single-model evaluations (SM-Turn and SM-Dialog)

See the [SM-Turn/SM-Dialog README](https://github.com/facebookresearch/ParlAI/blob/main/parlai/crowdsourcing/projects/humaneval/single_model_eval/README.md) for running and analyzing single-model evaluations.

## Citation

If you use the dataset or models in your own work, please cite with the
following BibTex entry:
    
    @misc{smith2022human,
      title={Human Evaluation of Conversations is an Open Problem: comparing the sensitivity of various methods for evaluating dialogue agents}, 
      author={Eric Michael Smith and Orion Hsu and Rebecca Qian and Stephen Roller and Y-Lan Boureau and Jason Weston},
      year={2022},
      eprint={2201.04723},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
