# Recipes for Safety in Open-domain Chatbots

Jing Xu, Da Ju, Margaret Li, Y-Lan Boureau, Jason Weston, Emily Dinan

## Abstract

Models trained on large unlabeled corpora of human interactions will learn patterns and mimic behaviors therein, which
include offensive or otherwise toxic behavior and unwanted biases. We investigate a variety of methods to mitigate these issues in the context of open-domain generative dialogue models. We introduce a new human-and-model-in-the-loop framework for both training safer models and for evaluating them, as well as a novel method to distill safety considerations inside generative models without the use of an external classifier at deployment time. We conduct experiments comparing these methods and find our new techniques are (i) safer than existing models as measured by automatic and human evaluations while (ii) maintaining usability metrics such as engagingness relative to the state of the art. We then discuss the limitations of this work by analyzing failure cases of our models.

## Paper

[Link](TBD)


## Data

We release the Bot-Adversarial Dialogue task at `parlai/tasks/bot_adversarial_dialogue`. To view the data, run:

```
parlai display_data -t bot_adversarial_dialogue
```

To view the data used for the fixed test set, run:

```
parlai display_data -t bot_adversarial_dialogue:HumanSafetyEvaluation
```

<p align="center"><img width="60%" src="BAD_safety_diagram.png" /></p>


Data (and models) the from the [Build-it, Break-it, Fix-it paper](https://arxiv.org/abs/1908.06083) can be found [here](parl.ai/projects/dialogue_safety).

## Models

TODO: fill me out!!

## Human Evaluations

- Evaluating safety: Mechanical Turk task for analyzing the safety of models will be released shortly. *Check back soon!*

- Evaluating engagingness: To run ACUTE-Eval human evaluations for engagingness, see [here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/acute_eval).


## Citation

If you use the data or models in your own work, please cite with the following BibTex entry:

    @inproceedings{xu2020safetyrecipes,
      author={Jing Xu, Da Ju, Margaret Li, Y-Lan Boureau, Jason Weston, Emily Dinan},
      title={Recipes for Safety in Open-domain Chatbots},
      journal={arXiv preprint arXiv:TBD},
      year={2020},
    }
