# SaFeRDialogues: Taking Feedback Gracefully after Conversational Safety Failures 

Megan Ung, Jing Xu, Y-Lan Boureau

## Abstract

Current open-domain conversational models can easily be made to talk in inadequate ways. Online learning from conversational feedback given by the conversation partner is a promising avenue for a model to improve and adapt, so as to generate fewer of these safety failures. However, current state-of-the-art models tend to react to feedback with defensive or oblivious responses. This makes for an unpleasant experience and may discourage conversation partners from giving feedback in the future. This work proposes SaFeRDialogues, a task and dataset of graceful responses to conversational feedback about safety failures.
We collect a dataset of 8k dialogues demonstrating safety failures, feedback signaling them, and a response acknowledging the feedback. We show how fine-tuning on this dataset results in conversations that human raters deem considerably more likely to lead to a civil conversation, without sacrificing engagingness or general conversational ability.

## Paper

[Link](https://arxiv.org/abs/2110.07518)

## Data

```
parlai display_data -t saferdialogues
```

## Models

We release a Blender 2.7B model fine-tuned on the SaFeRDialogues and BST (without persona) tasks to respond to feedback more gracefully after a safety failure.

```
parlai evaluate_model -mf zoo:saferdialogues/model -t saferdialogues
```

## Citation

If you use the dataset or models in your own work, please cite with the
following BibTex entry:
    
    @misc{ung2021saferdialogues,
        title={SaFeRDialogues: Taking Feedback Gracefully after Conversational Safety Failures}, 
        author={Megan Ung and Jing Xu and Y-Lan Boureau},
        year={2021},
        eprint={2110.07518},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }