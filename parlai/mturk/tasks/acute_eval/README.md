# ACUTE-Eval

## Paper information

Margaret Li, Stephen Roller, Jason Weston.
_[ACUTE-Eval](https://arxiv.org/abs/)_.

## Citation

If you use this evaluation method in your own work, please cite with the
following BibTex entry:

# Code Instructions
Once you have [installed ParlAI](https://github.com/facebookresearch/ParlAI/#installing-parlai),
follow the instructions below.

## Formatting conversation data

This task code assumes that you've parsed and saved your collected conversations in a simple .jsonl format. This is a template of the format with the minimal expected fields:

    {
      "model_name": "model_name",
      "conversation_id": "id",
      "dialog": [
        {"speaker": "model", "text": "Hi"},
        {"speaker": "human_evaluator", "text": "Hi back"},
        ...
      ]
    }

See `example/model*.jsonl` for examples of the format required.


** **
