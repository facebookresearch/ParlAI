# Safe Local Human

This is an agent which takes input from the command line in the `act` function, and applies a safety classifier to process the messages. It is used in the [safe interactive script](https://github.com/facebookresearch/ParlAI/blob/master/parlai/scripts/safe_interactive.py).

## Basic Examples

Interact with a Poly-encoder model:
```bash
python parlai/scripts/safe_interactive.py -mf zoo:pretrained_transformers/model_poly/model -t convai2
```
