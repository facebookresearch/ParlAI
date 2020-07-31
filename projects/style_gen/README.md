# Style-controlled generation

Agent for training and evaluating generative models conditioned on a style token, for instance, the "personality" string attached to each example of the Image-Chat dataset.

- `--model style_gen`: Subclass of `TransformerGeneratorAgent` for which style tokens can be appended to the history for every training/evaluation example. Found in `style_gen.py`.
