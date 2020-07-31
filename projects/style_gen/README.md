# Style-controlled generation

Agents used for training and evaluating generative models conditioned on a style token, for instance, the "personality" string attached to each example of the Image-Chat dataset.

- `--model style_gen`: Subclass of `TransformerGeneratorAgent` for which style tokens can be appended to the history for every training/evaluation example. Found in `style_gen.py`.
- `--model style_gen/classifier`: Subclass of `TransformerGeneratorAgent` that adds a classifier head on top of the base generator. Can be used for fine-tuning a classifier using the weights of a pretrained generator model. Classifier labels can be specified either in the `'labels'` field of the observation or in a separate `'personality'` field, and all encoder and decoder weights can optionally be frozen when training the classifier head. Found in `classifier.py`.
