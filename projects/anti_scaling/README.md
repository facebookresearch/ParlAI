# Anti-scaling

Contains utility code for speeding up the inference time of generator models.

## Distillation

`distillation.py` contains agents for performing knowledge distillation on a Transformer generator model. Two types of distillation are supported, DistilBERT-style and TinyBERT-style:

### DistilBERT-style distillation

{{{TODO: give paper}}}

With DistilBERT-style distillation, the student model is first initialized by removing a subsection of layers from the encoder and decoder, and the weights of the remaining layers and the embedding layer are initialized from weights in the teacher model.

Terms are added for losses on the encoder output, the outputs of the encoder and decoder hidden layers, and on the prediction layer (i.e. the soft target probabilities). `DistillTransformerAgent` is used for distilling `transformer/generator` models (i.e. models specified by `--model transformer/generator`), and `DistillBartAgent` is used for distilling `bart` models.

### TinyBERT-style distillation

{{{TODO: give paper; explain what it does. Say you can train narrower models}}}

In addition to the losses of DistilBERT-style distillation above, losses are also included on the embedding layer and on the per-layer query/key product matrices from encoder self-attention, decoder self-attention, and encoder/decoder attention. `DistillNarrowTransformerAgent` is used for distilling `transformer/generator` models, and `DistillNarrowBartAgent` is used for distilling `bart` models.

### Sample command

The following command can be used to launch distillation of the BlenderBot3B model, with {{{TODO: give some architecture details}}}. The best values for the loss coefficients will likely vary on which model is used as the teacher model, the dataset being fine-tuned, etc.

```
{{{TODO: best run of the sweep so far}}}
```
