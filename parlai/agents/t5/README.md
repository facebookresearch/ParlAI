# T5

"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

See https://arxiv.org/abs/1910.10683.


## Implementation

The T5 model in ParlAI is based on the `T5ForConditionalGeneration` provided by the [HuggingFace Transformers](https://github.com/huggingface/transformers) library. The model can be instantiated with any of the provided architectures there:

- `t5-small`: 60 million parameters
- `t5-base`: 220 million parameters
- `t5-large`: 770 million parameters
- `t5-3b`: 3 billion parameters
- `t5-11b`: 11 billion parameters

**Model Parallel**: HuggingFace has implemented model parallel for T5, however it is an experimental feature, so proceed at your own risk; you can use model parallel by simply specifying `--t5-model-parallel`.

## Basic Examples

### Train t5 large on convai2.
```bash
parlai train_model -m t5 -mf /tmp/model_file -t convai2 -bs 24 --fp16 true -eps 1 -lr 1e-5 --optimizer adam --t5-model-arch t5-large
```
