# Anti-scaling

Contains utility code for speeding up the inference time of generator models.

## Knowledge distillation

`distillation.py` contains agents for performing knowledge distillation on a Transformer sequence-to-sequence model consisting of an encoder and a decoder. Two types of distillation are supported, DistilBERT-style and TinyBERT-style:

### DistilBERT-style distillation

Distillation in the style of [Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *arXiv preprint arXiv:1910.01108* (2019).](https://arxiv.org/abs/1910.01108)

With DistilBERT-style distillation, the student model is created by removing a subset of layers from the encoder and decoder of the teacher model, and the weights of the remaining layers and the embedding layer are initialized from the corresponding weights in the teacher model.

When performing distillation, terms are added for losses on the encoder output, the outputs of the encoder and decoder hidden layers, and on the prediction layer (i.e. the soft target probabilities). `DistillTransformerAgent` is used for distilling `transformer/generator` models (i.e. models specified by `--model transformer/generator`), and `DistillBartAgent` is used for distilling `bart` models.

### TinyBERT-style distillation

Distillation in the style of [Jiao, Xiaoqi, et al. "Tinybert: Distilling bert for natural language understanding." *arXiv preprint arXiv:1909.10351* (2019).](https://arxiv.org/abs/1909.10351)

With TinyBERT-style distillation, the student model can have smaller hidden and FFN dimensions than the teacher model, and projection matrices will be used to measure losses such as those between the hidden-layer outputs. Unlike with DistilBERT-style distillation, the weights of the teacher model cannot be used to initialize the student model.

In addition to the losses of DistilBERT-style distillation above, losses are also included on the embedding layer and on the per-layer query/key product matrices from encoder self-attention, decoder self-attention, and encoder/decoder attention. `DistillNarrowTransformerAgent` is used for distilling `transformer/generator` models, and `DistillNarrowBartAgent` is used for distilling `bart` models.

After distillation, the projection matrices will still be included in the saved model file; run `scripts/remove_projection_matrices.py` to remove them. This is necessary for loading the model file as a `transformer/generator` or `bart` model.

### Sample command

The following command can be used to launch TinyBERT-style distillation of the BlenderBot3B model, with 15 of 24 decoder layers removed. The best values for the loss coefficients will likely vary depending which model is used as the teacher model, the dataset being fine-tuned, etc.

```
cd ${PARLAI_REPO_FOLDER}
python -c "from parlai.zoo.blender.blender_3B import download; download('data')"
# To manually download the BlenderBot3B model, required for specifying `--init-opt`
parlai train_model \
--allow-missing-init-opts True \
--init-model None \
--init-opt data/models/blender/blender_3B/model.opt \
--dict-file data/models/blender/blender_3B/model.dict \
--model projects.anti_scaling.distillation:DistillNarrowTransformerAgent \
--teacher-model data/models/blender/blender_3B/model \
--batchsize 16 \
--embedding-size 2560 \
--ffn-size 10240 \
--fp16 True \
--gpu -1 \
--learningrate 0.0001 \
--lr-scheduler reduceonplateau \
--max-lr-steps -1 \
--max-train-time -1 \
--model-parallel False \
--save-after-valid True \
--skip-generation True \
--task blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues \
-veps -1 \
-vmm min \
-vmt ppl \
-vp 20 \
-vtim 900 \
--n-encoder-layers 2 \
--n-decoder-layers 9 \
--embedding-loss-coeff 4 \
--hidden-loss-coeff 64 \
--self-attn-loss-coeff 4 \
--enc-dec-attn-loss-coeff 64 \
--encoder-loss-coeff 0 \
--pred-loss-coeff 64 \
--task-loss-coeff 1 \
--model-file ${SAVE_PATH}
```
