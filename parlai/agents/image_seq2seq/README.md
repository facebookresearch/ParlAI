# Image+Seq2Seq
The **Image+Seq2Seq** agent is a model that incorporates image features with a sequence to sequence transformer generator. A core component of the [dodecaDialogue](https://parl.ai/projects/dodecadialogue/) task.

## Basic Examples
Train an **Image+Seq2Seq** model on an image captioning task:
```bash
python parlai/scripts/train_model.py -m image_seq2seq -t flickr30k --image-mode resnext101_32x48d_wsl -mf /tmp/model
```

Train an **Image+Seq2Seq** model on a dialogue task:
```bash
python parlai/scripts/train_model.py -m image_seq2seq -t convai2 -mf /tmp/model
```

Multi-task train an **Image+Seq2Seq** model on a dialogue and captioning task:
```bash
python parlai/scripts/train_model.py -m image_seq2seq -t flickr30k,convai2 -mf /tmp/model --image-mode resnext101_32x48d_wsl
```
