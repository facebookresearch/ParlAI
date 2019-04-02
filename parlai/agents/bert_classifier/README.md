# BERT Classifier

This directory contains an implementations of a classifier based on a pretrained language model BERT (Devlin et al. https://arxiv.org/abs/1810.04805).
It relies on the pytorch implementation provided by Hugging Face (https://github.com/huggingface/pytorch-pretrained-BERT).


## Basic Examples

Train a classifier on the SNLI tas.
```bash
python examples/train_model.py -m bert_classifier -t snli --classes 'entailment' 'contradiction' 'neutral' -mf /tmp/BERT_snli -bs 20
```
