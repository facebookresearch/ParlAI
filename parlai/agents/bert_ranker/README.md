This directory contains several implementations of a ranker based on a pretrained language model BERT (Devlin et al. https://arxiv.org/abs/1810.04805). It relies on the pytorch implementation provided by Hugging Face (https://github.com/huggingface/pytorch-pretrained-BERT).

## Content

This directory contains 3 Torch Ranker Agents (see parlai/core/torch_ranker_agent.py). All of them are rankers, which means that given a context, they try to guess what is the next utterance among a set of candidates.
- BiEncoderRankerAgent associates a vector to the context and a vector to every possible utterance, and is trained to maximize the dot product between the correct utterance and the context.
- CrossEncoderRankerAgent concatenate the text with a candidate utterance and gives a score. This scales much less that BiEncoderRankerAgent at inference time since you can not precompute a vector per candidate. However, it tends to give higher accuracy.
- BothEncoderRankerAgent does both, it ranks the top N candidates using a BiEncoder and follows it by a CrossEncoder. Resulting in a scalable and precise system.

# Preliminary
In order to use those agents you need to install pytorch-pretrained-bert (https://github.com/huggingface/pytorch-pretrained-BERT).
```pip install pytorch-pretrained-bert```

## Training on convai2
The following lines can be used to train models on 8 recent GPUs using BERT based. Reduce the batch size to use fewer GPUS

* Biencoder
```
# How do I use that?

See the examples in testing.sh to train/eval/discuss with the model
