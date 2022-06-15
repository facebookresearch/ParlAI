# DIRECTOR: Generator-Classifiers For Supervised Language Modeling


Kushal Arora, Kurt Shuster, Sainbayar Sukhbaatar, Jason Weston

<!-- Paper: https://arxiv.org/2206. -->


## Abstract

Current language models achieve low perplexity but their resulting generations still suffer from toxic responses, repetitiveness and contradictions. The standard language modeling setup fails to address these issues. In this paper, we introduce a new architecture, {\sc Director}, that consists of a unified generator-classifier with both a language modeling and a classification head for each output token. Training is conducted jointly using both standard language modeling data, and data labeled with desirable and undesirable sequences. Experiments in several settings show that the model has competitive training and decoding speed compared to standard language models while yielding superior results, alleviating known issues while maintaining generation quality. It also outperforms existing model guiding approaches in terms of both accuracy and efficiency.


