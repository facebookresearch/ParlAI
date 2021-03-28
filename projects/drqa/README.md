# DrQA

This agent is a (partial) implementation of the DrQA Document Reader from:

*Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes. 2017.
Reading Wikipedia to Answer Open-Domain Questions.
In Association for Computational Linguistics (ACL).*

Read more about DrQA [here](https://arxiv.org/pdf/1704.00051.pdf).


## Basic Examples

Train a DrQA model on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).
```bash
python projects/drqa/train.py
```

Evaluate a pre-trained model on SQuAD.
```bash
python projects/drqa/eval_pretrained.py
```
