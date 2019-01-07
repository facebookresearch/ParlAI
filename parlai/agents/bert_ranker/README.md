This directory contains several implementations of a ranker based on a pretrained language model BERT (Devlin et al. https://arxiv.org/abs/1810.04805). It relies on the pytorch implementation provided by Hugging Face (https://github.com/huggingface/pytorch-pretrained-BERT).

## Content

This directory contains 3 Torch Ranker Agents (see parlai/core/torch_ranker_agent.py). All of them are rankers, which means that given a context, they try to guess what is the next utterance among a set of candidates.
- BiEncoderRankerAgent associates a vector to the context and a vector to every possible utterance, and is trained to maximize the dot product between the correct utterance and the context.
- CrossEncoderRankerAgent concatenate the text with a candidate utterance and gives a score. This scales much less that BiEncoderRankerAgent at inference time since you can not precompute a vector per candidate. However, it tends to give higher accuracy.
- BothEncoderRankerAgent does both, it ranks the top N candidates using a BiEncoder and follows it by a CrossEncoder. Resulting in a scalable and precise system.

## Preliminary
In order to use those agents you need to install pytorch-pretrained-bert (https://github.com/huggingface/pytorch-pretrained-BERT).
```pip install pytorch-pretrained-bert```

## Training on convai2
The following lines can be used to train models on 8 recent GPUs using BERT based. Reduce the batch size to use fewer GPUS

* Biencoder
```
#train
PYTHONPATH=. python -u examples/train_model.py -pyt convai2 -m parlai.agents.bert_ranker.biencoder_ranker:BiEncoderRankerAgent --batchsize 120 --dict-file ./mypath/dictionary --model-file ./mypath/model --multigpu True --num-epochs 3 --eval-batchsize 8 --learningrate 5e-5  --log_every_n_secs 20 --shuffle true --type-optimization all_encoder_layers

# should yield valid:{'exs': 7801, 'accuracy': 0.7868, 'f1': 0.8098, 'hits@1': 0.787, 'hits@5': 0.968, 'hits@10': 0.993, 'hits@100': 1.0, 'bleu': 0.7869, 'examples': 7801, 'loss': 750.0, 'mean_loss': 0.09614, 'mean_rank': 1.564}

#Interact with a set of fixed candidates
PYTHONPATH=. python examples/interactive.py -m parlai.agents.bert_ranker.biencoder_ranker:BiEncoderRankerAgent --fixed-candidates-path ./mypath/pc_all_cands.txt --model-file ./mypath/model --eval-candidates fixed
Enter Your Message: hi
[TorchAgent]: hi how are you today
```

* Crossencoder
```
#train
PYTHONPATH=. python -u examples/train_model.py -pyt convai2 -m parlai.agents.bert_ranker.crossencoder_ranker:CrossEncoderRankerAgent --batchsize 5 --dict-file ./mypath/dictionary --model-file ./mypath/model --multigpu True --num-epochs 3 --eval-batchsize 8 --learningrate 5e-5 --validation-every-n-secs 2400 --validation-max-exs 4000 --log_every_n_secs 20 --shuffle true --type-optimization all_encoder_layers
#should yield valid:{'exs': 7801, 'accuracy': 0.8632, 'f1': 0.8783, 'hits@1': 0.863, 'hits@5': 0.984, 'hits@10': 0.996, 'hits@100': 1.0, 'bleu': 0.8632, 'examples': 7801, 'loss': 5824.0, 'mean_loss': 0.7465, 'mean_rank': 1.32}
```

* Both
You can train both simultanously but it's lighter to train separately the BiEncoder then the cross encoder and then interact with them:
```
PYTHONPATH=. python examples/interactive.py -m parlai.agents.bert_ranker.bert_ranker:BothEncoderRankerAgent -pyt convai2 --fixed-candidates-path ./mypath/candidates --eval-candidates fixed --biencoder-model-file ./mypath/biencodermodel --crossencoder-model-file ./mypath/biencodermodel
Enter Your Message: hi
[TorchAgent]: hello there . how are you today ?
```
