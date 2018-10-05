# VSEPP Caption
 The *VSEPP Caption* agent is an agent that takes an image and retrieves a
 caption. It is based the paper [VSE++: Improving Visual-Semantic Embeddings with Hard Negatives (Faghri et al. 2017)] (<arxiv.org/abs/1707.05612>),
 and is adapted from [its original implementation](<https://github.com/fartashf/vsepp>).

 ## Basic Examples
 Train the agent on the Flickr30k dataset.
```bash
python examples/train_model.py -m vsepp_caption -t flickr30k -mf /tmp/vsepp
```
 After training, load and evaluate that model on the Flickr30k test set.
```bash
python examples/eval_model.py -t flickr30k -mf /tmp/vsepp -dt test
```
