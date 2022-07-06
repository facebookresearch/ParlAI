# Transformer

We offer a variety of agent implementations whose core model is the transformer, a self-attention based encoding mechanism first described in [Vaswani et al 2017](https://arxiv.org/abs/1706.03762).

## Agent Variations

- ``transformer/biencoder`` - A retrieval-based agent that encodes a context sequence and a candidate sequence with separate [BERT-based](https://arxiv.org/abs/1810.04805) Transformers. A candidate is chosen via the highest dot-product score between the context and candidate encodings. See [Humeau et al 2019](https://arxiv.org/pdf/1905.01969.pdf) for more details.

- ``transformer/classifier`` - A classifier agent with a Transformer as the model.

- ``transformer/crossencoder`` - A retrieval-based agent that jointly encodes a context and candidate sequence in a single [BERT-based](https://arxiv.org/abs/1810.04805) Transformer, with a final linear layer used to compute a score. A candidate is chosen via the highest scoring encoding. See [Humeau et al 2019](https://arxiv.org/pdf/1905.01969.pdf) for more details.

- ``transformer/generator`` - A generative-based agent that performs seq2seq encoding/decoding with transformer encoders/decoders.

- ``transformer/polyencoder`` - A retrieval-based agent that, similar to the bi-encoder agent, encodes context and candidate sequences with separate [BERT-based](https://arxiv.org/abs/1810.04805) Transformers. However, to compute a final score, the agent performs an additional layer of attention using global context vectors before computing the final dot product, thus incorporating the candidate encoding into the context encoding prior to producing a dot-product score. See [Humeau et al 2019](https://arxiv.org/pdf/1905.01969.pdf) for more details.

- ``transformer/ranker`` - A retrieval-based agent that encodes a context sequence and a candidate sequence with separate Transformers, before computing a dot-product to obtain a score for a candidate encoding.
