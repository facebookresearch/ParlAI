# HRED Agent
The HRED agent uses a traditional LSTM encoder decoder, but also utilizes a context LSTM that encodes the history.

The following papers outline more information regarding this model:
  - Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models [(IV Serban et al. 2015)](https://arxiv.org/abs/1507.04808)
  - A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues
    [(IV Serban et al. 2017)](http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf)

An important difference is that the model currently only supports LSTM RNN units, rather than the GRU units used in the papers. It also supports the decoding strategies in TorchGeneratorModel (such as beam search and greedy).

Example script to run on dailydialog:
parlai train_model -t dailydialog -mf /tmp/dailydialog_hred -bs 4 -eps 5 --model hred

