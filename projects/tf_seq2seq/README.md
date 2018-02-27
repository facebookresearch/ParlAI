
# Installation

Follow Google's instructions to install at https://google.github.io/seq2seq/getting_started/.

Then you can train with commands like the following:
```
python train.py --tf_train ~/src/seq2seq/bin/train.py -bs 128 -t personachat --config_paths="
./config/personachat.yml,
./config/train_seq2seq.yml,
./config/text_metrics_sp.yml"
```
