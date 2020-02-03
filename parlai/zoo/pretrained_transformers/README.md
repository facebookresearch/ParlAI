This directory contains:
 - pretrained transformers on Reddit and Wikipedia + Toronto Books that can be
  used as a base for pretraining
 - pretrained models fine tuned on ConvAI2 for the bi-encoder and poly-encoder
   The polyencoder scores 89+ on convai2 valid set and is fast enough to
   interact in real time with 100k candidates (which are provided.)

## Interacting with a pretrained model on Convai2

Run this command: (assumes your model zoo is in the default ./data/models)
```
python examples/interactive.py -m transformer/polyencoder \
    -mf zoo:pretrained_transformers/model_poly/model \
    --encode-candidate-vecs true \
    --eval-candidates fixed  \
    --fixed-candidates-path data/models/pretrained_transformers/convai_trainset_cands.txt
```

Example output:
```
Enter Your Message: your persona: i love to drink fancy tea.\nyour persona: i have a big library at home.\nyour persona: i'm a museum tour guide.\nhi how are you doing ?
[Polyencoder]: i am alright . i am back from the library .
Enter Your Message: oh, what do you do for a living?
[Polyencoder]: i work at the museum downtown . i love it there .
Enter Your Message: what is your favorite drink?
[Polyencoder]: i am more of a tea guy . i get my tea from china .
```

Note the polyencoder gives 89+ hits@1/20 on convai2, however, it expects data
that is close to the dataset. If you do not include the multiple 'your persona: ...\n'
at the beginning it will answer nonsense.

## Fine tuning on your own tasks

### bi-encoder

Execute this to train a biencoder scoring 86+ on Convai2 valid set
(requires 8 x GPU 32GB., If you don't have this, reduce the batch size )

```
python -u examples/train_model.py \
    --init-model zoo:pretrained_transformers/bi_model_huge_reddit/model \
    --batchsize 512 -t convai2 \
    --shuffle true --model transformer/biencoder --eval-batchsize 6 \
    --warmup_updates 100 --lr-scheduler-patience 0 \
    --lr-scheduler-decay 0.4 -lr 5e-05 --data-parallel True \
    --history-size 20 --label-truncate 72 --text-truncate 360 \
    --num-epochs 10.0 --max_train_time 200000 -veps 0.5 -vme 8000 \
    --validation-metric accuracy --validation-metric-mode max \
    --save-after-valid True --log_every_n_secs 20 --candidates batch \
    --dict-tokenizer bpe --dict-lower True --optimizer adamax \
    --output-scaling 0.06 \
     --variant xlm --reduction-type mean --share-encoders False \
     --learn-positional-embeddings True --n-layers 12 --n-heads 12 \
     --ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 \
     --n-positions 1024 --embedding-size 768 --activation gelu \
     --embeddings-scale False --n-segments 2 --learn-embeddings True \
     --share-word-embeddings False --dict-endtoken __start__ --fp16 True \
     --model-file <YOUR MODEL FILE>
```

### poly-encoder

Execute this to train a poly-encoder scoring 89+ on Convai2 valid set
(requires 8 x GPU 32GB., If you don't have this, reduce the batch size )

```
python -u examples/train_model.py \
  --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model \
  -t convai2 --shuffle true \
  --model transformer/polyencoder --batchsize 256 --eval-batchsize 10 \
  --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
  -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
  --text-truncate 360 --num-epochs 8.0 --max_train_time 200000 -veps 0.5 \
  -vme 8000 --validation-metric accuracy --validation-metric-mode max \
  --save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True \
  --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
  --variant xlm --reduction-type mean --share-encoders False \
  --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
  --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
  --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
  --learn-embeddings True --polyencoder-type n_first --poly-n-codes 64 \
  --poly-attention-type basic --dict-endtoken __start__ \
  --model-file <YOUR MODEL FILE>
```

### cross-encoder

Execute this to train a cross-encoder scoring 90+ on Convai2 valid set
(requires 8 x GPU 32GB., If you don't have this, reduce the batch size )

```
python -u examples/train_model.py \
  --init-model zoo:pretrained_transformers/cross_model_huge_reddit/model \
  -t convai2 --shuffle true \
  --model transformer/crossencoder --batchsize 16 --eval-batchsize 10 \
  --warmup_updates 1000 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
  -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
  --text-truncate 360 --num-epochs 12.0 --max_train_time 200000 -veps 0.5 \
  -vme 2500 --validation-metric accuracy --validation-metric-mode max \
  --save-after-valid True --log_every_n_secs 20 --candidates inline --fp16 True \
  --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
  --variant xlm --reduction-type first --share-encoders False \
  --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
  --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
  --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
  --learn-embeddings True --dict-endtoken __start__ \
  --model-file <YOUR MODEL FILE>
```
