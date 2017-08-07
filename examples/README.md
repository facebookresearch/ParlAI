# ParlAI examples

This directory contains a few particular examples of basic loops.

- base_train.py: _very simple example shows the outline of a training/validation loop using the default Agent parent class_
- display_data.py: _uses agent.repeat_label to display data from a particular task provided on the command-line_
- display_model.py: _shows the predictions of a provided model on a particular task provided on the command-line_
- eval_model.py: _uses the named agent to compute evaluation metrics data for a particular task provided on the command-line_
- build_dict.py: _build a dictionary from a particular task provided on the command-line using core.dict.DictionaryAgent_
- memnn_luatorch_cpu: _shows a few examples of training an end-to-end memory network on a few datasets_

## Running These Examples

Most of them can be run simply by typing `python {example}.py -t {task_name}`. Here are some examples:

Display 10 random examples from task 1 of the "1k training examples" bAbI task:
```bash
python display_data.py -t babi:task1k:1
```

Run a train/valid loop with the basic agent (which prints what it receives and then says hello to the teacher, rather than learning anything) on the babi task:
```bash
python base_train.py -t babi:task1k:1
```

Displays 100 random examples from multi-tasking on the bAbI task and the SQuAD dataset at the same time:
```bash
python display_data.py -t babi:task1k:1,squad -n 100
```

Evaluate on the bAbI test set with a human agent (using the local keyboard as input):
```bash
python examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid
```

Evaluate an IR baseline model on the validation set of the Movies Subreddit dataset:
```bash
python eval_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid
```

Display the predictions of that same IR baseline model:
```bash
python display_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid
```

Build a dictionary on a bAbI "1k training examples" task 1 and save it to /tmp/dict.tsv
```bash
python build_dict.py -t babi:task1k:1 --dict-file /tmp/dict.tsv
```

Train a simple sequence to sequence model on the "1k training examples" bAbI task 1 with batch size of 8 examples for one epoch (requires pytorch):
```bash
python train_model.py -m seq2seq -t babi:task1k:1 -bs 8 -e 1 -mf /tmp/model_s2s
```

Trains an attentive LSTM model of [Chen et al.](https://arxiv.org/abs/1704.00051) on the SQuAD dataset with a batch size of 32 examples (requires pytorch):
```bash
python train_model.py -m drqa -t squad -bs 32 -mf /tmp/model_drqa
```

Evaluates on an already trained SQuAD model:
```bash
wget https://s3.amazonaws.com/fair-data/parlai/_models/drqa/squad.mdl
python eval_model.py -m drqa -t squad -mf squad.mdl -dt valid
```

Interactive session on an already trained SQuAD model:
```bash
wget https://s3.amazonaws.com/fair-data/parlai/_models/drqa/squad.mdl
python interactive.py -m drqa -mf squad.mdl
```

Train a simple cpu-based memory network on the "10k training examples" bAbI task 1 with 8 threads (python processes) using Hogwild (requires zmq and Lua Torch):
```bash
python memnn_luatorch_cpu/full_task_train.py -t babi:task10k:1 -nt 8
```
