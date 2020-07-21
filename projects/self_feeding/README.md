# Self-feeding Chatbot

## Paper information
Braden Hancock, Antoine Bordes, Pierre-Emmanuel Mazaré, Jason Weston.
_[Learning from Dialogue after Deployment: Feed Yourself, Chatbot!](https://arxiv.org/abs/1901.05415)_.
To appear in ACL 2019.


## Abstract

The majority of conversations a dialogue agent sees over its lifetime occur after it has already been trained and deployed, leaving a vast store of potential training signal untapped.
In this work, we propose the self-feeding chatbot, a dialogue agent with the ability to extract new training examples from the conversations it participates in. As our agent engages in conversation, it also estimates user satisfaction in its responses.
When the conversation appears to be going well, the user’s responses become new training examples to imitate.
When the agent believes it has made a mistake, it asks for feedback; learning to predict the feedback that will be given improves the chatbot’s dialogue abilities further.
On the PersonaChat chit-chat dataset with over 131k training examples, we find that learning from dialogue with a self- feeding chatbot significantly improves performance, regardless of the amount of traditional supervision.

## Citation

If you use the dataset or models in your own work, please cite with the
following BibTex entry:

    @inproceedings{hancock2019feed,
      author={Braden Hancock and Antoine Bordes and Pierre-Emmanuel Mazar\'{e} and Jason Weston},
      booktitle={Association for Computational Linguistics (ACL)},
      title={Learning from Dialogue after Deployment: Feed Yourself, Chatbot!},
      url={https://arxiv.org/abs/1901.05415},
      year={2019},
    }

# Code Instructions

Once you have [installed ParlAI](https://github.com/facebookresearch/ParlAI/#installing-parlai), follow the instructions below.

## Download the data

Running the commands to train or chat with the models will automatically download the data for you.
Alternatively, you can manually download the data by running `python projects/self_feeding/download_data.py`. This will download the following files to `data/self_feeding/`:

- `{train, valid, test}_hh.txt`: DIALOGUE Human-Human (HH) conversations from the PersonaChat dataset, with one context and response per line (train: 131,438; valid: 2,000; test: 5,801).
- `train_hb.txt`: DIALOGUE Human-Bot (HB) conversations collected between crowdworkers and a trained chatbot, with only human utterances as responses (train: 131,923).
- `train_fb_a.txt`: FEEDBACK Human-Bot conversations wherein all responses are the feedback given by a human in response to a request by the bot after it estimated that the human was dissatisfied with its previous response. (The turns where the bot messed up, the human expressed dissatisfaction, and the bot requested feedback are removed so that the context is primarily normal-looking conversation). (train: 40,082)
- `train_fb_b.txt`: The same as `train_fb_a.txt` but with a chatbot that was retrained using the additional feedback examples collected from the A set (train: 21,257).
- `{valid, test}_fb.txt`: FEEDBACK validation and test sets collected at the same time and with the same model as the `train_fb_a.txt` file.

We also include three derivative files for convenience (as they were used in experiments and in some of the sample commands in the sections below):

- `train_fb.txt`: The result of `cat train_fb_a.txt train_fb_b.txt | shuf > train_fb.txt`
- `train_hb60k.txt`: The result of `head -n 60000 train_hb.txt > train_hb60k.txt`
- `train_hh131k_hb60k.txt`: The result of `cat train_hh.txt train_hb60k.txt > train_hh131k_hb60k.txt`

For more context on the scenarios in which these data were collected (including screenshots of crowdworker interfaces), refer to the paper.
In this distribution, we include all data collected of each type.
To recreate the exact datasets used in the paper, keep only the first X lines of each file such that the resulting sets match the sizes reported in Table 1.

## Train a model

To train a model, use the standard `ParlAI` protocol with `train_model.py`.
The following commands assume that you have set the following environment variables:

```
export PARLAIHOME=/path/to/ParlAI
export MODEL=/path/to/model
```

You may require a GPU to train a model to convergence in a reasonable amount of time.
On a P100 GPU, these training commands take approximately 10 minutes to converge.

### Train on the DIALOGUE (HH) examples
Here is a minimal command for training on the DIALOGUE task using Human-Human (HH) examples:

```
parlai train_model -t self_feeding:dialog --model projects.self_feeding.self_feeding_agent:SelfFeedingAgent --model-file /tmp/mymodel1 -bs 128
```

Or to recreate the results in the paper for training on 131k HH examples with the same hyperparameters that we used, run the following:

```
parlai train_model -t self_feeding:dialog --model-file /tmp/mymodel2 -ltim 5 -vtim 10 -vp 10 -m projects.self_feeding.self_feeding_agent:SelfFeedingAgent -cands batch --eval-candidates inline -histsz 2 --embedding-type fasttext_cc --embedding-size 300 --dict-maxtokens 250000 --num-epochs 100 --optimizer adamax --embeddings-scale false -bs 128 --relu-dropout 0 --attention-dropout 0 --n-heads 2 --n-layers 2 -lr 0.0025 --ffn-size 32 --lr-scheduler invsqrt --warmup-updates 500 -vmt dia_acc -vmm max
```

### Train on DIALOGUE (HH) + DIALOGUE (HB) examples

To train on both HH and HB DIALOGUE examples, point the model to a train file that includes examples from both sets. For example, if you combined 131k HH DIALOGUE examples and 60k HB dialogue examples into a file called `train_hh131k_hb60k.txt`, you could add the following flag to train on that combined file for the DIALOGUE task:

```
--dia-train train_hh131k_hb60k.txt
```

### Train on DIALOGUE (HH) + FEEDBACK examples

To train on more than one task (such as DIALOGUE and FEEDBACK), modify the command for training on DIALOGUE (HH) alone as follows:

- Change `-t self_feeding:dialog` to `-t self_feeding:diafee`. This will result in a different "teacher" agent being used to train the chatbot, one with access to both 'dia\[logue\]' and 'fee\[dback\]`.

Putting this all together, the command to recreate the 131k HH + 60k FB result from the paper is as follows (as reported in Table 9 in the paper, this setting had the same optimal hyperparameter settings as 131k HH):

```
parlai train_model -t self_feeding:diafee --model-file /tmp/mymodel3 -ltim 5 -vtim 10 -vp 10 -m projects.self_feeding.self_feeding_agent:SelfFeedingAgent -cands batch --eval-candidates inline -histsz 2 --embedding-type fasttext_cc --embedding-size 300 --dict-maxtokens 250000 --num-epochs 100 --optimizer adamax --embeddings-scale false -bs 128 --relu-dropout 0 --attention-dropout 0 --n-heads 2 --n-layers 2 -lr 0.0025 --ffn-size 32 --lr-scheduler invsqrt --warmup-updates 500 -vmt dia_acc -vmm max
```

### Train on DIALOGUE (HH) + DIALOGUE (HB) + FEEDBACK (FB) + SATISFACTION (ST) examples
You can train on all three tasks at once with the command below.
```
parlai train_model -t self_feeding:all --model-file /tmp/mymodel4 -ltim 5 -vtim 10 -vp 50 -m projects.self_feeding.self_feeding_agent:SelfFeedingAgent -cands batch --eval-candidates inline -histsz 2 --embedding-type fasttext_cc --embedding-size 300 --dict-maxtokens 250000 --num-epochs 500 --optimizer adamax --embeddings-scale false -bs 128 --relu-dropout 0 --attention-dropout 0 --n-heads 2 --n-layers 2 -lr 0.0025 --ffn-size 32 --lr-scheduler invsqrt --warmup-updates 500 --dia-train train_hh131k_hb60k.txt -vmt dia_acc -vmm max
```

### Evaluate a trained model
To evaluate a model, use the following command, which specifies which teacher to use (the one with all three tasks), which splits to test on (`test` or `valid`), and what batch size to use (larger will evaluate faster):
```
parlai eval_model -mf /tmp/mymodel1 -t self_feeding:all --datatype test -bs 20
```


## Using a pretrained Model

Running any of the following commands will automatically download a pretrained model
and place it in `data/models/self_feeding`. You can then evaluate this model on the
corpus, or even chat with it live!

### Evaluating the pretrained model

You can

```
parlai eval_model -mf zoo:self_feeding/hh131k_hb60k_fb60k_st1k/model -t self_feeding:all --datatype test -bs 20
```

### Chat with a pretrained model
To chat with a model that's already been trained, use the `interactive.py` script.
You can add the flag `--request-feedback true` to have the model ask for feedback based on its estimate of your satisfaction with the conversation.
```
python projects/self_feeding/interactive.py --model-file zoo:self_feeding/hh131k_hb60k_fb60k_st1k/model --no-cuda
```

You can change the filename to any of your own models to interactive with a
model you have trained.
