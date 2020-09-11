## Don’t Say _That_! Making Inconsistent Dialogue Unlikely with Unlikelihood Training

Margaret Li, Stephen Roller, Ilia Kulikov, Sean Welleck, Y-Lan Boureau, Kyunghyun Cho, Jason Weston

## Abstract

Generative dialogue models currently suffer from a number of problems which standard maximum likelihood training 
does not address.  They tend to produce generations that (i) rely too much on copying from the context, (ii) 
contain repetitions within utterances, (iii) overuse frequent words, and (iv) at a deeper level, contain logical flaws.
In this work we show how all of these problems can be addressed by extending the recently introduced unlikelihood loss 
(Welleck et al., 2019) to these cases. We show that appropriate loss functions regularizing the generated outputs to 
match human distributions are effective for the first three issues. For the last important general issue, we show that 
collecting training data of _what a model should not do_ is effective for improving logical consistency, 
potentially paving the way to generative models with greater reasoning ability. 
We demonstrate the efficacy of our approach across several dialogue tasks.

## Paper

[Link](https://drive.google.com/open?id=1Du-FhnApmH_72gqWnnQyjigKDpmN9mBI)


## Pretrained Models

We release 13 models via the [ParlAI Model Zoo](https://www.parl.ai/docs/zoo.html). 

- Nine models are trained with repetition unlikelihood to reduce repeats. For each of the three datasets ConvAI2, 
Wizard of Wikipedia, and ELI5, we release one model each whicih targets context repeats, label repeats, and both.
- Four models are trained with vocab unlikelihood on the ConvAI2 dataset with alpha values 1e0, 1e1, 1e2, and 1e3.


## Code

Code for the unlikelihood agents is released in `agents.py`


## Usage

### Evaluation

To evaluate the existing pretrained models, use the standard ParlAI eval model script. For example, we evaluate the 
vocab unlikelihood model with alpha 1e0:

```
parlai eval_model -mf zoo:dialogue_unlikelihood/vocab_alpha1e1/model 
-m projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent 
-t convai2 --beam-size 1 --skip-generation False
```


### Repetition Unlikelihood

To train a repetition unlikelihood agent on any task from scratch, use the standard ParlAI train model script:

```
parlai train_model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent -t convai2
```
Several command line arguments can be added to change the behavior of training. Notably, `--seq-ul-ratio` 
adjusts the weight of the unlikelihood loss (in the weighted sum with MLE loss) - higher values means more 
weight on unlikelihood. `--ctxt-beta` changes the weight of context repetition related unlikelihood against 
label repeats - higher values mean context repeats are more heavily penalized than label repeats. See `agents.py` 
for the full list of arguments.


### Vocab Unlikelihood

To use vocab unlikelihood you need to make sure to include a vocabulary incidence counts file. Each line of the 
counts file should be a json dict with at least these fields:
```
{"word_id": 15, "bin": "frequent"}
```
where "word_id" is id assigned to the word by the dictionary, and "bin" is "frequent", "medium", "rare", 
or "veryrare", depending on the probability of the word occurring.

To train a vocab unlikelihood agent on any task from scratch, use the same parlai train_model function.
This exact invocation below will only work if you make sure the `--counts-file` filepath is correct.

```
parlai train_model -m projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent 
-t convai2 --counts-file ~/ParlAI/data/models/dialogue_unlikelihood/vocab_alpha1e0/counts.txt
```

### Reward Unlikelihood

Our reward unlikelihood agent can be used to push down on any set of negative candidate responses. In our paper, 
we use this agent for our safety experiments. 
In order to use the reward unlikelihood agent, you'll need to have a task which includes negative rewards. 
To train a reward unlikelihood agent on any task from scratch

Note that models in the paper were trained from a pretrained base model, as reflected in the commands above. While 
it's possible to train a model from scratch using unlikelihood, it would be quite slow, and therefore we advise 
starting with a pretrained base. The easiest way to do this is to use a `transformer/generator` in our model zoo 
or train one from scratch:

```
parlai train_model -m transformer/generator -t reddit
```
