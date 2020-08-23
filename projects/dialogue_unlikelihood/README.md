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

## Code

Code for the unlikelihood agents is released in `agents.py`

## Pretrained Models

We release 13 models via the [ParlAI Model Zoo](https://www.parl.ai/docs/zoo.html). 

- Nine models are trained with repetition unlikelihood to reduce repeats. For each of the three datasets ConvAI2, Wizard of Wikipedia, and ELI5, we release one model each whicih targets context repeats, label repeats, and both.
- Four models are trained with vocab unlikelihood on the ConvAI2 dataset with alpha values 1e0, 1e1, 1e2, and 1e3.

