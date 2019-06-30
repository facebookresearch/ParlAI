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
Alternatively, you can manually download the data by running `python parlai/projects/self_feeding/download_data.py`. 
This will download the following files to `data/dialogue_sf`:
- `{train, valid, test}_hh.txt`: Human-Human (HH) conversations from the PersonaChat dataset, with one context and response per line (train: 131,438; valid: 2,000; test: 5,801).
- `train_hb.txt`: Human-Bot conversations collected between crowdworkers and a trained chatbot, with only human utterances as responses (train: 131,923).
- `{train, valid, test}_fb_a.txt`: Human-Bot conversations wherein all responses are the feedback given by a human in response to a request by the bot after it estimated that the human was dissatisfied with its previous response. (The turns where the bot messed up, the human expressed dissatisfaction, and the bot requested feedback are removed so that the context is primarily normal-looking conversation).
-`train_fb_b.txt`: The same as `train_fb_a.txt` but with a chatbot that was retrained using the additional feedback examples collected from the A set.

For more context on the scenarios in which these data were collected (including screenshots of crowdworker interfaces), refer to the paper.

## Train a model from scratch
To train a model from scratch, use the standard `ParlAI` protocol with `train_model.py`. For example, you might run the following from the root of the ParlAI directory:
```
export MODEL=model_from_scratch; python -u $PARLAIHOME/examples/train_model.py -t self_feeding:dialog:train -mf $PARLAIHOME/models/$MODEL -tblog true -tbcomment $MODEL -tbmetrics lr,dia_acc,dia_loss,dia_rank -ltim 5 -vtim 10 -vp 10 -m projects.self_feeding.metadialog_agent:SelfFeedingAgent -cands batch -ecands inline -histsz 2 --embedding-type fasttext_cc --embedding-size 300 --dict-maxtokens 250000 --num-epochs 100 --optimizer adamax --embeddings-scale false -bs 128 --relu-dropout 0 --attention-dropout 0 --n-heads 2 --n-layers 2 -lr 0.0025 --ffn-size 32 --lr-scheduler invsqrt --warmup-updates 500 -vmt dia_acc -vmm max
```
You may require a GPU to train a model to convergence in a reasonable amount of time.

## Load a pretrained model
To load a model that's already been trained, repeat the previous command but with the model-file flag (`-mf`) pointing to an existing model path.

## Chat with a trained model
To chat with a trained model, set `--interactive=1`
