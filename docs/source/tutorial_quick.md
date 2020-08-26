ParlAI Quick-start
==================

**Authors**: Alexander Holden Miller, Margaret Li

Colab Tutorial
--------------

As an alternative to this quick start tutorial, you may also consider our
[Google Colab tutorial](https://colab.research.google.com/drive/1bRMvN0lGXaTF5fuTidgvlAl-Lb41F7AD#scrollTo=KtVz5dCUmFkN),
which takes you through fine-tuning the small version of
[BlenderBot](https://parl.ai/projects/recipes/) (90M).


Install
-------

First, make sure you have Python 3. Now open up terminal and run the
following.

1.  Clone ParlAI Repository:

```bash
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
```

2.  Install ParlAI:

```bash
cd ~/ParlAI; python setup.py develop
```

This will add the parlai command to your system.

3.  Several models have additional requirements, such as
    [PyTorch](http://pytorch.org/).

View a task & train a model
---------------------------

Let's start by printing out the first few examples of the bAbI tasks,
task 1.

```bash
# display examples from bAbI 10k task 1
parlai display_data -t babi:task10k:1
```

Now let's try to train a model on it (even on your laptop, this should
train fast).

```bash
# train MemNN using batch size 1 and for 5 epochs
parlai train_model -t babi:task10k:1 -mf /tmp/babi_memnn -bs 1 -eps 5 -m memnn --no-cuda
```

Let's print some of its predictions to make sure it's working.

```bash
# display predictions for model save at specified file on bAbI task 1
parlai display_model -t babi:task10k:1 -mf /tmp/babi_memnn -ecands vocab
```

The "eval\_labels" and "MemNN" lines should (usually) match!

Let's try asking the model a question ourselves.

```bash
# interact with saved model
parlai interactive -mf /tmp/babi_memnn -ecands vocab
...
Enter your message: John went to the hallway.\n Where is John?
```

Hopefully the model gets this right!

Train a Transformer on Twitter
------------------------------

Now let's try training a Transformer (Vaswani, et al 2017) ranker model.
*Make sure to complete this section on a GPU with PyTorch installed.*

We'll be training on the Twitter task, which is a dataset of tweets and
replies. There's more information on tasks in these docs, including a
full list of [tasks](http://parl.ai/docs/tasks.html) and
[instructions](http://parl.ai/docs/tutorial_basic.html#training-and-evaluating-existing-agents)
on specifying arguments for training and evaluation (like the
`-t <task>` argument used here).

Let's begin again by printing the first few examples.

```bash
# display first examples from twitter dataset
parlai display_data -t twitter
```

Now, we'll train the model. This will take a while to reach convergence.

```bash
# train transformer ranker
parlai train_model -t twitter -mf /tmp/tr_twitter -m transformer/ranker -bs 16 -vtim 3600 -cands batch -ecands batch --data-parallel True
```

You can modify some of the command line arguments we use here -we set
batch size to 10, run validation every 3600 seconds, and take candidates
from the batch for training and evaluation.

The train model script will by default save the model after achieving
best validation results so far. The Twitter task is quite large, and
validation is run by default after each epoch (full pass through the
train data), but we want to save our model more frequently so we set
validation to run once an hour with `-vtim 3600`.

This train model script evaluates the model on the valid and test sets
at the end of training, but if we wanted to evaluate a saved model
-perhaps to compare the results of our newly trained Transformer against
the BlenderBot 90M baseline from our [Model
Zoo](http://parl.ai/docs/zoo.html), we could do the following:

```bash
# Evaluate the tiny BlenderBot model on twitter data
parlai eval_model -t twitter -mf zoo:blender/blender_90M/model
```

Finally, let's print some of our transformer's predictions with the same
display\_model script from above.

```bash
# display predictions for model saved at specific file on twitter
parlai display_model -t twitter -mf /tmp/tr_twitter -ecands batch
```

Add a simple model
------------------

Let's put together a super simple model which will print the parsed
version of what is said to it.

First let's set it up.

```bash
mkdir parlai/agents/parrot
touch parlai/agents/parrot/parrot.py
```

We'll inherit the TorchAgent parsing code so we don't have to write it
ourselves. Open parrot.py and copy the following:

```python
from parlai.core.torch_agent import TorchAgent, Output

class ParrotAgent(TorchAgent):
    def train_step(self, batch):
        pass

    def eval_step(self, batch):
        # for each row in batch, convert tensor to back to text strings
        return Output([self.dict.vec2txt(row) for row in batch.text_vec])

    def build_model(self, batch):
        # Our agent doesn't have a real model, so we will return a placeholder
        # here.
        return None
```

Now let's test it out:

```bash
parlai display_model -t babi:task10k:1 -m parrot
```

You'll notice the model is always outputting the "unknown" token. This
token is automatically selected because the dictionary doesn't recognize
any tokens, because we haven't built a dictionary yet. Let's do that
now.

```bash
parlai build_dict -t babi:task10k:1 -df /tmp/parrot.dict
```

Now let's try our Parrot agent again.

```bash
parlai display_model -t babi:task10k:1 -m parrot -df /tmp/parrot.dict
```

This ParrotAgent implements `eval_step`, one of two abstract functions
in TorchAgent. The other is `train_step`. You can easily and quickly
build a model agent by creating a class which implements only these two
functions with the most typical custom code for a model, and inheriting
vectorization and batching from TorchAgent.

As needed, you can also override any functions to change the default
argument values or to override the behavior with your own. For example,
you could change the vectorizer to return numpy arrays instead of Torch
Tensors.

Conclusion
----------

To see more details about ParlAI's general structure, how tasks and
models are set up, or how to use Mechanical Turk, Messenger,
Tensorboard, and more --check out the other tutorials.
