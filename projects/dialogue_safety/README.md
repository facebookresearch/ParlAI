# Build it Break it Fix it for Dialogue Safety: Robustness from Adversarial Human Attack

## Paper information
Emily Dinan, Samuel Humeau, Bharath Chintagunta, Jason Weston.
_[Build it Break it Fix it for Dialogue Safety: Robustness from Adversarial Human Attack](https://arxiv.org/abs/1908.06083)_.
To appear at EMNLP 2019.

## Abstract
The detection of offensive language in the context of a dialogue has become an increasingly important application of natural language processing. The detection of trolls in public forums (Galán-García et al., 2016), and the deployment of chatbots in the public domain (Wolf et al., 2017) are two examples that show the necessity of guarding against adversarially offensive behavior on the part of humans. In this work, we develop a training scheme for a model to become robust to such human attacks by an iterative build it, break it, fix it strategy with humans and models in the loop. In detailed experiments we show this approach is considerably more robust than previous systems. Further, we show that offensive language used within a conversation critically depends on the dialogue context, and cannot be viewed as a single sentence offensive detection task as in most previous work. Our newly collected tasks and methods will be made open source and publicly available.

## Data examples

To view the data for round 1 of the single turn **adversarial** data, try running:
```bash
python examples/display_data.py -t dialogue_safety:adversarial --round 1
```

To view the data for rounds 1-3 of the single turn **standard** data, try running:
```bash
python examples/display_data.py -t dialogue_safety:standard --round 3
```

To view the data for *only* rounds 2 of the single turn **adversarial** data, try running:
```bash
python examples/display_data.py -t dialogue_safety:adversarial --round 2 --round-only True
```

To view the data from the **multi-turn adversarial** setting, try running:
```bash
python examples/display_data.py -t dialogue_safety:multiturn
```

To view the same data (the **multi-turn adversarial** data), but without the context text provided, try running:
```bash
python examples/display_data.py -t dialogue_safety:multiturn --single-turn True
```

## Pre-trained models

Evaluate a pre-trained Transformer-based model on the test sets of rounds 1-3 or the **adversarial** task:
```bash
python examples/eval_model.py -t dialogue_safety:adversarial --round 3 -dt test -mf zoo:dialogue_safety/single_turn/model -bs 40
```

Evaluate the same pre-trained Transformer-based model on the test sets of rounds 1-3 or the **standard** task:
```bash
python examples/eval_model.py -t dialogue_safety:standard --round 3 -dt test -mf zoo:dialogue_safety/single_turn/model -bs 40
```

Evaluate a BERT-based model (using *segments* to distinguish the context from the utternace to be classified, as described in the paper) on the test set of the **multi-turn adversarial** task:
```bash
python examples/eval_model.py -t dialogue_safety:multiturn -dt test -mf zoo:dialogue_safety/multi_turn/model --split-lines True -bs 40
```

## Training examples

Train a
