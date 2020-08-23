# IR Baseline

This agent is a simple information retrieval baseline.
If `label_candidates` are provided in the task, it simply ranks them according to their similarity to the input message. To use TF-IDF similarity, you must build a dictionary and "train" on the train set first. See the examples below for clarification.


## Basic Examples

Evaluate the IR baseline model (without using TF-IDF) on the Persona-Chat task.
```bash
parlai eval_model -m ir_baseline -t personachat
```

Build a dictionary on the Persona-Chat task, and then evaluate the model using TF-IDF.
```bash
parlai train_model -m ir_baseline -t personachat --dict-file /tmp/personachat.dict -ttim 5
```
