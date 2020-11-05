# Unique Examples

This is an agent which simply asserts that it has only seen unique examples as a debugging tool.

## Basic Examples

Evaluate on a chunk teacher which returns unique examples.
```bash
parlai eval_model -m unique_examples -t integration_tests:chunky_unique_slow --cant_answer_percent 0.2
```
