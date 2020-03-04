# Fixed Response

This is an agent that simply always outputs the given fixed response.
Good for debugging purposes or as a baseline, e.g. if always predicting a given class.

## Basic Examples

Evaluate on the dialogue safety task, always predicting __notok__
```bash
python examples/eval_model.py  -t dialogue_safety -dt valid -m fixed_response -fr __notok__
```
