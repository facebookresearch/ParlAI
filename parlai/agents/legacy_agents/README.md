# Legacy Agents

In this directory, you will find older implementations of agents we have kept for posterity. These agents are deprecated, and we do not make any guarantees of maintaining their functionality - that is, models may not be backwards-compatible with older versions.

## How to use

To use a legacy agent, one must specify the model version along with the keyword ``legacy``, e.g. as below:
```bash
python examples/train_model.py -m legacy:memnn:0
```
