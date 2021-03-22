# Agent exported to TorchScript (JIT compilation)

This agent will read in a ParlAI agent that has been exported to TorchScript with JIT compilation, for use in greedy-search inference on CPU. This allows inference to be run on models without using any ParlAI overhead, either for tokenization or for the forward passes through the model. Currently, only BART models are supported.

Sample call for exporting a BART model to TorchScript:
```
parlai torchscript \
--model-file ${MODEL_FILE} \
--model bart \
--no-cuda \
--scripted-model-file ~/_test_scripted_model__bart.pt \
--input 'I am looking for a restaurant in the west part of town.|APIRESP: Restaurant 14 matches'
```

Interacting with an exported model using `parlai interactive`:
```
parlai interactive \
--model-file ~/_test_scripted_model__bart.pt \
--model parlai.torchscript.agents:TorchScriptAgent
```

Loading in and running inference on an exported model, without any ParlAI overhead:
```
python parlai/torchscript/scripts/test_exported_model.py \
--scripted-model-file ~/_test_scripted_model__bart.pt \
--input 'I am looking for a restaurant in the west part of town.|APIRESP: Restaurant 14 matches'
```
