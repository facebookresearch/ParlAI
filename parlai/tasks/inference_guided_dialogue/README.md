# Inference Guided Dialogue 

### Data Configurations

There are three main options for training: 

- `infqa_response`: First generate the question given the dialogue history, then generate an answer to it, then generate a response given all the context (dialogue history + question + answer).
- `inq_aresponse`: First generate ta answer to a given dialogue history and inference question, then generate a response given all the context. 
- `response` (default): Generate only the final response given the dialogue history + question + answer. 

Pass these key values as command line arguments via `-gt` or `--generation_target`. 
- i.e. `parlai dd -t inference_guided_dialogue -gt infqa_response` 