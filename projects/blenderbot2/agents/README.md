# BlenderBot 2 Agent Code: FAQ

Below, we have compiled a list of FAQs regarding the usage of BlenderBot2. Please open an issue if your issue is not addressed below.

### `ModuleNotFoundError: No module named 'transformers'`

Please run `pip install transformers==4.3.3`.

### `ModuleNotFoundError: No module named 'parlai.zoo.bart.bart_large`
Please make sure you have [installed fairseq](https://github.com/pytorch/fairseq#requirements-and-installation)

### `ModuleNotFoundError: No module named 'parlai.zoo.blenderbot2'`
If you have installed ParlAI from source, make sure you have pulled from the main branch. If you have installed via pip, make sure you are on version 1.4.0 or later.

### `ValueError: Must provide a valid server for search`
You'll need to setup your own search server; see discussion in [#3816](https://github.com/facebookresearch/ParlAI/issues/3816)

### `AssertionError`
```
    assert search_queries
AssertionError
```

Consult "How can I use BlenderBot2 with **only** search" below.

### `IndexError`
```
    File "/home/ParlAI/projects/blenderbot2/agents/modules.py", line 556, in <dictcomp>
    batch_id: memory_vec[batch_id, : num_memories[mem_id]]
IndexError: index 3 is out of bounds for dimension 0 with size 3
```
Make sure you're providing memories appropriately to the model; the field in which these are emitted is specified by the `--memory-key`. If you want to extract memories from the full context, set `--memory-key full_text`

```
    File "/home/ParlAI/projects/blenderbot2/agents/modules.py", line 556, in <dictcomp>
    for batch_id, mem_id in enumerate(indices)
IndexError: index 1 is out of bounds for dimension 0 with size 1
```
If you notice this during training, please set `--memory-decoder-model-file ''`.

### How can I use BlenderBot2 with **only** search?

You'll need to do two things:

1. Set `--knowledge-access-method search_only`
2. Set `--query-generator-model-file zoo:sea/bart_sq_gen/model`

### How can I manually provide documents for my model to use?

You may want to provide the model with "Gold" documents, or documents on which the generator *should condition*. We offer two ways of doing this:
1. **Only Gold**: Providing the model with **only** the documents you want, bypassing search.
2. **Insert Gold**: Mixing retrieved documents with gold documents.

This may be useful both for ensuring reproducibility between experiments for retrievers with stochastic responses (e.g., the internet). For example, see the [`WizIntGoldDocRetrieverFiDAgent`](https://github.com/facebookresearch/ParlAI/blob/f1a46aad3dbae55f8a7f8aaa70b2330135c23e35/parlai/agents/fid/fid.py#L374) model with wizard of internet dataset).

**ONLY GOLD DOCS**

To do this with BlenderBot2, you'll need to do a few things:

1. Setup a teacher/task such that gold documents are provided in the output `Message`s from the task. Suppose these are in the `gold_document` field of the `Message`.
2. Subclass [`GoldDocRetrieverFiDAgent`](https://github.com/facebookresearch/ParlAI/blob/6380ad53ba74d88280a336ef5b74bce513fcdccf/parlai/agents/fid/fid.py#L326) and implement the `get_retrieved_knowledge` method. This method should return a list of gold documents to consider. See `WizIntGoldDocRetrieverFiDAgent` for how this is done with the Wizard of the Internet dataset.
3. Create a Gold Document Retriever BB2 agent, [like so](https://github.com/facebookresearch/ParlAI/blob/6380ad53ba74d88280a336ef5b74bce513fcdccf/projects/blenderbot2/agents/blenderbot2.py#L897).
4. Specify `--model projects.blenderbot2.agents.blenderbot2:MyGoldDocAgent` in the train script.

**INSERT GOLD DOCS**

If you would like to simply insert gold documents among retrieved/searched documents, you'll need to do design your dataset such that you emit `Message`s with a 3 fields containing the following:
- gold documents: this is the gold retrieved passages. Specify the Message key for the model with `--gold-document-key`
- gold sentences: this is the golden selected sentence, if applicable. make sure to put something here; it's not super necessary, but the code currently requires it. Specify the key with `--gold-sentence-key`
- gold titles: the titles of the retrieved documents. Specify the key with `--gold-document-titles-key`

Then simply set `--insert-gold-docs True` and you're all set.

### I am attempting to use the ParlAI chat services for running BlenderBot2. What should my config look like?
Your config should look like to the following:
```
tasks:
  default:
    onboard_world: MessengerBotChatOnboardWorld
    task_world: MessengerBotChatTaskWorld
    timeout: 1800
    agents_required: 1
task_name: chatbot
world_module: parlai.chat_service.tasks.chatbot.worlds
overworld: MessengerOverworld
max_workers: 1
opt: # Additional model opts go here
  debug: True
  models:
    blenderbot2_400M:
      model: projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent
      model_file: zoo:blenderbot2/blenderbot2_400M/model
      interactive_mode: True
      no_cuda: True
      search_server: <SEARCH_SERVER>
      override:
        search_server: <SEARCH_SERVER>
additional_args:
  page_id: 1 # configure your own page
```
Additionally, any overrided flags you would normally specify on the command line should not only go under `blenderbot2_400M` but **ALSO** under the override key. Finally, make sure that the `MODEL_KEY` variable in parlai/chat_service/tasks/chatbot/worlds.py is set to `blenderbot2_400M`


### How can I see what the model is writing to long term memory, and what the model is using to generate its responses?

Set `--loglevel debug` to see more in depth logging for the model.

### How can I write to the long-term memory before starting the conversation?

It depends on your use case.

BB2 can either extract memories from a special field in the `Message`, or from the dialogue history. If the latter, you'll need to specify `--memory-key full_text` (the zoo models use `personas`). Then, the `--memory-extractor-phrase` is what is used to extract memories from the dialogue history; the default is `persona:`, so any line containing that word is extracted as a memory.

So, if you'd like to use the dialogue history, just have BB2 observe several lines of "memories" prior to the first message.

### How can I disable using the search server every turn?

You can specify the `--knowledge-access-method` to avoid web searches; the following is taken from the [parameter definition](https://github.com/facebookresearch/ParlAI/blob/7506a84e00e0ba526dca01b8aea97d009c91fa50/projects/blenderbot2/agents/blenderbot2.py#L183-L193)

- `classify` => classify the input text, determine which knowledge to access
- `memory_only` => only access memories
- `search_only` => only access search
- `all` => for each input, access from memories and search
- `none` => do not access any knowledge.
