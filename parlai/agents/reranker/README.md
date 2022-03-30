# Output Reranker

This agent provides an abstract implementation of a "re-ranker" object, as well as an abstract implementation of a `TransformerGeneratorAgent` that utilizes the re-ranker. A re-ranker can be used to re-rank outputs according to some other model's predictor score. The below steps outline how to build a re-ranker for your task.

## How to build your own re-ranker.

### 1. Train a classifier or ranker model.

The first step is to train a model -- e.g. `transformer/biencoder`, `transformer/polyencoder`, or `transformer/classifier` -- on a desired classification or ranking task.

### 2. Subclass `AbstractReranker`

To create your task-specific re-ranker, you can subclass the `AbstractReranker` in `reranker.py`, and implement the following methods:

- `get_class_to_rerank_for(observation: Message, full_context: str) -> str` --> This function will return the target class that the re-ranker should aim to maximize. In a contradiction setting, this might be `entails`.
- `is_context(utt: str) -> bool` --> This function will return whether a given utterance is an element of the "context" given to a model. This varies for different tasks; for example, in ConvAI2, this may return `True` for an utterance with "your persona: ..."; for LIGHT, this would return `True` for any utterance describing the setting or the characters.
- `get_predictor_label_candidates(observation: Message, context: str) -> List[str]` --> This function will return the candidates the re-ranker must rank/classify, given an incoming context and observation.

### 3. Subclass `AbstractGeneratorRerankAgent`

Finally, subclass the `AbstractGeneratorRerankAgent` in `reranker.py`, and implement one method:

- `get_reranker_class()` --> This method returns the class for the re-ranker.


## Case study: Classifier Re-Ranking.

If you want, you can use a standard classifier for re-ranking, where the classifier takes
the candidate outputs and chooses based on maximizing the probability of a given provided class.      
This is already implemented in classifier_reranker.py in this directory,
which can thus be used via the flags "-m reranker/classifier_reranker --target-label positive_class_name".


## Case study: LIGHT RPA Re-Ranking.

### 1. Train a classifier or ranker model.

For the LIGHT RPA Re-ranking task, the goal is to train a classifier than can predict which character is speaking in conversation. To do so, we train a poly-encoder on the RPA task:

    parlai train_model \
    -m transformer/polyencoder --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model \
    -t projects.light_whoami.task.agents.WhoIsSpeakingLeftToRightTeacher ...

### 2. Subclass `AbstractReranker`.

In [this file](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents/reranker/reranker.py), we implement the `RPAReranker` object, subclassing `AbstractReranker`.

- `get_class_to_rerank_for` --> this extracts the self character from the context.
- `is_context` --> returns `True` for any line starting with `_` (indicating LIGHT context)
- `get_predictor_label_candidates` --> extracts character names from the conversation, and returns the list.

### 3. Subclass `AbstractGeneratorRerankAgent`

In the same file as above, we implement `RPARerankAgent`, which only implements the `get_reranker_class` to return `RPAReranker` as built in step 2.
