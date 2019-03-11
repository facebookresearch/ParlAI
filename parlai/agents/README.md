# Agents

This directory contains a variety of different agents which use ParlAI's interface.

## Utility

- **local_human**: receives human input from the terminal. used for interactive mode, e.g. `parlai/scripts/interactive.py`.
- **legacy_agents**: contains deprecated agent code for posterity
- **random_candidate**: returns a random candidate, if candidates are available. simple baseline agent.
- **remote_agent**: uses ZMQ to communicate with a different process, either on the local machine or on a remote host.
- **repeat_label**: sends back the label if available. good for sanity checks such as checking statistics of the base dataset.
- **repeat_query**: repeats whatever is said to it. simple baseline agent.

## Non-neural agents

- **ir_baseline**: chooses response based on simple word overlap
- **retriever_reader**: used primarily for OpenSquad evaluation. retrieves documents from database and reads them back
- **tfidf_retriever**: returns candidate responses based on tfidf overlap
- **unigram**: returns top unigrams

## Text-based neural networks

- **bert_ranker**: BERT-based bi-ranker and cross-ranker retrieval models
- **drqa**: context-based question answering system
- **fairseq**: provides access to models from FAIR's FairSeq library (github.com/facebookresearch/fairseq)
- **language_model**: simple RNN-based language model
- **memnn**: memory network
- **ibm_seq2seq**: IBM's RNN-based sequence to sequence model
- **seq2seq**: our RNN-based sequence to sequence model
- **starspace**: embedding model
- **transformer**: both generative and retrieval-based transformer models

## Visual (+text) neural networks

- **mlb_vqa**: visual question answering model
- **vsepp_caption**: image captioning model
