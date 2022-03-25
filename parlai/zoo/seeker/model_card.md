# SeeKeR Dialogue 3B



SeeKeR Dialogue model; trained to search the internet, synthesize knowledge, and produce a dialogue response.
- Developed by Facebook AI Research using [ParlAI](https://parl.ai/)
-  Model started training on February 09, 2022.
- Type of model: projects.seeker.agents.Seeker:SeekerAgent

### Quick Usage


```
parlai i -mf zoo:seeker/seeker_dialogue_3B/model -o gen/seeker_dialogue --search-server <search_server>
```

### Sample Input And Output

```
Enter Your Message: Hey, what you can tell me about ai research?
[ComboFidSearchQuery]: The study of intelligent agents and how they perceive their environment is called AI research. What do you want to know about it?
```

## Intended Use

This model is intended for research purposes.

## Limitations

Our language models suffer the same issues as other  systems that exist today, specifically with problems of occasional inconsistency, contradictions, factual inaccuracies, potential repetition, and lack of deeper reasoning, amongst other issues. Further, generations can include toxic language and bias, especially with certain contexts and topics. Additionally, documents from the internet influence our generations, which can be a problem if undesirable content is retrieved.


## Datasets Used

This model was trained on the datasets below (use the `parlai display_data` commands to show data). Visit the [task (dataset) list](https://parl.ai/docs/tasks.html) for more details about the datasets.

- Wizard of the Internet
- Wizard of Wikipedia
- PersonaChat
- Empathetic Dialogues
- Blended Skill Talk
- Multi-Session Chat
- MS MARCO
- Natural Questions
- SQuAD
- TriviaQA


## Evaluation Results

This model was evaluated on the datasets below (use the `parlai display_data` commands to show data). Visit the [task (dataset) list](https://parl.ai/docs/tasks.html) for more details about the datasets.

- [Wizard_of_Internet](https://parl.ai/docs/tasks.html#wizard_of_internet): A dataset with conversations directly grounded with knowledge retrieved from internet. One of the participants has access to internet search. The other side has an assigned persona that provides the topic of the conversation. Contains 93.7k utterances from 9.6k conversations, split into train, test, and valid sets.


We used the metric `ppl` as the validation metric. Recall that `ppl` is perplexity. Click [here](https://en.wikipedia.org/wiki/Perplexity) for more info.

|  | Wizard_of_Internet
:---: | :---:
`ppl` | 16.6771



## Related Paper(s)

[Language Models that Seek for Knowledge: Modular Search & Generation for Dialogue and Prompt Completion](https://arxiv.org/abs/2203.13224). Kurt Shuster, Mojtaba Komeili, Leonard Adolphs, Stephen Roller, Arthur Szlam, Jason Weston.

## Hyperparameters



- `lr_scheduler`: ` reduceonplateau `
- `batchsize`: ` 1 `
- `learningrate`: ` 1e-06 `
- `model`: ` projects.seeker.agents.seeker:ComboFidGoldDocumentAgent `
- `validation_patience`: ` Not specified `
- `validation_metric`: ` Not specified `
- `multitask_weights`: ` 2,2,1,1 `
- `max_train_steps`: ` Not specified `
- `num_epochs`: ` Not specified `
<details>
 <summary> model / neural net info </summary>
 <br>

- `n_layers`: ` 22 `
- `ffn_size`: ` 8192 `
- `dropout`: ` 0.1 `
- `n_heads`: ` 32 `
- `n_positions`: ` 1024 `
- `variant`: ` prelayernorm `
- `activation`: ` gelu `
- `output_scaling`: ` 1.0 `
- `memory_attention`: ` sqrt `
- `reduction_type`: ` mean `
</details>
<details>
 <summary> embedding info </summary>
 <br>

- `retriever_embedding_size`: ` 768 `
- `embedding_projection`: ` random `
- `embedding_type`: ` random `
- `learn_positional_embeddings`: ` True `
- `embedding_size`: ` 2048 `
- `learn_embeddings`: ` True `
- `share_word_embeddings`: ` True `
- `embeddings_scale`: ` True `
</details>
<details>
 <summary> dictionary info/pre-processing </summary>
 <br>

- `dict_maxtokens`: ` -1 `
- `dict_max_ngram_size`: ` -1 `
- `dict_unktoken`: ` __unk__ `
- `dict_endtoken`: ` __end__ `
- `dict_tokenizer`: ` gpt2 `
- `dict_class`: ` parlai.core.dict:DictionaryAgent `
- `dict_starttoken`: ` __start__ `
- `cap_num_predictions`: ` 100 `
- `dict_textfields`: ` text,labels `
- `dict_nulltoken`: ` __null__ `
- `dict_language`: ` english `
</details>
<details>
 <summary> other dataset-related info </summary>
 <br>

- `truncate`: ` 1000 `
- `text_truncate`: ` 1000 `
- `label_truncate`: ` 128 `
- `split_lines`: ` True `
- `task`: ` projects.seeker.tasks.knowledge:KnowledgeTeacher,projects.seeker.tasks.knowledge:DialogueTeacher,projects.seeker.tasks.knowledge:SearchQueryTeacher,projects.seeker.tasks.knowledge:SearchDecisionTeacher `
</details>
<details>
 <summary> more batch and learning rate info </summary>
 <br>

- `encode_candidate_vecs_batchsize`: ` 256 `
- `invsqrt_lr_decay_gamma`: ` -1 `
- `lr_scheduler_decay`: ` 0.5 `
- `lr_scheduler_patience`: ` 3 `
</details>
<details>
 <summary> training info </summary>
 <br>

- `optimizer`: ` adamw `
- `gradient_clip`: ` 1.0 `
- `adam_eps`: ` 1e-08 `
- `nesterov`: ` True `
- `nus`: ` [0.7] `
- `betas`: ` [0.9, 0.999] `
- `warmup_updates`: ` 100 `
- `warmup_rate`: ` 0.0001 `
- `update_freq`: ` 1 `
- `fp16`: ` True `
</details>
<details>
 <summary> miscellaneous </summary>
 <br>

- `splitted_chunk_length`: ` 256 `
- `rag_model_type`: ` token `
- `gold_knowledge_passage_key`: ` checked_sentence `
- `max_doc_token_length`: ` 256 `
- `compressed_indexer_factory`: ` IVF4096_HNSW128,PQ128 `
- `beam_context_block_ngram`: ` -1 `
- `search_query_generator_beam_size`: ` 1 `
- `temperature`: ` 1.0 `
- `dpr_num_docs`: ` 25 `
- `history_add_global_end_token`: ` end `
- `doc_chunks_ranker`: ` head `
- `n_decoder_layers`: ` 22 `
- `loglevel`: ` info `
- `rag_query_truncate`: ` 512 `
- `topk`: ` 10 `
- `gold_knowledge_title_key`: ` title `
- `min_doc_token_length`: ` 64 `
- `rag_retriever_type`: ` observation_echo_retriever `
- `rag_turn_marginalize`: ` doc_then_turn `
- `hnsw_indexer_store_n`: ` 128 `
- `search_query_generator_text_truncate`: ` 512 `
- `regret_intermediate_maxlen`: ` 32 `
- `codes_attention_type`: ` basic `
- `adafactor_eps`: ` [1e-30, 0.001] `
- `n_ranked_doc_chunks`: ` 1 `
- `hnsw_ef_construction`: ` 200 `
- `beam_size`: ` 1 `
- `indexer_buffer_size`: ` 65536 `
- `poly_attention_num_heads`: ` 4 `
- `image_cropsize`: ` 224 `
- `interactive_candidates`: ` fixed `
- `indexer_type`: ` compressed `
- `candidates`: ` inline `
- `compressed_indexer_nprobe`: ` 64 `
- `beam_block_ngram`: ` -1 `
- `hnsw_ef_search`: ` 128 `
- `doc_chunk_split_mode`: ` word `
- `image_size`: ` 256 `
- `rag_turn_n_turns`: ` 2 `
- `codes_attention_num_heads`: ` 4 `
- `beam_length_penalty`: ` 0.65 `
- `n_encoder_layers`: ` 22 `
- `n_docs`: ` 5 `
- `skip_generation`: ` True `
- `search_query_generator_inference`: ` greedy `
- `tfidf_max_doc_paragraphs`: ` -1 `
- `eval_candidates`: ` inline `
- `poly_n_codes`: ` 64 `
- `encode_candidate_vecs`: ` True `
- `polyencoder_init_model`: ` wikito `
- `poly_score_initial_lambda`: ` 0.5 `
- `beam_min_length`: ` 1 `
- `checkpoint_activations`: ` True `
- `datatype`: ` train `
- `rag_retriever_query`: ` full_history `
- `t5_model_arch`: ` t5-base `
- `inference`: ` greedy `
- `fixed_candidate_vecs`: ` reuse `
- `fp16_impl`: ` safe `
- `poly_attention_type`: ` basic `
- `use_reply`: ` label `
- `beam_block_full_context`: ` True `
- `force_fp16_tokens`: ` True `
- `image_mode`: ` raw `
- `query_model`: ` bert `
- `search_query_generator_beam_min_length`: ` 1 `
- `beam_delay`: ` 30 `
- `rank_top_k`: ` -1 `
- `generation_model`: ` bart `
- `rag_turn_discount_factor`: ` 1.0 `
- `topp`: ` 0.9 `
- `repeat_blocking_heuristic`: ` True `
- `skip_retrieval_key`: ` skip_retrieval `
- `woi_doc_chunk_size`: ` 500 `
- `polyencoder_type`: ` codes `
- `share_encoders`: ` True `
</details>

## Feedback

We would love any feedback about the model (or the model card script)! Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/ParlAI/issues) :blush:


[back-to-top](#seeker-dialogue-3b)
