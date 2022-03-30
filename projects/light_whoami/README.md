# Am I Me or You? State-of-the-Art Dialogue Models Cannot Maintain Identity

Kurt Shuster, Jack Urbanek, Arthur Szlam, Jason Weston

## Abstract

State-of-the-art dialogue models still often stumble with regards to factual accuracy and self-contradiction. Anecdotally, they have been observed to fail to maintain character identity throughout discourse; and more specifically, may take on the role of their interlocutor. In this work we formalize and quantify this deficiency, and show experimentally through human evaluations that this is indeed a problem. In contrast, we show that discriminative models trained specifically to recognize who is speaking can perform well; and further, these can be used as automated metrics. Finally, we evaluate a wide variety of mitigation methods, including changes to model architecture, training protocol, and decoding strategy. Our best models reduce mistaken identity issues by nearly 65% according to human annotators, while simultaneously improving engagingness. Despite these results, we find that maintaining character identity still remains a challenging problem.


## Paper

[Link to arXiv](https://arxiv.org/abs/2112.05843)

## Tasks

### RPA Classifier Training

Full Datasplit:

    parlai dd -t projects.light_whoami.task.agents:WhoIsSpeakingTeacher

Left to Right:

    parlai dd -t projects.light_whoami.task.agents:WhoIsSpeakingLeftToRightTeacher

### RPA Evaluation of Model Responses

    parlai dd -t projects.light_whoami.task.agents:ResponseClassifierTeacher

### Multi-Objective Training

    parlai dd -t projects.light_whoami.task.agents:MultiObjectiveTeacher


## Agent Code

**NOTE**: Each agent specified below can be used in tandem with the [long-context generator agents](https://github.com/facebookresearch/ParlAI/blob/89e8c323090a9ae48552b78518ea3b553474722f/projects/msc/agents/long_tga.py) from the [MSC project](https://parl.ai/projects/msc/) by simply adding `Long` in front of the final agent name. E.g., `projects.light_whoami.agents.rpa_rerank:RPARerankAgent` becomes projects.light_whoami.agents.rpa_rerank:LongRPARerankAgent`, and so on.

### RPA Re-ranker Agents

These agents will re-rank beams from the base model according to RPA score. One must specify a `--predictor-model-file` pointing to an RPA Classifier.

    parlai i -m projects.light_whoami.agents.rpa_rerank:RPARerankAgent \
    -mf <path_to_model> --predictor-model-file <path_to_predictor_model>


If you'd like to use a predictor model file other than that used for RPA re-ranking, please see instructions [here](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents/reranker/) for how to implement your own re-ranker. Then, subclass the [`AbstractGeneratorRerankAgent`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents/reranker/reranker.py), implementing the `get_reranker_class` method to point to your re-ranker.

### PACER Agents

In addition to re-ranking the final beams according to RPA score, these models will apply ranking on _partial sequences_. Use the following parameters to control this level of ranking:

- `--pacer-n-tokens`: How many tokens to consider when rescoring on partial sequences
- `--pacer-frequency-ratio`: How often to apply PACER re-ranking when decoding.

    parlai i -m projects.light_whoami.agents.pacer:PacerAgent \
    -mf <path_to_model> --predictor-model-file <path_to_predictor_model_file>

If you'd like to use a predictor model file other than that used for RPA Re-Ranking, simply subclass the `PacerAgent` and implement `get_reranker_class()` to return your constructed re-ranker object (see steps [here](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents/reranker/)).

### Unlikelihood Agents

One can apply RPA Unlikelihood in order to discourage the agent from generating tokens that yield the wrong predicted speaker. This agent requires a predictor model file as well. The following parameters are important for controlling training:

- `--ul-top-k-toks`: How many tokens to apply the UL loss to.
- `--only-wrong-class-toks`: Set `True` to only apply UL loss to tokens that yield the wrong predicted speaker.
- `--all-wrong-class-toks`: Set `True` to apply UL loss to all tokens in an utterance that result in the wrong predicted speaker.

    parlai train_model -m projects.light_whoami.agents.rpa_ul:RpaUlAgent \
    --predictor-model-file <path_to_predictor_model_file> \
    --init-model <path_to_init_model> ...

### Multi-Objective Agents

One can utilize the multi-objective agents to train both the generator NLL loss and a character prediction ranking loss. Important parameters:

- `--n-multiobjective-layers/heads`: Specify number of layers/heads to use as additional components in predicting the speaker.
- `--multiobjective-latent-representation`: One of `['encoder_final_layer', 'decoder_final_layer', 'encoder_and_decoder']`, sets which representations to use when predicting the speaker.

    parlai train_model -m projects.light_whoami.agents.multi_objective:MultiObjectiveGeneratorAgent \
    --init-model <path_to_init_model> ...


### Profile Expanded Decoder Attention

Use these agents in an "expanded" attention scenario, where a portion of the input (or something otherwise specified) is attended to in a third round of attention in the decoder (following self-attention and encoder-attention). The following parameters are useful:

*To set the context from which to pull expanded attention input*
- `--expanded-attention-input-key`: Key in the teacher message to pull from for expanded attention
- `--expanded-attention-input-extractor-phrases`: If specified, the input for expanded attention will consist only of pieces of the delimited input that contain these phrases.
- `--expanded-attention-num-rounds`: How many rounds to apply the expanded attention.

    parlai train_model -m projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAgent \
    --init-model <path_to_init_model> ...

### Automated Expanded Decoder Attention

To automatically learn what to re-attend to within the context, you can use the same agent as above, but specify `--expanded-attention-type <automated_classifier/automated_trainable_mask>`. For `automated_trainable_mask`, there are no additional parameters required. For `automated_classifier`, one must specify the `--predictor-model-file` as before.

### Automated Expanded Decoder Attention + Multi-Objective Training

To leverage multi-objective training within an automated expanded attention scenario, simply set `--expanded-attention-type automated_trainable_mask`, and the proper agent, along with any desired multi-objective arguments from above:

    parlai train_model -m projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAndMultiObjectiveAgent \
    --expanded-attention-type automated_trainable_mask --init-model <path_to_init_model> \
    ...

### Expanded Decoder Attention + RPA Re-ranking / PACER

The following agents combine expanded decoder attention with RPA Re-Ranking or PACER Re-Ranking functionality:

    parlai i -m projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAndRPARerankerAgent \
    --model-file <path_to_expanded_attention_agent> --predictor-model-file <path_to_predictor_model_file >...

    parlai i -m projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAndPacerAgent \
    --model-file <path_to_expanded_attention_agent> --predictor-model-file <path_to_predictor_model_file >...

## Pre-Trained Models

The following table provides the zoo paths for the released pre-trained models (used in `--model-file` or `--init-model`):

Model | RPA | Mistaken Identity | Zoo Path |
------|------------------------:| ------------------------:|------------------------:|
LTR RPA Re-Ranker | - | - | zoo:light_whoami/rpa_reranker/model |
128-Truncate Vanilla Baseline | 87.61 | 6.45% | zoo:light_whoami/vanilla_128/model |
1024-Truncate Vanilla Baseline | 87.71 | 7.35% | zoo:light_whoami/vanilla_1024/model |
128-Truncate RPA Unlikelihood (Top1) | 87.48 | 7.13% | zoo:light_whoami/rpa_ul_128/model |
1024-Truncate RPA Unlikelihood (Top1) | - | - | zoo:light_whoami/rpa_ul_1024/model |
Multi-Objective (Vanilla, Dec. Only) | 87.67 | 10.00% | zoo:light_whoami/multiobjective/model |
Profile Expanded Attention (128, 2 rounds over ABC) | 91.70 | 4.82% | zoo:light_whoami/profile_expanded_attention_128/model |
Profile Expanded Attention (1024, 2 rounds over ABCD) | 92.18 | 4.00% | zoo:light_whoami/profile_expanded_attention_1024/model |
Automated Expanded Attention (1024, Classifier Attn.) | 90.93 | 5.51% | zoo:light_whoami/automated_expanded_attention_1024/model |
Automated Expanded Attention + Multi-Objective (1024, Dec. Only) | 88.95 | 4.43% | zoo:light_whoami/expanded_and_multiobjective_1024/model |