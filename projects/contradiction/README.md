# *I like fish <span>&#x1F41F;</span>, especially dolphins <span>&#x1F42C;</span>:*<sup>[∗](#dolphion)</sup> Addressing Contradictions in Dialogue Modeling

A study on *contradiction* detection and *non-contradiction* generation in dialogue modeling.
The paper can be found here: [Nie et al. (2020)](https://arxiv.org/abs/2012.13391).

## Abstract

To quantify how well natural language understanding models can capture consistency in a general conversation, we introduce the **D**ialogu**E** **CO**ntradiction **DE**tection task (**DECODE**) and a new conversational dataset containing both human-human and human-bot contradictory dialogues. We then compare a structured utterance-based approach of using pre-trained Transformer models for contradiction detection with the typical unstructured approach.

Results reveal that:
<ol>
<li>Our newly collected dataset is notably more effective at providing supervision for the dialogue contradiction detection task than existing NLI data including those aimed to cover the dialogue domain;</li>
<li>The structured utterance-based approach is more robust and transferable on both analysis and out-of-distribution dialogues than its unstructured counterpart.</li>
</ol>

We also show that our best contradiction detection model correlates well with human judgements and further provide evidence for its usage in both automatically evaluating and improving the consistency of state-of-the-art generative chatbots.

## Data
As described in the paper, **DECODE** includes 6 groups of dialogues: (Main) *Train*, (Main) *Dev*, (Main) *Test*, *Human-Bot*, *A2T*, *RCT*.

| Group Name    | Count         | Description  |
| ------------- |---------------| -------------|
| *Train*       | 27,184     | Human-written dialogues |
| *Dev*         | 4,026      | Human-written dialogues |
| *Test*        | 4,216      | Human-written dialogues |
| *Human-Bot*   | 764        | Human-Bot interaction dialogues |
| *A2T*         | 2,079      | Auxiliary test set created by transforming examples in *Test* |
| *RCT*         | 2,011      | Auxiliary test set created by transforming examples in *Test* |

The details of each group can be found in the [Nie et al. (2020)](https://arxiv.org/abs/2012.13391).

## Load Data from ParlAI
The **DECODE** can be loaded directly from ParlAI. The correct arguments to load data belonging to the different subsets is given below.
```
parlai display_data -t decode -dt train -v                          # Train
parlai display_data -t decode -dt valid -v                          # Dev
parlai display_data -t decode -dt test --test_type vanilla -v       # Test
parlai display_data -t decode -dt test --test_type human-bot -v     # Human-Bot
parlai display_data -t decode -dt test --test_type a2t -v           # A2T
parlai display_data -t decode -dt test --test_type rct -v           # RCT
```

## Directly Download Data.
You can also download the data directly from s3.
See [download data from s3 with raw format](https://github.com/facebookresearch/ParlAI/blob/main/projects/contradiction/download_with_raw_format.md).

## Citation
If you use the dataset or models in your own work, please cite with the following BibTex entry:
```
@misc{nie2020i,
      title={I like fish, especially dolphins: Addressing Contradictions in Dialogue Modelling},
      author={Yixin Nie and Mary Williamson and Mohit Bansal and Douwe Kiela and Jason Weston},
      year={2020},
      eprint={2012.13391},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

_________________
<a name="dolphion"><sup>∗</sup></a> Dolphins <span>&#x1F42C;</span> are mammals, not fish <span>&#x1F41F;</span>.
