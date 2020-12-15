# *I like fish :fish:, especially dolphins :dolphin::*<sup>[∗](#dolphion)</sup> Addressing Contradictions in Dialogue Modelling

A study on *contradiction* detection and *non-contradiction* generation in dialogue modelling.  
The paper can be found in [Nie et al. (2020)]().

## Abstract

To quantify how well natural language understanding models can capture consistency in a general conversation, we introduce the **D**ialogu**E** **CO**ntradiction **DE**tection task (**DECODE**) and a new conversational dataset containing both human-human and human-bot contradictory dialogues. We then compare a structured utterance-based approach of using pre-trained Transformer models for contradiction detection with the typical unstructured approach. 

Results reveal that:
1. our newly collected dataset is notably more effective at providing supervision for the dialogue contradiction detection task than existing NLI data including those aimed to cover the dialogue domain; 
2. the structured utterance-based approach is more robust and transferable on both analysis and out-of-distribution dialogues than its unstructured counterpart.  

We also show that our best contradiction detection model correlates well with human judgements and further provide evidence for its usage in both automatically evaluating and improving the consistency of state-of-the-art generative chatbots.

## Data

The dataset (DECODE) can be download in [s3_link]().  
As described in the paper, DECODE includes 6 groups of dialogues: *Train*, *Dev*, *Test*, *Human-Bot*, *A2T*, *RCT*.

| Group Name    | Count         | Description  |
| ------------- |---------------| -------------|
| *Train*       | 27,184     | Human-written dialogues |
| *Dev*         | 4,026      | Human-written dialogues |
| *Test*        | 4,216      | Human-written dialogues |
| *Human-Bot*   | 764        | Human-Bot interaction dialogues |
| *A2T*         | 2,079      | Auxiliary test set created by transforming examples in *Test* |
| *RCT*         | 2,011      | Auxiliary test set created by transforming examples in *Test* |

The details of each group can be found in the [paper]().

### Format

```
{
    # Primary field needed for contradiction detection task.
    "conversation_id": "9cb462d9-86f1-4296-af36-009d2e4d90f8#truncated#4",
    "record_id": "1f47fe86-cfc3-469a-bae3-506c81871bf5",

    "turns": [
        {"turn_id": 0, "agent_id": 0, "text": "i've been to new york city once crazy place that city .", "turn_context": "", "auxiliary": {"contradiction": null}}, 
        {"turn_id": 1, "agent_id": 1, "text": "i wish i could go there . i'm sure they have a place with great meatloaf !", "turn_context": "", "auxiliary": {"contradiction": null}}, 
        {"turn_id": 2, "agent_id": 0, "text": "They probably do, somewhere! You can find nearly any cuisine there you want.", "turn_context": "", "auxiliary": {"contradiction": null}}, 
        {"turn_id": 3, "agent_id": 1, "text": "I wonder if they have anything different, I wonder if anyone has tried to make meatloaf with tofu instead.", "turn_context": "", "auxiliary": {"contradiction": null}}, 
        {"turn_id": 4, "agent_id": 0, "text": "I'm sure somebody has, though I am not sure how it would taste.", "turn_context": "", "auxiliary": {"contradiction": null}}, 
        {"turn_id": 5, "agent_id": 1, "text": "I make tofu meatloaf all the time, it is delicious", "turn_context": "", "auxiliary": {"contradiction": true}}], 

    "is_contradiction": true, 
    "aggregated_contradiction_indices": [3, 5], 
       
    # Other collection related field.
    "num_of_turns_by_writer": 2
    "writer_contradiction_indices": [3, 5], 
    "verifications": [
        {"verification_id": "36236842-98fc-495d-9c04-182f1b77c246", "is_contradiction": true, "verifier_contradiction_indices": [3, 5]}, 
        {"verification_id": "fd25e3f4-7366-42a5-aed5-0d20082cc833", "is_contradiction": true, "verifier_contradiction_indices": [3, 5]}, 
        {"verification_id": "c1648e0f-cd8d-408c-89b5-54a8dc7f522b", "is_contradiction": true, "verifier_contradiction_indices": [3, 5]}],
    
    # Other field you normally wouldn't need.
    "agents": {
        "1": {"is_human": true, "persona_lines": []}, 
        "0": {"is_human": true, "persona_lines": []}
    }, 
    "conversation_contexts": null, 
    "is_truncated": true,
    "auxiliary": {   
        "workers": ["A248D5XVN1YGCZ", "A2RBVMD2HTBCZ9"], 
        "assignment_ids": ["3TAYZSBPLO5EAUXXRU24479226QS2T", "3SNVL38CI7PTKTCRJEI8PYREX3QCK5"], 
        "hit_ids": ["344M16OZKKC72DJEMGBHJUCXCGTNET", "35XW21VSVIBIOWLBBYF7VJCN8C2LSQ"], 
        "source": "BST_test"
    },
}
```



_________________
<a name="dolphion"><sup>∗</sup></a> Dolphins :dolphin: are mammals, not fish :fish:.