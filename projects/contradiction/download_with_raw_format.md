## Directly Download Raw Data

The dataset (**DECODE**) can be download in [this_link](http://parl.ai/downloads/decode/decode_v0.1.zip).
As described in the paper, **DECODE** includes 6 groups of dialogues: *Train*, *Dev*, *Test*, *Human-Bot*, *A2T*, *RCT*.

| Group Name    | Count         | Description  |
| ------------- |---------------| -------------|
| *Train*       | 27,184     | Human-written dialogues |
| *Dev*         | 4,026      | Human-written dialogues |
| *Test*        | 4,216      | Human-written dialogues |
| *Human-Bot*   | 764        | Human-Bot interaction dialogues |
| *A2T*         | 2,079      | Auxiliary test set created by transforming examples in *Test* |
| *RCT*         | 2,011      | Auxiliary test set created by transforming examples in *Test* |

The details of each group can be found in the [Nie et al. (2020)](https://arxiv.org/abs/2012.13391).

### Format
The format of the file is `JSONL`. Each line in the file is one dialogue example saved in a `JSON`.
Primary fields that are required for the contradiction detection task:
- `record_id`: It is the unique ID for the example.
- `turns`: The field contains a list of turns that presents a conversation between two speaker.
- `is_contradiction`: The field indicates the label of the example. `true` means that the last turn (`turns[-1]`) contradicts some turns in the dialogue history. `false` means that the last turn is not a contradiction.
- `aggregated_contradiction_indices`: The field is a list of the indices that gives the location of the supporting evidence for the contradiction (w.r.t. to the `turns` list). Notices that the `turn_id` of the last turn (`turns[-1]`) will always be in this list.

Other fields that is related to the contradiction task:
- `num_of_turns_by_writer`: The field indicates how many turns the annotator created for providing a contradiction utterance.
- `writer_contradiction_indices`: The original contradiction indicates provided by the annotator.
- `verifications`: For each examples, we asked another three annotator for verification. This field gives the results of the verification.

Field not described are not required for the contradiction detection task.

A example `JSON` is shown below:
```
{
    # Primary field needed for contradiction detection task.
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
        "source": "BST_test"
    },
    "conversation_id": "9cb462d9-86f1-4296-af36-009d2e4d90f8#truncated#4",
}
```
