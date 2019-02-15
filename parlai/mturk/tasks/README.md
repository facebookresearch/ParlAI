## Included

`parlai/mturk/tasks` contains various MTurk tasks for inspiration and help on creating new experiments or for reproducing existing experiments:

### Tasks

No actual tasks have been created on the new frontend yet

### Demo Tasks

- **multi_agent_dialog/**: Demo task where two agents on MTurk communicate in a world with another agent, in this case a local agent that sends messages from the command line.
- **qa_data_collection/**: Demo task that replicates the collection process of typical question-answer datasets (such as SQuAD) in the ParlAI-MTurk environment.
- **react_task_demo/**: Demo task that leverages the custom component development process for the react frontend to create 3 separate roles each with their own components. Comes in two varieties, `react_custom_no_extra_deps` uses the same dependencies that are specified in `parlai/mturk/core/react_server/package.json` and `react_custom_with_deps` specifies its own build process to be able to use custom dependencies. *These are a great starting point for working using the react frontend.*

### Legacy Tasks

These tasks were created using the old frontend and may or may not be ported to the new version in the frontend. They generally exist for reproducibility, but the newer frontend version should be preferred as the legacy version is no longer maintained or updated with any of the new functionality.

- **convai2_model_eval/**: a task used to evaluate models used in the convai2 dialogue competition.
- **dealnodeal/**: contains a negotiation task that asks agents to divide a set of objects with differing values to each agent between the two of them.
- **image_chat/**: a task that largely follows the steps of the `personality_captions` task, in having agents assume a personality trait and make comments and responses for an image.
- **model_evaluator/**: baseline code used for evaluating a model's perfomance based on workers' ratings.
- **personachat/**: chit-chat task where two agents are given personalities and asked to have a conversation.
- **personality_captions/**: an image captioning task where agents are asked to give an engaging caption based on a provided personality trait.
- **qualification_flow_example/**: Demo task that leverages the qualification setup along with storing local state in the `run.py` file in order to demonstrate how initial tasks can be set up as 'tests' that workers must pass to work on the real task.
- **talkthewalk/**: a navigation task where one agent has a map of an area and another agent has a street view, and the navigator and tourist have to work together to get the tourist to the correct location using only dialogue.
- **wizard_of_wikipedia/**: chit-chat task where two agents have a dialogue; one chats freely, perhaps based on a persona, while the other is the 'wizard', who bases his/her responses on documents (i.e. sentences) retrieved based on what the other agent says.
