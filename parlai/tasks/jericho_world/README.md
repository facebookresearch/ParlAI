# Jericho World
ParlAI teachers for the [JerichoWorld Dataset](https://github.com/JerichoWorld/JerichoWorld).

## Knowledge Graph Teachers
* **StaticKGTeacher**: Maps the location name and description, and list of surounding objects to the Knowledge Graph. This taecher only looks at its current single state, hence *static*.
* **ActionKGTeacher**: Maps the location name and description, list of surounding objects, and a received *action from player* to the *mutations* that are caused to the knowledge graph after the action. In other words, what changes (mutations) should happen to the knowledge graph after this action.

## Action Teachers
* **StateToValidActionsTeacher**:  Maps the game state (location name and description, knowledge graph, and list of surounding objects) to the set of valid actions. This is what was addressed in the main Jericho World paper.
* **StateToActionTeacher**: Maps the game state (location name and description, knowledge graph, and list of surounding objects) to the player action in that round.

> **_NOTE:_**  There were often incomplete graphs in the dataset (see [this issue](https://github.com/JerichoWorld/JerichoWorld/issues/3)).
Here, we discard knowledge graph, or their entities that seem corrupted. 
In addition, setting `--prune-knowledge-graph` to true forces the knowledge graph to keep only the entities that are mentioned in the description for `StaticKGTeacher` or objects that are in the inventory (eg, `[you , have , x]` edges in the knowledge graph). This is to avoid hallucination.