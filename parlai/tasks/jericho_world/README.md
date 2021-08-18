# Jericho World
ParlAI teachers for the [JerichoWorld Dataset](https://github.com/JerichoWorld/JerichoWorld).
WIP: More teachers will be added soon.

## Knowledge Graph Teachers
* **StaticKGTeacher**: Maps the location name and description, and list of surounding objects to the Knowledge Graph. This taecher only looks at its current single state, hence *static*.
* **ActionKGTeacher**: Maps the location name and description, list of surounding objects, and a received *action from player* to the *mutations* that are caused to the knowledge graph after the action. In other words, what changes (mutations) should happen to the knowledge graph after this action.

## Action Teachers

* **StateToActionTeacher**: Maps the location name and description, and list of surounding objects to the player action in that round.