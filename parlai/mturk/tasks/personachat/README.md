This directory contains three separate tasks related to the collection of the
Persona-Chat dataset featured in [this](https://arxiv.org/pdf/1801.07243.pdf) paper.


1. **personachat_collect_personas** is the task of collecting character descriptions.
2. **personachat_chat** assumes we already have character descriptions. It randomly assigns
two Mechanical Turk workers characters, and asks them to chat to each other using that character.
3. **personachat_rephrase** is the task of collecting revised versions of a set of character descriptions.
It gives a worker a character and asks them to rephrase it one sentence at a time.
