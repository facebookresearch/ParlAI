# BlenderBot 3 Agent Information
This README outlines some of the internals of the BB3 agents.

## Modules, Explained

The top level BB3 agents initialize a series of sub-agents that accomplish the necessary modular tasks of BB3, whether that be search decision, memory generation, final dialogue response, etc. The [modules file](https://github.com/facebookresearch/ParlAI/blob/main/projects/bb3/agents/module.py) defines all of the modules that comprise BB3, and lots of information can be gleaned from there. Below, we outline all of the modules, as well as what format of the context each module expects when computing its relevant output. However, a few notes before we do so:

**Controlling module-specific parameters**: When utilizing the full BB3 setup (with the appropriate `--init-opt` presets), you can control module-specific generation parameters by simply prefixing any normal parameter with its module prefix; so, if you wanted to use nucleus sampling for the search response module, you could specify `--srm-inference nucleus`.

**Per-Module Interaction**: To interact with a module on its own (outside of a BB3 context), you can simply specify the following:

_BB3 3B without search_
```bash
parlai interactive --model projects.bb3.agents.r2c2_bb3_agent:BB3SubSearchAgent --model-file zoo:bb3/bb3_3B/model --force-skip-retrieval True --search-server none
```

_BB3 3B with search_
```bash
parlai interactive --model projects.bb3.agents.r2c2_bb3_agent:BB3SubSearchAgent --model-file zoo:bb3/bb3_3B/model --rag-retriever-type search_engine --search-server RELEVANT_SEARCH_SERVER
```

_BB3 30B/175B_
```bash
parlai interactive --model projects.bb3.agents.opt_api_agent:BB3OPTAgent --module <MODULE_PREFIX>
```

More details on the relevant context setup for each module is described below.

### `SDM`: Search Decision Module

Used for determining whether internet search is required. Only looks at the final turn of dialogue, generally. Default inference uses greedy decoding.

#### BB3 3B

##### Context
```
I wonder what the largest galaxy is __is-search-required__
```

##### Expected Output
- `__do-search__`: Internet search is required
- `__do-not-search__`: Internet search is **not** required

#### BB3 30B/175B

##### Context

```
Person 1: I wonder what the largest galaxy is
Search Decision:
```

##### Expected Output
- `search`: Internet search is required
- `do not search`: Internet search is **not** required.


###  `MDM`: Memory Decision Module

Used for determining whether the model requires accessing long-term memory. Only looks at the final turn of dialogue, along with the store of memories. Default inference uses greedy decoding.

#### BB3 3B

##### Context
```
your persona: I am an AI
partner's persona: I have a dog. My dog's name is bubbles.
I love my pet dog! __is-memory-required__
```

##### Expected Output
- `__do-access-memory__`: Long-term memory access is required
- `__do-not-access-memory__`: Long-term memory access is **not** required

#### BB3 30B/175B

##### Context
```
Personal Fact: Person 2's Persona: I am an AI
Personal Fact: Person 1's Persona: I have a dog. My dog's name is bubbles.
Person 1: I love my pet dog!
Memory Decision:
```

##### Expected Output
- `access memory`: Long-term memory access is required
- `do not access memory`: Long-term memory access is **not** required


###  `SGM`: Search Query Generation Module

Used for generating a search query given a dialogue context. Default inference uses greedy decoding.

#### BB3 3B

##### Context
```
I am a big fan of the New York Yankees
Me too! I wonder what their record is this year __generate-query__
```

##### Expected Output
A search query for an internet search engine

#### BB3 30B/175B

##### Context
```
Person 1: I am a big fan of the New York Yankees
Person 2: Me too! I wonder what their record is this year
Query:
```

##### Expected Output
A search query for an internet search engine.


###  `MGM`: Memory Generation Module

Used for generating a new memory to write to the long-term memory store. Conditioned on the last turn of the dialogue context. Default inference uses beam search in the 3B model and greedy decoding in the 30B/175B models.

#### BB3 3B

##### Context
```
I am a big fan of the New York Yankees __generate-memory__
```

##### Expected Output
A memory to write to the long-term memory store

#### BB3 30B/175B

##### Context
```
Person 1: I am a big fan of the New York Yankees.
Memory:
```

##### Expected Output
A memory to write to the long-term memory store


###  `CKM`: Contextual Knowledge Module

Extracts an entity from the context that the model can condition on in a dialogue response. Looks at the whole dialogue context. Default inference uses beam search in the 3B model and greedy decoding in the 30B/175B models.

#### BB3 3B

##### Context
```
I love baseball, whether its watching or playing it.
Me too! I am a big fan of the New York Yankees
I wonder what their record is?
The Yankees are 50-20 this year.
Wow, that's pretty good. __extract-entity__
```

##### Expected Output
An entity from the conversation on which to condition a dialogue response.

#### BB3 30B/175B

##### Context
```
Person 1: I love baseball, whether its watching or playing it.
Person 2: Me too! I am a big fan of the New York Yankees
Person 1: I wonder what their record is?
Person 2: The Yankees are 50-20 this year.
Person 1: Wow, that's pretty good.
Previous Topic:
```

##### Expected Output
An entity from the conversation on which to condition a dialogue response.


###  `MKM`: Memory Knowledge Module

Accesses the long-term memory store and retrieves/generates a memory on which to condition a dialogue response. Looks at the whole dialogue context, as well as the long-term memory store. Default inference uses beam search in the 3B model and greedy decoding in the 30B/175B models.

#### BB3 3B

##### Context
```
I love baseball, whether its watching or playing it.
Me too! I am a big fan of the New York Yankees
I wonder what their record is?
The Yankees are 50-20 this year.
Wow, that's pretty good. __access-memory__
```

Note: The memories here are/should be passed in as _documents_ to the model. If using the `BB3SubSearchAgent` directly, this would require calling `agent.set_memory(memories)` prior to response. `memories` should be a list of persona strings.

##### Expected Output
A memory on which to condition a dialogue response.

#### BB3 30B/175B

##### Context
```
Person 1's Persona: I love baseball. I am a baseball fan.
Person 2's Persona: I live in New York.
Person 1: I love baseball, whether its watching or playing it.
Person 2: Me too! I am a big fan of the New York Yankees
Person 1: I wonder what their record is?
Person 2: The Yankees are 50-20 this year.
Person 1: Wow, that's pretty good.
Personal Fact:
```

Note: The memories here are passed in _directly in the context_, with the appropriate prefixes.

##### Expected Output
A memory on which to condition a dialogue response.


###  `SKM`: Search Knowledge Module

Generates a knowledge sentence from a set of retrieved external documents. Looks at the whole dialogue context, as well as retrieved documents from the internet. Default inference uses beam search in the 3B model and greedy decoding in the 30B/175B models.

#### BB3 3B

##### Context
```
I love baseball, whether its watching or playing it.
Me too! I am a big fan of the New York Yankees
I wonder what their record is? __generate-knowledge__
```

Note: If using a search server and the `BB3SubSearchAgent`, you'll want to call `agent.model_api.set_search_queries(search_queries)` directly, where `search_queries` is a list of strings (one for each batch example); this will seed the agent's search engine with the appropriate query strings before asking the model to generate a knowledge sentence.

##### Expected Output
Knowledge sentence(s) on which to ground a dialogue response.

#### BB3 30B/175B

##### Context
```
External Knowledge: The New York Yankees have a record of 50-20 in 2022.
External Knowledge: The New York Yankees play at Yankee Stadium in the Bronx.
External Knowledge: The New York Yankees have won the World Series 27 times.
Person 1: I love baseball, whether its watching or playing it.
Person 2: Me too! I am a big fan of the New York Yankees
Person 1: I wonder what their record is?
Interesting Fact:
```

Note: The search documents here are passed in _directly in the context_, with the appropriate prefixes.

##### Expected Output
Knowledge sentence(s) on which to ground a dialogue response.

###  `CRM`: Contextual Response Module

Given an extracted entity from the context, generate a dialogue response. Looks at the whole dialogue context. Default inference uses beam search in the 3B model and [factual nucleus decoding](https://arxiv.org/abs/2206.04624) in the 30B/175B models.

#### BB3 3B

##### Context
```
I love baseball, whether its watching or playing it.
Me too! I am a big fan of the New York Yankees
I wonder what their record is?
The Yankees are 50-20 this year.
Wow, that's pretty good.
__entity__ playing __endentity__
```

##### Expected Output
A dialogue response conditioned on a specific phrase in the context.

#### BB3 30B/175B

##### Context
```
Person 1: I love baseball, whether its watching or playing it.
Person 2: Me too! I am a big fan of the New York Yankees
Person 1: I wonder what their record is?
Person 2: The Yankees are 50-20 this year.
Person 1: Wow, that's pretty good.
Previous Topic: playing
Person 2:
```

##### Expected Output
A dialogue response conditioned on a specific phrase in the context.

###  `MRM`: Memory Response Module

Given a memory from the long-term memory store, generate a dialogue response. Looks at the whole dialogue context. Default inference uses beam search in the 3B model and [factual nucleus decoding](https://arxiv.org/abs/2206.04624) in the 30B/175B models.

#### BB3 3B

##### Context
```
I love baseball, whether its watching or playing it.
Me too! I am a big fan of the New York Yankees
I wonder what their record is?
The Yankees are 50-20 this year.
Wow, that's pretty good.
__memory__ partner's persona: I love playing baseball __endmemory__
```

##### Expected Output
A dialogue response conditioned on a chosen memory from the long-term memory store.

#### BB3 30B/175B

##### Context
```
Person 1: I love baseball, whether its watching or playing it.
Person 2: Me too! I am a big fan of the New York Yankees
Person 1: I wonder what their record is?
Person 2: The Yankees are 50-20 this year.
Person 1: Wow, that's pretty good.
Personal Fact: Person 1's Persona: I love playing baseball
Person 2:
```


##### Expected Output
A dialogue response conditioned on a chosen memory from the long-term memory store.

###  `SRM`: Search Response Module

Given a knowledge sentence, generate a dialogue response. Looks at the whole dialogue context, as well as retrieved documents from the internet. Default inference uses beam search in the 3B model and [factual nucleus decoding](https://arxiv.org/abs/2206.04624) in the 30B/175B models.

#### BB3 3B

##### Context
```
I love baseball, whether its watching or playing it.
Me too! I am a big fan of the New York Yankees
I wonder what their record is?
__knowledge__ The New York Yankees have a 50-20 record __endknowledge__
```

##### Expected Output
A dialogue response conditioned on a knowledge sentence.

#### BB3 30B/175B

##### Context
```
Person 1: I love baseball, whether its watching or playing it.
Person 2: Me too! I am a big fan of the New York Yankees
Person 1: I wonder what their record is?
Interesting Fact: The New York Yankees have a record of 50-20
Person 2:
```

##### Expected Output
A dialogue response conditioned on a knowledge sentence.


###  `VRM`: Vanilla Response Module

Generate a dialogue response. This response is *only* conditioned on the dialogue context. Default inference uses beam search in the 3B model and [factual nucleus decoding](https://arxiv.org/abs/2206.04624) in the 30B/175B models.

#### BB3 3B

##### Context
```
I love baseball, whether its watching or playing it.
Me too! I am a big fan of the New York Yankees
I wonder what their record is?
The New York Yankees have a 50-20 record this year.
Wow, that's really good!
```

##### Expected Output
A dialogue response conditioned on only the dialogue history.

#### BB3 30B/175B

##### Context
```
Person 1: I love baseball, whether its watching or playing it.
Person 2: Me too! I am a big fan of the New York Yankees
Person 1: I wonder what their record is?
Interesting Fact: The New York Yankees have a record of 50-20
Person 2: The New York Yankees have a 50-20 record this year.
Person 1: Wow, that's really good!
Person 2:
```

##### Expected Output
A dialogue response conditioned on only the dialogue history.


### Combining knowledge in a dialogue response.

You can combine different knowledge sources in the dialogue response by simply adding them to the end of the context sent to the model:

#### BB3 3B

##### Context
```
I love baseball, whether its watching or playing it.
Me too! I am a big fan of the New York Yankees
I wonder what their record is?
__entity__ playing __endentity__
__memory__ partner's persona: I love playing baseball __endmemory__
__knowledge__ The New York Yankees have a 50-20 record __endknowledge__
```

#### BB3 30B/175B

##### Context
```
Person 1: I love baseball, whether its watching or playing it.
Person 2: Me too! I am a big fan of the New York Yankees
Person 1: I wonder what their record is?
Previous Topic: playing
Personal Fact: Person 1's Persona: I love playing baseball
Interesting Fact: The New York Yankees have a 50-20 record
Person 2:
```
