#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BB3 with R2C2 base.

SDM: Search Decision Model
MDM: Memory Decision Model
SGM: Search Query Generator Model
MGM: Memory Generator Model
MKM: Memory Knowledge Model
CKM: Contextual Knowledge Model
SKM: Search Knowledge Model
MRM: Memory Dialogue Response Model
CRM: Contextual Dialogue Response Model
SRM: Search Dialogue Response Model
"""
from collections import defaultdict
import copy
import torch
from types import MethodType
from typing import List, Tuple, Optional, Dict, Any, Union

from parlai.agents.rag.retrievers import (
    Document,
    RagRetriever,
    BLANK_DOC,
    RetrieverType,
)
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.torch_agent import History
import parlai.utils.logging as logging

from projects.seeker.agents.seeker import (
    SeekerAgent,
    ComboFidSearchQueryAgent,
    ComboFidGoldDocumentAgent,
)
from projects.seeker.agents.seeker_modules import (
    combo_fid_retriever_factory,
    ComboFidModel,
    ComboFidSearchQuerySearchEngineRetriever,
)
from projects.seeker.utils import (
    SearchDecision,
    krm_get_batch_context_only_knowledge,
    krm_get_batch_context,
    drm_get_batch_context,
)
from projects.seeker.agents.modular_agent import ModularAgentMixin


import projects.bb3.constants as CONST
from projects.bb3.agents.module import Module
import projects.bb3.tasks.mutators  # noqa: F401
from projects.bb3.agents.utils import clean_text, Decision, MemoryUtils


def bb3_retriever_factory(
    opt: Opt, dictionary: DictionaryAgent, shared=None
) -> Optional[RagRetriever]:
    """
    Build a BB3 retriever, which can handle both memories and search.
    """
    if opt.get('converting'):
        return None
    retriever = RetrieverType(opt['rag_retriever_type'])
    if retriever is RetrieverType.SEARCH_ENGINE:
        return BB3Retriever(opt, dictionary, shared=shared)  # type: ignore
    else:
        return combo_fid_retriever_factory(opt, dictionary, shared)


class BB3Model(ComboFidModel):
    """
    Override the ComboFid model to allow memory computation.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, retriever_shared=None):
        super().__init__(opt, dictionary, retriever_shared)
        self.retriever = bb3_retriever_factory(opt, dictionary, shared=retriever_shared)
        self.top_docs = []

    def set_memory(self, memories: List[List[str]]):
        """
        Set retriever's memories.

        :param memories:
            batchsize-length list of memories for each batch item
        """
        assert self.retriever is not None
        self.retriever.set_memory(memories)

    def get_memory(self) -> List[List[str]]:
        """
        Get retriever's memories.

        :return memories:
            return a batchsize-length list of memories for each batch item.
        """
        assert self.retriever is not None
        return self.retriever.get_memory()

    def set_retriever_type(self, r_type: str):
        """
        Informs the retriever whether to retrieve via search queries or long term
        memory.

        :param r_type:
            string indicating whether we're using search or memory.
        """
        assert self.retriever is not None
        self.retriever.retriever_type = r_type


class BB3Retriever(ComboFidSearchQuerySearchEngineRetriever):
    """
    Override the retriever to function as both a long-term memory accessor and internet-
    accessor.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared):
        super().__init__(opt, dictionary, shared=shared)
        self.n_docs = opt['n_docs']
        self.dict = dictionary
        self.memory: List[List[str]] = [[] for _ in range(opt['batchsize'])]
        self._retriever_type = 'search'

    def set_memory(self, memories: List[List[str]]):
        """
        Set retriever memories.
        """
        self.memory = memories

    def get_memory(self) -> List[List[str]]:
        """
        Return retriever memories.
        """
        return self.memory

    @property
    def retriever_type(self) -> str:
        return self._retriever_type

    @retriever_type.setter
    def retriever_type(self, r_type: str):
        self._retriever_type = r_type

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Return the search engine docs if in search mode; otherwise, return memories.
        """
        if self.retriever_type == 'search':
            return super().retrieve_and_score(query)

        top_docs = []
        top_doc_scores = []
        max_n_docs: int = self.n_docs
        for memories in self.memory:
            docs_i = []
            scores_i = []
            for memory in memories:
                docs_i.append(Document(docid='', text=memory, title=''))
                scores_i.append(1)
            # Change this debug later
            max_n_docs = max(max_n_docs, len(docs_i))
            top_docs.append(docs_i)
            top_doc_scores.append(scores_i)
        # Pad with empty docs
        for i in range(len(top_docs)):
            n_empty = max_n_docs - len(top_docs[i])
            if n_empty:
                top_docs[i] = top_docs[i] + [BLANK_DOC] * n_empty
                top_doc_scores[i] = top_doc_scores[i] + [0] * n_empty
        self.top_docs = top_docs
        return top_docs, torch.Tensor(top_doc_scores).to(query.device)


class BB3SubAgentMixin(Agent):
    """
    Override Combo Agent to allow storing memories in the retriever.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'BlenderBot3'

    def build_model(self) -> BB3Model:
        """
        Build and return BB3Model.
        """
        if self.generation_model == 't5':
            raise RuntimeError('T5 currently not supported')
        else:
            model = BB3Model(self.opt, self.dict)
        return model

    def set_memory(self, memories: List[List[str]]):
        """
        Set retriever's memories.
        """
        self.model_api.set_memory(memories)

    def get_memory(self) -> List[List[str]]:
        """
        Get retriever's memories.
        """
        return self.model_api.get_memory()

    def set_retriever_type(self, r_type: str):
        """
        Set retriever type.
        """
        self.model_api.set_retriever_type(r_type)


class BB3SubSearchAgent(BB3SubAgentMixin, ComboFidSearchQueryAgent):
    pass


class BB3SubGoldAgent(BB3SubAgentMixin, ComboFidGoldDocumentAgent):
    pass


class BlenderBot3Agent(ModularAgentMixin):
    """
    BB3 Agent.

    We DO NOT subclass SeeKeR, and re-implement everything.

    In places, it is noted where things could be abstracted to their own function.
    """

    MODEL_TYPE: str = 'R2C2'

    @classmethod
    def get_additional_agent_args(cls) -> ParlaiParser:
        return SeekerAgent.get_additional_agent_args()

    @classmethod
    def add_additional_subagent_args(cls, parser: ParlaiParser) -> ParlaiParser:
        """
        Setup the subagent args for all submodules.
        """
        additional_agent_parser = cls.get_additional_agent_args()
        for action in additional_agent_parser._actions:
            key = max(action.option_strings, key=lambda x: len(x))
            type = action.type

            for m in Module:
                parser.add_argument(
                    f'--{m.tag()}-{key.strip("-")}', type=type, required=False
                )
        return parser

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Command line args for BB3.
        """
        cls.add_additional_subagent_args(parser)
        group = parser.add_argument_group('SeeKeR Agent Args')
        for module in Module:
            tag = module.tag()
            group.add_argument(
                f'--{tag}-model',
                type=str,
                help=f'agent (not model file) to load for {tag}',
            )
            group.add_argument(
                f'--{tag}-memory-retriever',
                type='bool',
                default=False,
                help='specify for memory retriever',
            )
            if module.r2c2_prompt():
                group.add_argument(
                    f'--{module.message_name()}-control-token',
                    type=str,
                    default=module.r2c2_prompt(),
                    help=f'control token for {tag}',
                )
            if module.is_dialogue():
                group.add_argument(
                    f'--beam-disregard-knowledge-for-{tag}-context-blocking',
                    type='bool',
                    default=False,
                    help=f'If True, disregard the knowledge input for {module.agent_name()} context blocking.',
                )
            elif module.is_knowledge():
                group.add_argument(
                    f'--include-knowledge-in-{tag}-context-blocking',
                    type=bool,
                    default=True,
                    help=f'If True, put the {module.agent_name()} responses in the context for the {module.agent_name()}.',
                )
                group.add_argument(
                    f'--exclude-context-in-{tag}-context-blocking',
                    type=bool,
                    default=False,
                    help=f'Used in conjunction with --include-knowledge-in-{tag}-context-blocking. '
                    f'If specified, only block on the knowledge, and not the concatenation.',
                )
        group.add_argument(
            '--memory-decision-do-access-reply',
            type=str,
            default=CONST.DO_ACCESS_MEMORY,
            help='control token returned by MDM to indicate using memory',
        )
        group.add_argument(
            '--memory-decision-dont-access-reply',
            type=str,
            default=CONST.DONT_ACCESS_MEMORY,
            help='control token returned by MDM to indicate not using memory',
        )
        group.add_argument(
            '--knowledge-conditioning',
            type=str,
            choices=['combined', 'separate', 'both'],
            default='combined',
            help='Specify the way in which the model uses knowledge.\n'
            'combined: condition on all knowledge simultaneously'
            'separate: condition on all knowledge separately, re-ranke later'
            'both: do both combined and separate and re-rank final beam',
        )
        # Copied from Seeker
        group.add_argument(
            '--search-decision-do-search-reply',
            type=str,
            default=CONST.DO_SEARCH,
            help='control token returned by SDM to indicate searching',
        )
        group.add_argument(
            '--search-decision-dont-search-reply',
            type=str,
            default=CONST.DO_NOT_SEARCH,
            help='control token returned by SDM to indicate not searching',
        )
        group.add_argument(
            '--all-model-path',
            type=str,
            default=None,
            help='If specified, load all models with this path',
        )
        group.add_argument(
            '--search-decision',
            type=str,
            default=Decision.COMPUTE.value,
            choices=[s.value for s in SearchDecision],
        )
        group.add_argument(
            '--memory-decision',
            type=str,
            default=Decision.COMPUTE.value,
            choices=[s.value for s in SearchDecision],
        )
        group.add_argument(
            '--memory-decision-use-memories',
            type='bool',
            default=True,
            help='If true, the memory decision module will have access to memories when making the decision',
        )
        group.add_argument(
            '--contextual-knowledge-decision',
            type=str,
            default=Decision.COMPUTE.value,
            choices=[s.value for s in SearchDecision],
        )
        group.add_argument(
            '--inject-query-string',
            type=str,
            default=None,
            help='If set, this string is appended to all search queries.',
        )
        group.add_argument(
            '--search-server', type=str, default=None, help='search server to use.'
        )
        group.add_argument(
            '--serializable-output',
            type='bool',
            default=False,
            help='Whether to make output serializable. Specify True when using e.g. mechanical turk.',
        )

        return parser

    def __init__(self, opt, shared=None):
        self.id = 'BlenderBot3'
        opt = copy.deepcopy(opt)
        self.opt = opt
        one_model = opt.get('all_model_path') or opt.get('model_file')
        assert one_model, "Must specify a model file for this agent."
        for k in Module:
            opt[k.model_file_path_key()] = one_model
        self._construct_subagent_opts(opt)

        self.agents: Dict[Union[str, Module], Agent] = {}
        self.clones: Dict[Union[str, Module], List[Agent]] = {}
        if not shared:
            agent_opts = self.opts[Module.SEARCH_KNOWLEDGE]
            agent = self._init_top_agent(agent_opts)
            # two mappings; one for code access, one for debug access.
            self.agents[Module.SEARCH_KNOWLEDGE.agent_name()] = agent
            self.agents[Module.SEARCH_KNOWLEDGE] = agent
            logging.verbose(f"options for {Module.SEARCH_KNOWLEDGE.agent_name()}")
            if logging.logger.isEnabledFor(logging.VERBOSE):
                Opt(self.agents[Module.SEARCH_KNOWLEDGE].opt).log()
        else:
            agent = create_agent_from_shared(shared['search_knowledge_agent_share'])
            self.agents[Module.SEARCH_KNOWLEDGE.agent_name()] = agent
            self.agents[Module.SEARCH_KNOWLEDGE] = agent

        self.memories = []
        self.search_knowledge_responses = ['__SILENCE__']
        self.memory_knowledge_responses = ['__SILENCE__']
        self.contextual_knowledge_responses = ['__SILENCE__']
        self.search_decision = Decision(opt['search_decision'])
        self.memory_decision = Decision(opt['memory_decision'])
        self.contextual_knowledge_decision = Decision(
            opt['contextual_knowledge_decision']
        )
        self.inject_query_string = opt.get('inject_query_string', '')
        self.knowledge_conditioning = opt['knowledge_conditioning']

        for m in Module:
            agent = self._init_shared_model(m)
            self.agents[m.agent_name()], self.agents[m] = agent, agent
            if m.is_dialogue():
                self.clones[m.agent_name()], self.clones[m] = [agent], [agent]

        if not shared:
            # the main agent needs batchsize-many clones for handling history during batch act.
            for m in Module.dialogue_modules():
                self.clones[m.agent_name()] += [
                    self.agents[m.agent_name()].clone()
                    for _ in range(opt.get('batchsize', 1) - 1)
                ]
                self.clones[m] = self.clones[m.agent_name()]

        self._apply_context_blocking_patches()

        for m in Module:
            if m is Module.MEMORY_DECISION and opt['memory_decision_use_memories']:
                continue
            if m.is_one_turn_history():
                try:
                    assert (
                        self.agents[m].history.size == 1
                    ), f"wrong history size! set --{m.tag()}-history-size 1"
                except AttributeError:
                    pass

        super().__init__(opt, shared)

    def _init_top_agent(self, opt: Opt) -> Agent:
        """
        Initialize the toplevel agent.
        """
        return create_agent(opt, requireModelExists=True)

    def _apply_context_blocking_patches(self):
        """
        Optionally monkey-patch custom context blocking functions.
        """
        for m in Module:
            agent = self.agents[m]
            if m.is_knowledge():
                if self.opt[f'include_knowledge_in_{m.tag()}_context_blocking']:
                    orig_fun = agent._get_batch_context
                    if self.opt[f'exclude_context_in_{m.tag()}_context_blocking']:
                        agent._get_batch_context = MethodType(
                            lambda self, batch: krm_get_batch_context_only_knowledge(
                                self, batch, orig_fun
                            ),
                            agent,
                        )
                    else:
                        agent._get_batch_context = MethodType(
                            lambda self, batch: krm_get_batch_context(
                                self, batch, orig_fun
                            ),
                            agent,
                        )
            elif m.is_dialogue():
                if self.opt[f'beam_disregard_knowledge_for_{m.tag()}_context_blocking']:
                    orig_fun = agent._get_batch_context
                    agent._get_batch_context = MethodType(
                        lambda self, batch: drm_get_batch_context(
                            self, batch, orig_fun=orig_fun
                        ),
                        agent,
                    )

    @property
    def history(self) -> History:
        return self.agents[Module.SEARCH_DIALOGUE].history

    @property
    def knowledge_agent_history(self) -> History:
        return self.agents[Module.SEARCH_KNOWLEDGE].history

    def get_history_str(self) -> str:
        return self.history.get_history_str()

    def reset(self, clones_only: bool = False):
        """
        Override to reset all sub agents.

        :param clones_only:
            if true, only reset the dialogue clones
        """
        for agents in self.clones.values():
            for a in agents:
                a.reset()
        if not clones_only:
            for agent in self.agents.values():
                agent.reset()
            self.search_knowledge_responses = ['__SILENCE__']
            self.contextual_knowledge_responses = ['__SILENCE__']
            self.memory_knowledge_responses = ['__SILENCE__']
            self.memories = []

    def _construct_subagent_opts(self, opt: Opt):
        """
        Construct opts for each sub agent.

        :param opt:
            original Opt.
        """
        if opt['search_server']:
            opt['skm_search_server'] = opt['search_server']
            if 'override' in opt:
                opt['override'][f'{Module.SEARCH_KNOWLEDGE.tag()}_search_server'] = opt[
                    'search_server'
                ]
        self.opts = {}
        self.opts['init'] = opt
        override_opts = defaultdict(dict)
        for k, v in opt['override'].items():
            k_set = False
            for m in Module:
                if k.startswith(f'{m.tag()}_'):
                    override_opts[m][k] = v
                    k_set = True
                    break
            if not k_set:
                override_opts['general'][k] = v
        self.opts['override'] = override_opts
        for m in Module:
            if f'{m.tag()}_interactive_mode' not in override_opts[m]:
                override_opts[m][f'{m.tag()}_interactive_mode'] = opt.get(
                    'interactive_mode', False
                )

        for m in Module:
            filename = opt[m.model_file_path_key()]
            if not filename:
                continue
            self.opts[m] = self._get_subagent_opt(
                filename=filename,
                specific_override_args=override_opts[m],
                general_override_args=override_opts['general'],
            )
            self.opts[m]['model_file'] = filename
            self.opts[m]['override']['model_file'] = filename

    def _init_shared_model(self, opt_key: Module) -> Agent:
        """
        Initialize shared version of a model, for sub modules.
        """
        return super().init_shared_model(
            self.opts[opt_key], self.agents[Module.SEARCH_KNOWLEDGE]
        )

    def _get_subagent_opt(
        self,
        filename: str,
        specific_override_args: Dict[str, Any],
        general_override_args: Dict[str, Any],
    ) -> Opt:
        """
        Return the specific subagent opt parameters.
        """
        return super().get_subagent_opt(
            self.opt['datapath'],
            filename,
            specific_override_args,
            general_override_args,
        )

    def share(self):
        """
        Share the top agent.
        """
        shared = super().share()
        shared['search_knowledge_agent_share'] = self.agents[
            Module.SEARCH_KNOWLEDGE
        ].share()
        return shared

    def get_mdm_observation(self, ag_obs: Message) -> Message:
        """
        Add memories to the memory decision observation.

        :param ag_obs:
            incoming mdm observation

        :return mdm_obs:
            return mdm observation with memories in the context.
        """
        # Reset the history for one-turn agents
        # for now, protect this heavily, as this is being added for mdm only
        self.agents[Module.MEMORY_DECISION].reset()
        if self.memories:
            self_memories = [
                m.replace('your persona: ', '')
                for m in self.memories
                if m.startswith('your')
            ]
            partner_memories = [
                m.replace("partner's persona: ", '')
                for m in self.memories
                if m.startswith('partner')
            ]
            memories = f"your persona: {' '.join(self_memories)}\npartner's persona: {' '.join(partner_memories)}"
            ag_obs.force_set('text', '\n'.join([memories, ag_obs['text']]))
        return ag_obs

    def observe(self, observation: Message) -> Dict[Module, Message]:
        """
        Observe in all modules besides dialogue ones, which require outputs from the
        knowledge stage.

        :param observation:
            incoming message

        :return self.observation:
            returned observation is actually a dictionary mapping
            module name to the corresponding observation
        """
        observations = {}
        if not isinstance(observation, Message):
            observation = Message(observation)
        for key in ['label_candidates', 'knowledge']:
            # Delete unnecessarily large keys
            observation.pop(key, '')

        raw_observation = copy.deepcopy(observation)
        raw_observation['memories'] = self.memories
        observations['raw'] = raw_observation

        if observation.get('episode_done'):
            self.search_knowledge_responses = ['__SILENCE__']
            self.contextual_knowledge_responses = ['__SILENCE__']
            self.memory_knowledge_responses = ['__SILENCE__']

        for m in Module:
            # observe for all non-dialogue agents
            if m.is_dialogue():
                continue
            ag_obs = copy.deepcopy(observation)
            if self.opt[f"{m.message_name()}_control_token"]:
                ag_obs.force_set(
                    'temp_history', f" {self.opt[f'{m.message_name()}_control_token']}"
                )
            if m.skip_search():
                ag_obs.force_set('skip_retrieval', True)
            if m is Module.CONTEXTUAL_KNOWLEDGE:
                ag_obs['prior_knowledge_responses'] = ' '.join(
                    self.contextual_knowledge_responses
                )
            elif m is Module.SEARCH_KNOWLEDGE:
                ag_obs['prior_knowledge_responses'] = ' '.join(
                    self.search_knowledge_responses
                )
            elif m is Module.MEMORY_KNOWLEDGE:
                ag_obs['prior_knowledge_responses'] = ' '.join(
                    self.memory_knowledge_responses
                )
            if m is Module.MEMORY_DECISION and self.opt['memory_decision_use_memories']:
                ag_obs = self.get_mdm_observation(ag_obs)

            observations[m] = self.agents[m].observe(ag_obs)

        self.observations = observations
        return observations

    def batch_act_decision(
        self,
        observations: List[Dict[Union[str, Module], Message]],
        module: Module,
        agent: Agent,
    ) -> Tuple[List[Message], List[int]]:
        """
        Decision agent batch act.

        :param observations:
            observations for batch act.
        :param module:
            module making the decision

        :return (batch_reply, indices):
            batch_reply: reply from the decision agent
            indices: batch indices with which to use search/memory.
        """
        indices = []
        batch_reply = [{} for _ in range(len(observations))]
        if module is Module.SEARCH_DECISION:
            do = self.opt['search_decision_do_search_reply']
            dont = self.opt['search_decision_dont_search_reply']
            decision = self.search_decision
        else:
            assert module is Module.MEMORY_DECISION
            do = self.opt['memory_decision_do_access_reply']
            dont = self.opt['memory_decision_dont_access_reply']
            decision = self.memory_decision

        if decision is Decision.ALWAYS:
            indices = list(range(len(observations)))
        elif decision is Decision.NEVER:
            indices = []
        else:
            assert decision is Decision.COMPUTE
            assert agent
            batch_reply = agent.batch_act([o[module] for o in observations])
            for i, reply in enumerate(batch_reply):
                logging.debug(f"Example {i}, {module.agent_name()}: {reply['text']}")
                if reply['text'] == do or reply['text'].startswith(do):
                    indices.append(i)
                elif reply['text'] == dont or reply['text'].startswith(do):
                    continue
                else:
                    logging.error(
                        f"Decision Reply: {reply['text']}; defaulting to no search/memory"
                    )

        return batch_reply, indices

    def batch_act_sgm(
        self,
        observations: List[Dict[Union[str, Module], Message]],
        search_indices: List[int],
        agent: Agent,
    ) -> List[Message]:
        """
        Search Query Generator batch act.

        :param observations:
            list of observations
        :param search_indices:
            list of batch indices for which search is required.
        :param agent:
            search query generator agent

        :return batch_reply:
            return the batch reply from the search query agent
        """
        return super().batch_act_search_query_generation(
            observations,
            [o[Module.SEARCH_QUERY] for o in observations],
            search_indices,
            agent,
            self.agents[Module.SEARCH_KNOWLEDGE],
            self.inject_query_string,
        )

    def batch_act_mgm(
        self,
        observations: Optional[List[Dict[Union[str, Module], Message]]] = None,
        self_messages: Optional[List[Message]] = None,
        agent: Optional[Agent] = None,
    ) -> List[Message]:
        """
        Memory Generator batch act.

        :param observations:
            list of observations, where the text is the partner's message
        :param self_messages:
            list of messages that the bot generated.
        :param agent:
            memory generator agent

        :return batch_reply:
            return the batch reply from the search query agent
        """
        assert agent is not None
        if observations is not None:
            batch_reply_mgm = agent.batch_act(
                [o[Module.MEMORY_GENERATOR] for o in observations]
            )
            logging.debug(f"Partner Memories: {[a['text'] for a in batch_reply_mgm]}")
        else:
            assert self_messages is not None
            control_token = self.opt[
                f"{Module.MEMORY_GENERATOR.message_name()}_control_token"
            ]
            batch_reply_mgm = agent.batch_respond(
                [
                    Message(
                        {
                            'text': f"{m['text']} {control_token}"
                            if control_token
                            else m['text'],
                            'episode_done': True,
                        }
                    )
                    for m in self_messages
                ]
            )
            batch_reply_mgm = [Message({'text': t}) for t in batch_reply_mgm]
            logging.debug(f"Self Memories: {[a['text'] for a in batch_reply_mgm]}")
        agent.reset()
        return batch_reply_mgm

    def batch_act_knowledge(
        self,
        observations: List[Dict[Union[str, Module], Message]],
        search_indices: List[int],
        memory_indices: List[int],
        contextual_indices: List[int],
        batch_agents: Dict[Module, Agent],
        top_docs: Optional[List[List[Document]]] = None,
        top_memories: Optional[List[List[str]]] = None,
    ) -> List[Message]:
        """
        Batch act with Knowledge Models.

        :param observations:
            list of observations
        :param search_indices:
            list of indices for which we search.
        :param memory_indices:
            list of indices for which we access memory.
        :param contextual_indices:
            list of indices for which we access knowledge from context.
        :param top_docs:
            list of top documents; send if not normally returned in the batch act
        :param top_memories:
            list of top memories; send if not normally returned in the batch act

        :return batch_reply:
            batch_reply: batch reply from knowledge modules
        """
        batch_reply_knowledge = [Message({}) for _ in range(len(observations))]
        ckm_obs, skm_obs, mkm_obs = [], [], []
        for i, o in enumerate(observations):
            if i in contextual_indices:
                ckm_obs.append(o[Module.CONTEXTUAL_KNOWLEDGE])
            if i in search_indices:
                skm_obs.append(o[Module.SEARCH_KNOWLEDGE])
            if i in memory_indices:
                mkm_obs.append(o[Module.MEMORY_KNOWLEDGE])

        batch_reply_ckm = batch_agents[Module.CONTEXTUAL_KNOWLEDGE].batch_act(ckm_obs)
        batch_agents[Module.SEARCH_KNOWLEDGE].set_retriever_type('search')
        batch_reply_skm = batch_agents[Module.SEARCH_KNOWLEDGE].batch_act(skm_obs)
        batch_agents[Module.MEMORY_KNOWLEDGE].set_retriever_type('memory')
        batch_reply_mkm = batch_agents[Module.MEMORY_KNOWLEDGE].batch_act(mkm_obs)

        search_offset = 0
        memory_offset = 0
        contextual_offset = 0
        for i, message in enumerate(batch_reply_knowledge):
            if i in contextual_indices:
                message[Module.CONTEXTUAL_KNOWLEDGE.message_name()] = batch_reply_ckm[
                    contextual_offset
                ]['text']
                logging.debug(
                    f"Contextual KNOWLEDGE for example {i}: {message[Module.CONTEXTUAL_KNOWLEDGE.message_name()]}"
                )
                contextual_offset += 1
            if i in search_indices:
                message[Module.SEARCH_KNOWLEDGE.message_name()] = batch_reply_skm[
                    search_offset
                ]['text']
                logging.debug(
                    f"Search KNOWLEDGE for example {i}: {message[Module.SEARCH_KNOWLEDGE.message_name()]}"
                )
                docs = (
                    top_docs[search_offset]
                    if top_docs is not None
                    else batch_reply_skm[search_offset]['top_docs']
                )
                message[f'{Module.SEARCH_KNOWLEDGE.message_name()}_top_docs'] = docs
                search_offset += 1
            if i in memory_indices:
                message[Module.MEMORY_KNOWLEDGE.message_name()] = batch_reply_mkm[
                    memory_offset
                ]['text']
                logging.debug(
                    f"Memory KNOWLEDGE for example {i}: {message[Module.MEMORY_KNOWLEDGE.message_name()]}"
                )
                docs = (
                    top_memories[memory_offset]
                    if top_memories is not None
                    else batch_reply_mkm[memory_offset]['top_docs']
                )
                message[f'{Module.MEMORY_KNOWLEDGE.message_name()}_top_docs'] = docs
                memory_offset += 1

        return batch_reply_knowledge

    def batch_act_dialogue_combined(
        self,
        observations: List[Dict[Union[str, Module], Message]],
        batch_reply_knowledge: List[Message],
    ) -> List[Message]:
        """
        Dialogue batch act.

        Combine all available knowledge sources for generation.

        :param observations:
            observations from self.observe
        :param batch_reply_knowledge:
            batch reply from the knowledge module

        :return batch_reply_dialogue:
            return the reply from the dialogue models.
        """
        full_text = [
            clean_text(
                o[Module.CONTEXTUAL_KNOWLEDGE].get('full_text', o.get('text', ''))
            )
            for o in observations
        ]
        clones = self.clones[Module.SEARCH_DIALOGUE]
        agent = self.agents[Module.SEARCH_DIALOGUE]
        dialogue_agent_observations = []
        for i, (obs, knowledge_obs) in enumerate(
            zip(observations, batch_reply_knowledge)
        ):
            temp_history = '\n'
            srm_obs = copy.deepcopy(obs['raw'])
            for m in Module.knowledge_modules():
                if knowledge_obs.get(m.message_name()):
                    tokens = m.special_tokens()
                    temp_history += (
                        f"{tokens[0]} {knowledge_obs[m.message_name()]} {tokens[1]}"
                    )
            assert temp_history
            srm_obs.force_set('temp_history', temp_history)
            srm_obs.force_set('skip_retrieval', True)
            if not clones[i].history.get_history_str():
                srm_obs.force_set('text', full_text[i])
            dialogue_agent_observations.append(clones[i].observe(srm_obs))
        batch_reply_srm = agent.batch_act(dialogue_agent_observations)
        for i, obs in enumerate(batch_reply_srm):
            logging.debug(
                f"Combined DIALOGUE response for example {i}: {obs['text']}; score: {obs['beam_texts'][0][-1]:.2f}"
            )
            clones[i].self_observe(obs)
            # manually clear
            clones[i].history.temp_history = None

        return batch_reply_srm

    def batch_act_dialogue_separate(
        self,
        observations: List[Dict[Union[Module, str], Message]],
        batch_reply_knowledge: List[Message],
        search_indices: List[int],
        memory_indices: List[int],
        contextual_indices: List[int],
    ) -> List[Message]:
        """
        Dialogue batch act.

        Separately generate dialogue responses for each module with knowledge.

        Then, re-rank according to beam likelihood.

        :param observations:
            observations from self.observe
        :param batch_reply_knowledge:
            batch reply from the knowledge module
        :param search_indices:
            indices that used search
        :param memory_indices:
            indices that used memory
        :param contextual_indices:
            indices that used contextual knowledge

        :return batch_reply_dialogue:
            return the reply from the dialogue module.
        """
        full_text = [
            clean_text(
                o[Module.CONTEXTUAL_KNOWLEDGE].get('full_text', o.get('text', ''))
            )
            for o in observations
        ]
        srm = self.agents[Module.SEARCH_DIALOGUE]
        mrm = self.agents[Module.MEMORY_DIALOGUE]
        crm = self.agents[Module.CONTEXTUAL_DIALOGUE]
        srm_clones = self.clones[Module.SEARCH_DIALOGUE]
        crm_clones = self.clones[Module.MEMORY_DIALOGUE]
        mrm_clones = self.clones[Module.CONTEXTUAL_DIALOGUE]

        contextual_observations, search_observations, memory_observations = [], [], []
        batch_reply_crm, batch_reply_srm, batch_reply_mrm = [], [], []

        # First, we generate the observations given the knowledge outputs.
        for i, (obs, knowledge_obs) in enumerate(
            zip(observations, batch_reply_knowledge)
        ):
            dialogue_obs = copy.deepcopy(obs['raw'])
            dialogue_obs.force_set('skip_retrieval', True)
            if i in search_indices:
                factual_obs = copy.deepcopy(dialogue_obs)
                tokens = Module.SEARCH_KNOWLEDGE.special_tokens()
                factual_obs.force_set(
                    'temp_history',
                    f"\n{tokens[0]} {knowledge_obs[Module.SEARCH_KNOWLEDGE.message_name()]} {tokens[1]}",
                )
                if not srm_clones[i].history.get_history_str():
                    factual_obs.force_set('text', full_text[i])
                search_observations.append(srm_clones[i].observe(factual_obs))
            if i in memory_indices:
                memory_obs = copy.deepcopy(dialogue_obs)
                tokens = Module.MEMORY_KNOWLEDGE.special_tokens()
                memory_obs.force_set(
                    'temp_history',
                    f"\n{tokens[0]} {knowledge_obs[Module.MEMORY_KNOWLEDGE.message_name()]} {tokens[1]}",
                )
                if not mrm_clones[i].history.get_history_str():
                    memory_obs.force_set('text', full_text[i])
                memory_observations.append(mrm_clones[i].observe(memory_obs))
            if i in contextual_indices:
                contextual_obs = copy.deepcopy(dialogue_obs)
                tokens = Module.CONTEXTUAL_KNOWLEDGE.special_tokens()
                contextual_obs.force_set(
                    'temp_history',
                    f"\n{tokens[0]} {knowledge_obs[Module.CONTEXTUAL_KNOWLEDGE.message_name()]} {tokens[1]}",
                )
                if not crm_clones[i].history.get_history_str():
                    contextual_obs.force_set('text', full_text[i])
                contextual_observations.append(crm_clones[i].observe(contextual_obs))

        if search_observations:
            batch_reply_srm = srm.batch_act(search_observations)
        if memory_observations:
            batch_reply_mrm = mrm.batch_act(memory_observations)
        if contextual_observations:
            batch_reply_crm = crm.batch_act(contextual_observations)

        batch_reply_dialogue = [Message({}) for _ in range(len(observations))]
        search_offset = 0
        memory_offset = 0
        contextual_offset = 0
        # Second, we generate the dialogue responses.
        for i in range(len(observations)):
            if i in search_indices:
                search_act = batch_reply_srm[search_offset]
                srm_clones[search_offset].self_observe(search_act)
                batch_reply_dialogue[i][
                    Module.SEARCH_DIALOGUE.message_name()
                ] = search_act['text']
                batch_reply_dialogue[i][
                    f"{Module.SEARCH_DIALOGUE.message_name()}_score"
                ] = search_act['beam_texts'][0][-1]
                search_offset += 1
                logging.debug(
                    f"Search DIALOGUE response for {i}: {search_act['text']}; score: {search_act['beam_texts'][0][-1]:.2f}"
                )
            if i in memory_indices:
                memory_act = batch_reply_mrm[memory_offset]
                mrm_clones[memory_offset].self_observe(memory_act)
                batch_reply_dialogue[i][
                    Module.MEMORY_DIALOGUE.message_name()
                ] = memory_act['text']
                batch_reply_dialogue[i][
                    f"{Module.MEMORY_DIALOGUE.message_name()}_score"
                ] = memory_act['beam_texts'][0][-1]
                memory_offset += 1
                logging.debug(
                    f"Memory DIALOGUE response for {i}: {memory_act['text']}; score: {memory_act['beam_texts'][0][-1]:.2f}"
                )
            if i in contextual_indices:
                contextual_act = batch_reply_crm[contextual_offset]
                crm_clones[contextual_offset].self_observe(contextual_act)
                batch_reply_dialogue[i][
                    Module.CONTEXTUAL_DIALOGUE.message_name()
                ] = contextual_act['text']
                batch_reply_dialogue[i][
                    f"{Module.CONTEXTUAL_DIALOGUE.message_name()}_score"
                ] = contextual_act['beam_texts'][0][-1]
                contextual_offset += 1
                logging.debug(
                    f"Contextual DIALOGUE response for {i}: {contextual_act['text']}; score: {contextual_act['beam_texts'][0][-1]:.2f}"
                )
            # manually clear
            srm_clones[i].history.temp_history = None
            mrm_clones[i].history.temp_history = None
            crm_clones[i].history.temp_history = None

        # Third, we re-rank according to beam likelihood.
        for reply in batch_reply_dialogue:
            options, scores = [], []
            for i, m in enumerate(
                [
                    Module.SEARCH_DIALOGUE,
                    Module.MEMORY_DIALOGUE,
                    Module.CONTEXTUAL_DIALOGUE,
                ]
            ):
                options.append(reply.get(m.message_name(), ''))
                scores.append(
                    (i, reply.get(f"{m.message_name()}_score", -float('inf')))
                )
            max_score = max(scores, key=lambda x: x[-1])
            reply['text'] = options[max_score[0]]
            reply['max_score'] = max_score[1]

        return batch_reply_dialogue

    def collate_batch_acts(
        self,
        batch_reply_sdm: List[Message],
        batch_reply_mdm: List[Message],
        batch_reply_sgm: List[Message],
        batch_reply_mgm_self: List[Message],
        batch_reply_mgm_partner: List[Message],
        batch_reply_knowledge: List[Message],
        batch_reply_dialogue: List[Message],
        available_memory: List[List[str]],
    ) -> List[Message]:
        """
        Collate all of the batch acts from the various modules.

        :param batch_reply_X:
            batch reply from module X
        :param available_memory:
            list of memories for each batch item

        :return batch_reply:
            return the agent's final batch reply
        """
        final_batch_reply = []
        for sdm, mdm, sgm, mgm_self, mgm_partner, km, srm, mems in zip(
            batch_reply_sdm,
            batch_reply_mdm,
            batch_reply_sgm,
            batch_reply_mgm_self,
            batch_reply_mgm_partner,
            batch_reply_knowledge,
            batch_reply_dialogue,
            available_memory,
        ):
            if srm.is_padding():
                continue
            reply = Message(
                {
                    k: v
                    for k, v in srm.items()
                    if k not in ['top_docs']  # leave as list for future use cases
                }
            )
            reply.force_set(Module.SEARCH_DECISION.message_name(), sdm.get('text', ''))
            reply.force_set(Module.MEMORY_DECISION.message_name(), mdm.get('text', ''))
            reply.force_set(Module.SEARCH_QUERY.message_name(), sgm.get('text', ''))
            reply.force_set(
                f'{Module.MEMORY_GENERATOR.message_name()}_self',
                mgm_self.get('text', ''),
            )
            reply.force_set(
                f'{Module.MEMORY_GENERATOR.message_name()}_partner',
                mgm_partner.get('text', ''),
            )
            reply.force_set('memories', mems)
            if MemoryUtils.is_valid_memory(
                reply['memories'],
                mgm_self.get('text', ''),
                MemoryUtils.get_memory_prefix('self', self.MODEL_TYPE),
            ):
                reply.force_set(
                    'memories',
                    reply['memories']
                    + [
                        MemoryUtils.add_memory_prefix(
                            mgm_self['text'], 'self', self.MODEL_TYPE
                        )
                    ],
                )
            if MemoryUtils.is_valid_memory(
                reply['memories'],
                mgm_partner.get('text', ''),
                MemoryUtils.get_memory_prefix('partner', self.MODEL_TYPE),
            ):
                reply.force_set(
                    'memories',
                    reply['memories']
                    + [
                        MemoryUtils.add_memory_prefix(
                            mgm_partner['text'], 'partner', self.MODEL_TYPE
                        )
                    ],
                )
            reply.force_set(
                Module.SEARCH_KNOWLEDGE.message_name(),
                km.get(Module.SEARCH_KNOWLEDGE.message_name(), ''),
            )
            reply.force_set(
                Module.CONTEXTUAL_KNOWLEDGE.message_name(),
                km.get(Module.CONTEXTUAL_KNOWLEDGE.message_name(), ''),
            )
            reply.force_set(
                Module.MEMORY_KNOWLEDGE.message_name(),
                km.get(Module.MEMORY_KNOWLEDGE.message_name(), ''),
            )
            for m in Module:
                # set all the knowledge responses
                if m.is_knowledge():
                    reply.force_set(m.message_name(), km.get(m.message_name(), ''))
                # if separate, set all of the dialogue responses as well
                elif m.is_dialogue():
                    reply.force_set(m.message_name(), srm.get(m.message_name(), ''))
                    reply.force_set(
                        f"{m.message_name()}_score",
                        reply.get(f"{m.message_name()}_score", -float('inf')),
                    )

                if not m.skip_search():
                    docs = km.get(
                        f'{m.message_name()}_top_docs', [Document("", "", "")]
                    )
                    reply.force_set(
                        f'{m.message_name()}_doc_titles', [d.get_title() for d in docs]
                    )
                    reply.force_set(
                        f'{m.message_name()}_doc_content', [d.get_text() for d in docs]
                    )
                    reply.force_set(
                        f'{m.message_name()}_doc_urls', [d.get_id() for d in docs]
                    )
            logging.debug(reply)
            final_batch_reply.append(reply)

        if self.opt['serializable_output']:
            for i in range(len(final_batch_reply)):
                final_batch_reply[i] = Message(
                    {
                        k: v
                        for k, v in final_batch_reply[i].items()
                        if k in ['text', 'id', 'episode_done']
                    }
                )
        return final_batch_reply

    def batch_act(
        self, observations: List[Dict[Union[str, Module], Message]]
    ) -> List[Message]:
        """
        Full batch_act pipeline.

        :param observations:
            batchsize-length list of observations from self.observe

        :return reply:
            return batchsize-length list of final replies.
        """
        # First, determine whether we're searching or accessing memory
        try:
            self.agents[Module.MEMORY_KNOWLEDGE].set_memory(
                [o['raw']['memories'] for o in observations]
            )
            available_memory = self.agents[Module.MEMORY_KNOWLEDGE].get_memory()
        except AttributeError:
            # Gold Docs
            available_memory = [[]] * len(observations)
            pass
        batch_reply_sdm, search_indices = self.batch_act_decision(
            observations, Module.SEARCH_DECISION, self.agents[Module.SEARCH_DECISION]
        )
        batch_reply_mdm, memory_indices = self.batch_act_decision(
            observations, Module.MEMORY_DECISION, self.agents[Module.MEMORY_DECISION]
        )
        memory_indices = [i for i in memory_indices if available_memory[i]]
        if self.contextual_knowledge_decision is Decision.ALWAYS:
            contextual_indices = list(range(len(observations)))
        elif self.contextual_knowledge_decision is Decision.NEVER:
            contextual_indices = []
        else:
            assert self.contextual_knowledge_decision is Decision.COMPUTE
            contextual_indices = [
                i
                for i in list(range(len(observations)))
                if i not in memory_indices + search_indices
            ]

        # Second, generate search queries and new memories
        batch_reply_sgm = self.batch_act_sgm(
            observations, search_indices, self.agents[Module.SEARCH_QUERY]
        )
        batch_reply_mgm_partner = self.batch_act_mgm(
            observations=observations, agent=self.agents[Module.MEMORY_GENERATOR]
        )

        # Third, generate the knowledge sentences
        batch_reply_knowledge = self.batch_act_knowledge(
            observations,
            search_indices,
            memory_indices,
            contextual_indices,
            {m: self.agents[m] for m in Module if m.is_knowledge()},
        )

        # Fourth, generate the dialogue response!
        if self.knowledge_conditioning == 'combined':
            batch_reply_dialogue = self.batch_act_dialogue_combined(
                observations, batch_reply_knowledge
            )
        elif self.knowledge_conditioning == 'separate':
            batch_reply_dialogue = self.batch_act_dialogue_separate(
                observations,
                batch_reply_knowledge,
                search_indices,
                memory_indices,
                contextual_indices,
            )
        else:
            assert self.knowledge_conditioning == 'both'
            reply_combined = self.batch_act_dialogue_combined(
                observations, batch_reply_knowledge
            )
            self.reset(clones_only=True)
            reply_separate = self.batch_act_dialogue_separate(
                observations,
                batch_reply_knowledge,
                search_indices,
                memory_indices,
                contextual_indices,
            )
            batch_reply_dialogue = []
            for r_c, r_s in zip(reply_combined, reply_separate):
                reply = r_c
                reply_score = reply['beam_texts'][0][-1]
                max_seperate_score = r_s['max_score']
                if max_seperate_score > reply_score:
                    reply.force_set('text', r_s['text'])
                batch_reply_dialogue.append(reply)

        # Fifth, generate new memories
        batch_reply_mgm_self = self.batch_act_mgm(
            self_messages=batch_reply_dialogue,
            agent=self.agents[Module.MEMORY_GENERATOR],
        )

        # Sixth, combine them all in the srm batch reply.
        final_batch_reply = self.collate_batch_acts(
            batch_reply_sdm,
            batch_reply_mdm,
            batch_reply_sgm,
            batch_reply_mgm_self,
            batch_reply_mgm_partner,
            batch_reply_knowledge,
            batch_reply_dialogue,
            available_memory,
        )

        return final_batch_reply

    def act(self):
        """
        Call batch_act with the singleton batch.
        """
        response = self.batch_act([self.observations])[0]
        self.self_observe(response)
        return response

    def self_observe(self, self_message: Message):
        """
        Override TA.self_observe.

        Make sure all agents have same history.
        """
        self.agents[Module.SEARCH_KNOWLEDGE].self_observe(self_message)
        self.search_knowledge_responses.append(
            self_message.get(Module.SEARCH_KNOWLEDGE.message_name(), '')
        )
        self.contextual_knowledge_responses.append(
            self_message.get(Module.CONTEXTUAL_KNOWLEDGE.message_name(), '')
        )
        self.memory_knowledge_responses.append(
            self_message.get(Module.MEMORY_KNOWLEDGE.message_name(), '')
        )
        for person in ['self', 'partner']:
            if MemoryUtils.is_valid_memory(
                self.memories,
                self_message.get(
                    f'{Module.MEMORY_GENERATOR.message_name()}_{person}', ''
                ),
                MemoryUtils.get_memory_prefix(person, self.MODEL_TYPE),
            ):
                self.memories.append(
                    MemoryUtils.add_memory_prefix(
                        self_message[
                            f'{Module.MEMORY_GENERATOR.message_name()}_{person}'
                        ],
                        person,
                        self.MODEL_TYPE,
                    )
                )
        observation = {
            'text': clean_text(
                self.agents[Module.SEARCH_KNOWLEDGE].history.get_history_str() or ''
            )
        }
        for agent_name, agent in self.agents.items():
            if isinstance(agent, list):
                for a in agent:
                    a.reset()
                    a.history.update_history(
                        observation, temp_history=a.get_temp_history(observation)
                    )
            elif agent_name != Module.SEARCH_KNOWLEDGE.message_name():
                agent.reset()
                agent.history.update_history(
                    observation, temp_history=agent.get_temp_history(observation)
                )
