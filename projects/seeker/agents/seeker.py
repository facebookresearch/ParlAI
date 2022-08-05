#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
SeeKeR Agent.

This agent enables a four stage process:

1) determine if search
2) generate search query
3) generate knowledge response
4) generate dialogue response
"""
from collections import defaultdict
import copy
import torch
from types import MethodType
from typing import List, Tuple, Optional, Dict, Any

from parlai.agents.bart.bart import BartAgent
from parlai.agents.fid.fid import (
    FidAgent,
    GoldDocRetrieverFiDAgent,
    SearchQueryFiDAgent,
    SearchQuerySearchEngineFiDAgent,
    WizIntGoldDocRetrieverFiDAgent,
)
from parlai.agents.rag.args import setup_rag_args
from parlai.agents.rag.retrievers import Document
from parlai.core.agents import Agent, create_agent_from_shared, create_agent
from parlai.core.message import Message
from parlai.core.mutators import MessageMutator, Mutator
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.torch_agent import Batch, Output, History
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
import parlai.utils.logging as logging

from projects.seeker.agents.seeker_modules import ComboFidModel
from projects.seeker.utils import (
    SearchDecision,
    drm_get_batch_context,
    krm_get_batch_context,
    krm_get_batch_context_only_knowledge,
    GENERATE_QUERY,
    IS_SEARCH_REQUIRED,
    DO_SEARCH,
    DO_NOT_SEARCH,
)
from projects.seeker.agents.modular_agent import ModularAgentMixin


class ComboFidAgent(FidAgent):
    """
    The ComboFidAgent is used to *train* a SeeKeR model.

    This agent operates as a standard FiD model when there are retrieved documents.

    Otherwise, it operates as a standard transformer/generator.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        combo_fid = parser.add_argument_group('Combo Fid Group')
        combo_fid.add_argument(
            '--skip-retrieval-key',
            type=str,
            default='skip_retrieval',
            help='key in observation determining whether to skip retrieval.',
        )
        combo_fid.add_argument(
            '--serializable',
            type='bool',
            default=False,
            help='Whether to make model act output fully serializable.',
        )
        combo_fid.add_argument(
            '--force-skip-retrieval',
            type='bool',
            default=False,
            help='If True, we force skip retrieval on any/all incoming examples',
        )

    def build_model(self) -> ComboFidModel:
        """
        Build and return ComboFidModel.
        """
        if self.generation_model == 't5':
            raise RuntimeError('T5 currently not supported')
        else:
            model = ComboFidModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        """
        Overrides FidAgent.batchify to add skip retrieval input vec.

        Additionally adds the prior knowledge responses to the batch. This allows
        context blocking in a KRM setup that includes the prior generations from the
        knowledge component.
        """
        batch = super().batchify(obs_batch, sort)
        valid_exs = [ex for ex in obs_batch if self.is_valid(ex)]
        if valid_exs:
            if self.opt.get('force_skip_retrieval', False):
                skip_retrieval = [True] * len(valid_exs)
            else:
                skip_retrieval = [
                    ex.get(self.opt['skip_retrieval_key'], False) for ex in valid_exs
                ]
            batch.skip_retrieval_vec = torch.BoolTensor(skip_retrieval)
            if any(ex.get('prior_knowledge_responses') for ex in valid_exs):
                vecs, _lens = self._pad_tensor(
                    [
                        self.dict.txt2vec(ex.get('prior_knowledge_responses'))
                        for ex in valid_exs
                    ]
                )
                batch.prior_knowledge_responses_vec = vecs
        return batch

    def _model_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.BoolTensor,
    ]:
        """
        Override FidModel._model_input to add skip_retrieval_vec.
        """
        return (
            batch.text_vec,
            batch.text_vec.ne(self.NULL_IDX).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
            batch.skip_retrieval_vec,
        )

    def get_retrieved_knowledge(self, message: Message) -> List[Document]:
        if message.get('skip_retrieval'):
            return []
        return super().get_retrieved_knowledge(message)

    def eval_step(self, batch: Batch) -> Optional[Output]:
        """
        Add top documents to the output.
        """
        output = TorchGeneratorAgent.eval_step(self, batch)
        if output is not None and not self.opt.get('serializable', False):
            output.top_docs = self.model_api.get_top_docs()
        return output


class ComboFidSearchQueryAgent(ComboFidAgent, SearchQueryFiDAgent):
    pass


class ComboFidGoldDocumentAgent(ComboFidAgent, WizIntGoldDocRetrieverFiDAgent):
    pass


class SeekerAgent(ModularAgentMixin):
    """
    SeeKeR Agent.

    One module performs search query generation, knowledge response generation,
    and dialogue response generation.

    Additionally, may also perform search decision generation.

    To provide arguments to a sub-agent, simply prepend the argument with one of
    `sdm/sqm/krm/drm`.
    """

    knowledge_agent: Agent
    dialogue_agent: Agent

    @classmethod
    def get_additional_agent_args(cls) -> ParlaiParser:
        """
        Return a parser with arguments sourced from several sub models.
        """
        additional_agent_parser = ParlaiParser(add_parlai_args=False)
        BartAgent.add_cmdline_args(additional_agent_parser)
        setup_rag_args(additional_agent_parser)
        GoldDocRetrieverFiDAgent.add_cmdline_args(additional_agent_parser)
        SearchQuerySearchEngineFiDAgent.add_cmdline_args(additional_agent_parser)
        WizIntGoldDocRetrieverFiDAgent.add_cmdline_args(additional_agent_parser)
        ComboFidAgent.add_cmdline_args(additional_agent_parser)
        return additional_agent_parser

    @classmethod
    def add_additional_subagent_args(cls, parser: ParlaiParser) -> ParlaiParser:
        """
        Add additional args for the sub agents.
        """
        additional_agent_parser = cls.get_additional_agent_args()
        for action in additional_agent_parser._actions:
            key = max(action.option_strings, key=lambda x: len(x))
            type = action.type

            for prefix in ['krm', 'drm', 'sqm', 'sdm']:
                parser.add_argument(
                    f'--{prefix}-{key.strip("-")}', type=type, required=False
                )
        return parser

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Command line args for the Seeker Agent.
        """
        cls.add_additional_subagent_args(parser)
        group = parser.add_argument_group('SeeKeR Agent Args')
        group.add_argument(
            '--krm-model', type=str, help='agent (not model file) to load for krm'
        )
        group.add_argument(
            '--drm-model', type=str, help='agent (not model file) to load for krm'
        )
        group.add_argument(
            '--sqm-model', type=str, help='agent (not model file) to load for sqm'
        )
        group.add_argument(
            '--sdm-model', type=str, help='agent (not model file) to load for sdm'
        )
        group.add_argument(
            '--search-decision-control-token',
            type=str,
            default=IS_SEARCH_REQUIRED,
            help='control token for search decision model',
        )
        group.add_argument(
            '--search-decision-do-search-reply',
            type=str,
            default=DO_SEARCH,
            help='control token returned by SDM to indicate searching',
        )
        group.add_argument(
            '--search-decision-dont-search-reply',
            type=str,
            default=DO_NOT_SEARCH,
            help='control token returned by SDM to indicate not searching',
        )
        group.add_argument(
            '--search-query-control-token',
            type=str,
            default=GENERATE_QUERY,
            help='control token for search query model',
        )
        group.add_argument(
            '--knowledge-response-control-token',
            type=str,
            default=None,
            help='control token for knowledge response model',
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
            default=SearchDecision.ALWAYS.value,
            choices=[s.value for s in SearchDecision],
        )
        group.add_argument(
            '--min-knowledge-length-when-search',
            type=int,
            default=-1,
            help='Set the minimum generated knowledge length when searching.',
        )
        group.add_argument(
            '--beam-disregard-knowledge-for-context-blocking',
            type='bool',
            default=False,
            help='If True disregard the knowledge input for DRM context blocking.',
        )
        group.add_argument(
            '--include-knowledge-in-krm-context-blocking',
            type=bool,
            default=True,
            help='If True, put the KRM responses in the context for the KRM.',
        )
        group.add_argument(
            '--exclude-context-in-krm-context-blocking',
            type=bool,
            default=False,
            help='Used in conjunction with --include-knowledge-in-krm-context-blocking. '
            'If specified, only block on the knowledge, and not the concatenation.',
        )
        group.add_argument(
            '--inject-query-string',
            type=str,
            default=None,
            help='If set, this string is appended to all search queries.',
        )
        group.add_argument(
            '--krm-message-mutators',
            type=str,
            default=None,
            help='message mutators for the KRM model',
        )
        group.add_argument(
            '--drm-message-mutators',
            type=str,
            default=None,
            help='message mutators for the DRM model',
        )
        group.add_argument(
            '--search-server', type=str, default=None, help='search server to use.'
        )
        return parser

    def __init__(self, opt, shared=None):
        self.id = 'SeekerAgent'
        self.opt = opt
        one_model = opt.get('all_model_path') or opt.get('model_file')
        assert one_model, "Must specify a model file for this agent."
        for k in ['knowledge_', 'dialogue_', 'search_query_', 'search_decision_']:
            opt[f'{k}response_model_path'] = one_model
        if opt['search_server']:
            opt['krm_search_server'] = opt['search_server']
            if 'override' in opt:
                opt['override']['krm_search_server'] = opt['search_server']

        self._construct_subagent_opts(opt)
        self.search_query_agent = None
        self.search_decision_agent = None

        if not shared:
            self.knowledge_agent = create_agent(
                self.opts['knowledge_agent'], requireModelExists=True
            )
            logging.verbose("options for knowledge agent")
            if logging.logger.isEnabledFor(logging.VERBOSE):
                self.knowledge_agent.opt.log()
        else:
            self.knowledge_agent = create_agent_from_shared(
                shared['knowledge_agent_share']
            )
        self.knowledge_responses = ['__SILENCE__']

        # Monkey-patch 1: KRM Context Blocking (include knowledge responses)
        if opt['include_knowledge_in_krm_context_blocking']:
            orig_fun = self.knowledge_agent._get_batch_context
            if opt['exclude_context_in_krm_context_blocking']:
                self.knowledge_agent._get_batch_context = MethodType(
                    lambda self, batch: krm_get_batch_context_only_knowledge(
                        self, batch, orig_fun
                    ),
                    self.knowledge_agent,
                )
            else:
                self.knowledge_agent._get_batch_context = MethodType(
                    lambda self, batch: krm_get_batch_context(self, batch, orig_fun),
                    self.knowledge_agent,
                )

        # Monkey-patch 2: DRM Context Blocking (exclude knowledge responses)
        self.dialogue_agent = self._init_shared_model('dialogue_agent')
        if opt['beam_disregard_knowledge_for_context_blocking']:
            orig_fun = self.dialogue_agent._get_batch_context
            self.dialogue_agent._get_batch_context = MethodType(
                lambda self, batch: drm_get_batch_context(
                    self, batch, orig_fun=orig_fun
                ),
                self.dialogue_agent,
            )

        # Shared Agents
        self.search_query_agent = self._init_shared_model('search_query_agent')
        self.search_decision_agent = self._init_shared_model('search_decision_agent')

        self.dialogue_agent_clones = [self.dialogue_agent]
        if not shared:
            # the main agent needs batchsize-many clones for handling history during batch act.
            self.dialogue_agent_clones += [
                self.dialogue_agent.clone() for _ in range(opt.get('batchsize', 1) - 1)
            ]

        self._init_mutators(opt)

        # Other Attrs
        self.search_decision = SearchDecision(opt['search_decision'])
        self.min_knowledge_length_when_search = opt['min_knowledge_length_when_search']
        self.inject_query_string = opt.get('inject_query_string', '')

        super().__init__(opt, shared)

    @property
    def history(self) -> History:
        return self.dialogue_agent.history

    @property
    def knowledge_agent_history(self) -> History:
        return self.knowledge_agent.history

    def reset(self):
        """
        Reset all sub agents.
        """
        if self.knowledge_agent:
            self.knowledge_agent.reset()
        for agent in self.dialogue_agent_clones:
            agent.reset()
        if self.search_query_agent:
            self.search_query_agent.reset()
        if self.search_decision_agent:
            self.search_decision_agent.reset()
        self.knowledge_responses = ['__SILENCE__']

    def _init_mutators(self, opt: Opt):
        """
        Initialize mutator objects for sub agents.
        """
        self.krm_mutators = None
        self.drm_mutators = None
        if opt['krm_message_mutators']:
            logging.warning(
                'WARNING: If specifying KRM Mutators, they MUST be message mutators'
            )
            mutator_types = Mutator.load_mutator_types(opt.get('krm_message_mutators'))
            self.krm_mutators = [mutator(opt) for mutator in mutator_types]
        if opt['drm_message_mutators']:
            logging.warning(
                'WARNING: If specifying DRM Mutators, they MUST be message mutators'
            )
            mutator_types = Mutator.load_mutator_types(opt.get('drm_message_mutators'))
            self.drm_mutators = [mutator(opt) for mutator in mutator_types]

    def _init_shared_model(self, opt_key: str):
        """
        Initialize a shared version of the "knowledge" model.

        This just makes sure that each "agent" has the same params, but different
        history objects.

        :param opt_key:
            which sub agent to create with the shared model.
        """
        return super().init_shared_model(self.opts[opt_key], self.knowledge_agent)

    def _construct_subagent_opts(self, opt: Opt):
        """
        Construct opts for each sub agent.

        :param opt:
            original Opt.
        """
        self.opts = {}
        self.opts['init'] = opt
        override_opts = defaultdict(dict)
        agent_mapping = {
            'krm': 'knowledge_agent',
            'drm': 'dialogue_agent',
            'sqm': 'search_query_agent',
            'sdm': 'search_decision_agent',
        }
        for k, v in opt['override'].items():
            k_set = False
            for prefix, ag_opt in agent_mapping.items():
                if k.startswith(f'{prefix}_'):
                    override_opts[ag_opt][k] = v
                    k_set = True
                    break
            if not k_set:
                override_opts['general'][k] = v
        self.opts['override'] = override_opts
        for prefix, key in agent_mapping.items():
            if f'{prefix}_interactive_mode' not in override_opts[key]:
                override_opts[key][f'{prefix}_interactive_mode'] = opt.get(
                    'interactive_mode', False
                )

        for agent in agent_mapping.values():
            filename = opt[f"{agent.replace('agent', 'response')}_model_path"]
            if not filename:
                continue
            self.opts[agent] = self._get_subagent_opt(
                filename=filename,
                specific_override_args=override_opts[agent],
                general_override_args=override_opts['general'],
            )
            self.opts[agent]['model_file'] = filename
            self.opts[agent]['override']['model_file'] = filename

    def _get_subagent_opt(
        self,
        filename: str,
        specific_override_args: Dict[str, Any],
        general_override_args: Dict[str, Any],
    ) -> Opt:
        """
        Given an agent opt, construct the new opt for the agent.

        :param filename:
            opt path
        :param specific_override_args:
            args for the specific agent
        :param general_override_args:
            args specified for all agents
        """
        return super().get_subagent_opt(
            self.opt['datapath'],
            filename,
            specific_override_args,
            general_override_args,
        )

    def share(self):
        shared = super().share()
        shared['knowledge_agent_share'] = self.knowledge_agent.share()
        return shared

    def observe(self, observation: Message) -> Dict[str, Message]:
        """
        Observe in 3 out of the 4 modules.

        :param observation:
            incoming message

        :return self.observation:
            returned observation is actually a dictionary mapping
            agent module name to the corresponding observation
        """
        if not isinstance(observation, Message):
            observation = Message(observation)
        for key in ['label_candidates', 'knowledge']:
            # Delete unnecessarily large keys
            observation.pop(key, '')
        observation.force_set(
            'knowledge_response', observation.get('checked_sentence', '')
        )

        raw_observation = copy.deepcopy(observation)
        # This part is *specifically* for document chunking.
        if self.krm_mutators:
            observation = observation.copy()
            for mutator in self.krm_mutators:
                assert isinstance(mutator, MessageMutator), "not message mutator"
                observation = next(mutator([observation]))

        knowledge_observation = self.knowledge_agent.observe(observation)
        knowledge_observation['prior_knowledge_responses'] = ' '.join(
            self.knowledge_responses
        )
        if observation.get('episode_done'):
            self.knowledge_responses = ['__SILENCE__']
        search_query_observation = None
        if self.search_query_agent:
            sqm_obs = copy.deepcopy(observation)
            if self.opt['search_query_control_token']:
                sqm_obs.force_set(
                    'temp_history', f" {self.opt['search_query_control_token']}"
                )
            sqm_obs.force_set('skip_retrieval', True)
            search_query_observation = self.search_query_agent.observe(sqm_obs)

        search_decision_observation = None
        if (
            self.search_decision_agent
            and self.search_decision is SearchDecision.COMPUTE
        ):
            assert (
                self.search_decision_agent.history.size == 1
            ), "wrong history size! set --sdm-history-size 1"
            sdm_obs = copy.deepcopy(observation)
            if self.opt['search_decision_control_token']:
                sdm_obs.force_set(
                    'temp_history', f" {self.opt['search_decision_control_token']}"
                )
            sdm_obs.force_set('skip_retrieval', True)
            search_decision_observation = self.search_decision_agent.observe(sdm_obs)

        observations = {
            'raw': raw_observation,
            'knowledge_agent': knowledge_observation,
            'search_query_agent': search_query_observation,
            'search_decision_agent': search_decision_observation,
        }
        self.observations = observations
        return observations

    def batch_act_sdm(
        self,
        observations: List[Dict[str, Message]],
        knowledge_agent_observations: List[Message],
    ) -> Tuple[List[Message], List[int], List[Message]]:
        """
        Search Decision batch act.

        :param observations:
            observations for batch act.
        :param knowledge_agent_observations:
            observations to modify with the decision from the search decision agent.

        :return (batch_reply, search_indices, observations):
            batch_reply: reply from the search decision agent
            search_indices: batch indices with which to use search.
            observations: modified knowledge agent observations
        """
        search_indices = []
        batch_reply_sdm = [{} for _ in range(len(knowledge_agent_observations))]
        if self.search_decision is SearchDecision.ALWAYS:
            [o.force_set('skip_retrieval', False) for o in knowledge_agent_observations]
            search_indices = list(range(len(knowledge_agent_observations)))
        elif self.search_decision is SearchDecision.NEVER:
            [o.force_set('skip_retrieval', True) for o in knowledge_agent_observations]
        else:
            assert self.search_decision is SearchDecision.COMPUTE
            assert self.search_decision_agent
            batch_reply_sdm = self.search_decision_agent.batch_act(
                [o['search_decision_agent'] for o in observations]
            )
            for i, reply in enumerate(batch_reply_sdm):
                logging.debug(f"Example {i}: {reply['text']}")
                if reply['text'] == self.opt['search_decision_do_search_reply']:
                    search_indices.append(i)
                    knowledge_agent_observations[i].force_set('skip_retrieval', False)
                elif reply['text'] == self.opt['search_decision_dont_search_reply']:
                    knowledge_agent_observations[i].force_set('skip_retrieval', True)
                else:
                    logging.error(
                        f"SDM Reply: {reply['text']}; defaulting to no search"
                    )
                    knowledge_agent_observations[i].force_set('skip_retrieval', True)

        return batch_reply_sdm, search_indices, knowledge_agent_observations

    def batch_act_sqm(
        self, observations: List[Dict[str, Message]], search_indices: List[int]
    ) -> List[Message]:
        """
        Search Query Generator batch act.

        :param observations:
            list of observations
        :param search_indices:
            list of batch indices for which search is required.

        :return batch_reply:
            return the batch reply from the search query agent
        """
        return super().batch_act_search_query_generation(
            observations,
            [o['search_query_agent'] for o in observations],
            search_indices,
            self.search_query_agent,
            self.knowledge_agent,
            self.inject_query_string,
        )

    def batch_act_krm(
        self,
        observations: List[Dict[str, Message]],
        knowledge_agent_observations: List[Message],
        search_indices: List[int],
    ) -> List[Message]:
        """
        Knowledge Response Model batch act.

        :param observations:
            list of observations
        :param knowledge_agent_observations:
            observations for the knowledge agent.
        :param search_indices:
            list of indices for which we search.
            important for min length generation when search.

        :return batch_reply:
            batch_reply: batch reply from KRM
        """
        old_min_length = self.knowledge_agent.beam_min_length
        batch_reply_krm = [{} for _ in range(len(observations))]

        if search_indices and self.min_knowledge_length_when_search > 0:
            # Need to handle min length for searching
            self.knowledge_agent.beam_min_length = self.min_knowledge_length_when_search
            search_replies = self.knowledge_agent.batch_act(
                [
                    o
                    for i, o in enumerate(knowledge_agent_observations)
                    if i in search_indices
                ]
            )
            self.knowledge_agent.beam_min_length = old_min_length
            non_search_replies = (
                self.knowledge_agent.batch_act(
                    [
                        o
                        for i, o in enumerate(knowledge_agent_observations)
                        if i not in search_indices
                    ]
                )
                if len(search_indices) != len(observations)
                else []
            )
            search_offset = 0
            no_search_offset = 0
            for i in range(len(observations)):
                if i in search_indices:
                    batch_reply_krm[i] = search_replies[search_offset]
                    search_offset += 1
                else:
                    batch_reply_krm[i] = non_search_replies[no_search_offset]
                    no_search_offset += 1
        else:
            batch_reply_krm = self.knowledge_agent.batch_act(
                knowledge_agent_observations
            )

        self.knowledge_agent.beam_min_length = old_min_length

        return batch_reply_krm

    def batch_act_drm(
        self, observations: List[Dict[str, Message]], batch_reply_krm: List[Message]
    ) -> List[Message]:
        """
        Batch act in the DRM.

        Generate knowledge-infused observations (via temp history).
        Then batch act.
        Then observe all of the replies.

        :param observations:
            observations from self.observe
        :param batch_reply_krm:
            batch reply from the KRM module

        :return batch_reply_drm:
            return the reply from the DRM.
        """
        knowledge = [
            reply_knowledge.get('text', '') for reply_knowledge in batch_reply_krm
        ]
        full_text = [
            o['knowledge_agent'].get('full_text', o.get('text', ''))
            for o in observations
        ]
        logging.debug(f'Generated knowledge: {knowledge}')
        dialogue_agent_observations = []
        for i, obs in enumerate(observations):
            drm_obs = copy.deepcopy(obs['raw'])
            drm_obs.force_set(
                'temp_history',
                f"\n{TOKEN_KNOWLEDGE} {batch_reply_krm[i].get('text', '')} {TOKEN_END_KNOWLEDGE}",
            )
            drm_obs.force_set('skip_retrieval', True)
            if not self.dialogue_agent_clones[i].history.get_history_str():
                drm_obs.force_set('text', full_text[i])
            dialogue_agent_observations.append(
                self.dialogue_agent_clones[i].observe(drm_obs)
            )
            assert all(
                t in self.dialogue_agent_clones[i].history.get_history_str()
                for t in [TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE]
            )
        batch_reply_drm = self.dialogue_agent.batch_act(dialogue_agent_observations)
        for i, obs in enumerate(batch_reply_drm):
            self.dialogue_agent_clones[i].self_observe(obs)
            # manually clear
            self.dialogue_agent_clones[i].history.temp_history = None

        return batch_reply_drm

    def batch_act(self, observations: List[Dict[str, Message]]) -> List[Message]:
        """
        Full batch_act pipeline.

        :param observations:
            batchsize-length list of observations from self.observe

        :return reply:
            return batchsize-length list of final replies.
        """
        knowledge_agent_observations = [o['knowledge_agent'] for o in observations]
        # First, determine whether we're searching
        (
            batch_reply_sdm,
            search_indices,
            knowledge_agent_observations,
        ) = self.batch_act_sdm(observations, knowledge_agent_observations)
        # Second, generate search queries
        batch_reply_sqm = self.batch_act_sqm(observations, search_indices)

        # Third, generate the knowledge sentence
        batch_reply_krm = self.batch_act_krm(
            observations, knowledge_agent_observations, search_indices
        )

        # Fourth, generate the dialogue response!
        batch_reply_drm = self.batch_act_drm(observations, batch_reply_krm)

        # Finaly, combine them all in the drm batch reply.
        for sdm, sqm, krm, drm in zip(
            batch_reply_sdm, batch_reply_sqm, batch_reply_krm, batch_reply_drm
        ):
            if drm.is_padding():
                continue
            drm.force_set('search_decision', sdm.get('text', ''))
            drm.force_set('search_query', sqm.get('text', ''))
            drm.force_set('knowledge_response', krm.get('text', ''))
            docs = krm.get('top_docs', [Document("", "", "")])
            drm.force_set('doc_titles', [d.get_title() for d in docs])
            drm.force_set('doc_content', [d.get_text() for d in docs])
            drm.force_set('doc_urls', [d.get_id() for d in docs])

        return batch_reply_drm

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

        Make sure that knowledge agent and other agents have the same history.

        This eliminates unnecessary copies of the previous knowledge in the history.
        """
        self.knowledge_agent.self_observe(self_message)
        self.knowledge_responses.append(self_message.get('knowledge_response', ''))
        observation = {'text': self.knowledge_agent.history.get_history_str()}
        for agent in self.dialogue_agent_clones:
            agent.reset()
            agent.history.update_history(
                observation,
                temp_history=self.dialogue_agent.get_temp_history(observation),
            )
        if (
            self.search_decision_agent
            and self.search_decision is SearchDecision.COMPUTE
        ):
            self.search_decision_agent.self_observe(self_message)
            self.search_decision_agent.history.reset()
            self.search_decision_agent.history.update_history(
                observation,
                temp_history=self.search_decision_agent.get_temp_history(observation),
            )
        if self.search_query_agent:
            self.search_query_agent.self_observe(self_message)
            self.search_query_agent.history.reset()
            self.search_query_agent.history.update_history(
                observation,
                temp_history=self.search_query_agent.get_temp_history(observation),
            )
