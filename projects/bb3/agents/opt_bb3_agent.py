#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BB3 with OPT-175B Base.

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
import copy
import random
from typing import List, Optional, Dict, Any, Union, Tuple
from parlai.agents.rag.retrievers import Document

from parlai.core.agents import Agent, create_agent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import parlai.utils.logging as logging

from projects.bb3.agents.module import Module
from projects.bb3.agents.r2c2_bb3_agent import BlenderBot3Agent as R2C2Agent
from projects.bb3.agents.opt_api_agent import BB3OPTAgent
from projects.bb3.agents.search_agent import SearchAgent
from projects.bb3.agents.utils import (
    Decision,
    APIUtils,
    MemoryUtils,
    is_opener,
    DisplayUtils,
    set_failed_reply,
)
import projects.bb3.prompts as PROMPT


BB3_TO_OPT_KEYS = {
    'num_shots': 'num_shots',
    'opt_server': 'server',
    'include_prompt': 'include_prompt',
    'all_vanilla_prompt': 'all_vanilla_prompt',
    'metaseq_max_retry_api': 'max_retry_api',
    'metaseq_server_timeout': 'server_timeout',
    'memory_decision_use_memories': 'memory_decision_use_memories',
}

BB3_OPT_GEN_KEYS = [
    'prompt',
    'raw_prompt',
    'inference',
    'skip_generation',
    'beam_size',
    'beam_min_length',
    'beam_max_length',
    'topp',
    'temperature',
    'server',
    'omega_bound',
    'lambda_decay',
    'max_retry_api',
    'server_timeout',
    'alpha_presence',
    'alpha_frequency',
    'alpha_presence_src',
    'alpha_frequency_src',
]


class BlenderBot3Agent(R2C2Agent):
    """
    OPT BB3 Agent.
    """

    MODEL_TYPE: str = 'OPT'

    @classmethod
    def get_additional_agent_args(cls) -> ParlaiParser:
        parser = R2C2Agent.get_additional_agent_args()
        return BB3OPTAgent.add_cmdline_args(parser)

    def _apply_context_blocking_patches(self):
        """
        Don't do anything here.
        """
        pass

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Command line args for OPT BB3.
        """
        cls.add_additional_subagent_args(parser)
        parser = R2C2Agent.add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group('OPT Agent Group')
        group.add_argument(
            '--opt-server',
            type=str,
            help='which server to connect to for hosted OPT model',
            default=None,
        )
        group.add_argument(
            '--raw-search-server',
            type=str,
            help='specify a search server address.',
            default=None,
        )
        group.add_argument(
            '--num-shots',
            type=int,
            default=None,
            help='how many k-shot examples to put in the prompt. Default <0 is all',
        )
        group.add_argument(
            '--include-prompt',
            type='bool',
            default=None,
            help='Whether to include prompt',
        )
        group.add_argument(
            '--all-vanilla-prompt',
            type='bool',
            default=None,
            help='If True, and --include-prompt, then all prompts are just the vanilla, "a conversation", prompts.',
        )
        group.add_argument(
            '--knowledge-chunk-size',
            type=int,
            default=100,
            help='Chunk size, in words, of knowledge to keep',
        )
        group.add_argument('--debug-bb3', type='bool', default=False, dest='debug_bb3')
        group.add_argument(
            '--max-prompt-len',
            type=int,
            default=PROMPT.MAX_PROMPT_LEN,
            help='Longest sequence to send to API',
        )
        parser.add_argument(
            '--metaseq-max-retry-api',
            default=-1,
            type=int,
            help='Number of times to retry on API request failures (< 0 for unlimited retry).',
        )
        parser.add_argument('--metaseq-server-timeout', default=20.0, type=float)
        return parser

    def __init__(self, opt, shared=None):
        opt['model_file'] = 'zoo:'
        super().__init__(opt, shared)
        # Always init search agent
        if not shared:
            agent_opts = self.opts[Module.SEARCH_KNOWLEDGE]
            top_agent = self._init_top_agent(agent_opts)
            # two mappings; one for code access, one for debug access.
            self.agents[Module.SEARCH_KNOWLEDGE.agent_name()] = top_agent
            self.agents[Module.SEARCH_KNOWLEDGE] = top_agent
        else:
            top_agent = shared['top_agent'].clone()
            self.agents[Module.SEARCH_KNOWLEDGE.agent_name()] = top_agent
            self.agents[Module.SEARCH_KNOWLEDGE] = top_agent

        self.dictionary = top_agent.dictionary
        # continue
        self.max_prompt_len = opt.get('max_prompt_len', PROMPT.MAX_PROMPT_LEN)
        self.search_agent = SearchAgent(
            {
                'server': self.opt.get('search_server', 'default'),
                'raw_server': self.opt.get('raw_search_server', None),
                'intra_doc_delimiter': ' ',
                'n_docs': 5,
                'search_server_timeout': opt.get('search_server_timeout', 0),
            }
        )
        self.vanilla = all(
            decision is Decision.NEVER
            for decision in [
                self.contextual_knowledge_decision,
                self.search_decision,
                self.memory_decision,
            ]
        )
        self.batch_agents = {m: self.agents[m].clone() for m in Module}
        top_batch_agent = self.batch_agents[Module.SEARCH_KNOWLEDGE]
        for key in BB3_OPT_GEN_KEYS:
            for agent in self.batch_agents.values():
                if key not in agent.opt:
                    logging.debug(f"Overriding {key} to {top_batch_agent.opt[key]}")
                    agent.opt[key] = top_batch_agent.opt[key]

    def share(self):
        shared = super().share()
        shared['top_agent'] = self.agents[Module.SEARCH_KNOWLEDGE]
        return shared

    def _init_top_agent(self, opt: Opt) -> Agent:
        opt.pop('model_file', '')
        return create_agent(opt)

    def _construct_subagent_opts(self, opt: Opt):
        """
        Override to set prompts.
        """
        super()._construct_subagent_opts(opt)
        for m in Module:
            self.opts[m]['raw_prompt'] = "DUMMY"
            self.opts[m]['api_key'] = APIUtils.DEFAULT_KEY
            for self_k, agent_k in BB3_TO_OPT_KEYS.items():
                if opt[self_k] is not None:
                    old_val = self.opts[m].get(agent_k)
                    self.opts[m][agent_k] = opt[self_k]
                    logging.debug(
                        f"Overriding {agent_k} for {m} from {old_val} to {opt[self_k]}"
                    )

    def opening_token(self):
        return PROMPT.OPENING_PREFIX

    def get_mdm_observation(self, ag_obs: Message) -> Message:
        """
        Add memories to the memory decision observation.

        :param ag_obs:
            incoming mdm observation

        :return mdm_obs:
            return mdm observation with memories in the context.
        """
        self.agents[Module.MEMORY_DECISION].reset()

        original_text = ag_obs['text']
        self_memory_text, partner_memory_text = '', ''
        self_memories = [
            m.replace(f"{PROMPT.SELF_MEMORY_PREFIX}: ", '')
            for m in self.memories
            if m.startswith(PROMPT.SELF_MEMORY_PREFIX)
        ]
        if self_memories:
            self_memory_text = f"{PROMPT.MEMORY_KNOWLEDGE_PREFIX}: {PROMPT.SELF_MEMORY_PREFIX}: {' '.join(self_memories)}\n"
        partner_memories = [
            m.replace(f"{PROMPT.PARTNER_MEMORY_PREFIX}: ", '')
            for m in self.memories
            if m.startswith(PROMPT.PARTNER_MEMORY_PREFIX)
        ]
        if partner_memories:
            partner_memory_text = f"{PROMPT.MEMORY_KNOWLEDGE_PREFIX}: {PROMPT.PARTNER_MEMORY_PREFIX}: {' '.join(partner_memories)}\n"

        new_text = f"{self_memory_text}{partner_memory_text}{original_text}"
        ag_obs.force_set('text', new_text)
        return ag_obs

    def get_orm_observation(
        self, observation: Message, opening_memories: List[str]
    ) -> Message:
        """
        Return the appropriate ORM observation.

        :param observation:
            raw observation
        :param opening_memories:
            memories seeded to the model

        :return observation:
            return the ORM observation
        """
        agent = self.agents[Module.OPENING_DIALOGUE]
        agent.reset()
        for i, mem in enumerate(opening_memories):
            mem = MemoryUtils.maybe_add_memory_prefix(mem, 'partner', self.MODEL_TYPE)
            opening_memories[i] = mem

        new_obs = copy.deepcopy(observation)
        new_obs.force_set(
            'text', self._check_and_limit_len('\n'.join(opening_memories))
        )
        new_obs.force_set('memories', opening_memories)

        return agent.observe(new_obs)

    def observe(self, observation: Message) -> Dict[Module, Message]:
        # handle passed memories as well
        observation = Message(observation)
        opening_memories = observation.pop('memories', None)
        observations = super().observe(observation)
        for m in Module.dialogue_modules():
            ag_obs = copy.deepcopy(observation)
            observations[m] = self.agents[m].observe(ag_obs)
        if is_opener(observation['text'], opening_memories):
            orm_obs = self.get_orm_observation(observation, opening_memories)
            self.memories = orm_obs['memories']
            observations[Module.OPENING_DIALOGUE] = orm_obs
        else:
            observations[Module.OPENING_DIALOGUE] = Message({})

        if self.opt['debug_bb3']:
            DisplayUtils.display_observations(observations)
        return observations

    def _get_subagent_opt(
        self,
        filename: str,
        specific_override_args: Dict[str, Any],
        general_override_args: Dict[str, Any],
    ) -> Opt:
        # Remove the prefix for the model for the specific override args.
        specific_override_args = {
            '_'.join(k.split('_')[1:]): v for k, v in specific_override_args.items()
        }

        return {**general_override_args, **specific_override_args, 'override': {}}

    def get_search_results(
        self, batch_reply_sgm: List[Message], search_indices: List[int]
    ) -> Tuple[List[List[str]], List[List[Document]]]:
        """
        Retrieve search results; return documents as well.

        :param batch_reply_sgm:
            batch reply with the search queries
        :param search_indices:
            indices into batch in which we search.

        :return results, docs:
            return the search results in raw string form
            return the top documents as Documents
        """
        search_queries: List[str] = [
            o['text'] if i in search_indices else ''
            for i, o in enumerate(batch_reply_sgm)
        ]
        search_results: List[List[str]] = []
        top_docs: List[List[Document]] = []
        for q in search_queries:
            if q:
                search_results.append(
                    self.search_agent.respond({'text': q}).split('\n')
                )
                top_docs.append(self.search_agent.top_docs)
                # clear the top_docs on search agent side
                self.search_agent.top_docs = []
            else:
                search_results.append([])
                top_docs.append([])

        for i, results_i in enumerate(search_results):
            # chunk docs; save only first 500 chars of each document
            for j, d in enumerate(results_i):
                results_i[j] = ' '.join(
                    d.split(' ')[: self.opt['knowledge_chunk_size']]
                )
            print_results = '\t\n\n'.join(
                [
                    f'Document {j}:' + ' '.join(r.split()[:50])
                    for j, r in enumerate(results_i)
                ]
            )
            logging.debug(f"Search Results (50 toks each) for {i}:\n {print_results}")

        return search_results, top_docs

    def _check_and_limit_len(self, text: str) -> str:
        # check knowledge prompt len to make sure we don't overdo it.
        while len(self.dictionary.txt2vec(text)) >= self.max_prompt_len:
            text = '\n'.join(text.split('\n')[1:])
        return text

    def batch_act_knowledge(
        self,
        observations: List[Dict[Union[str, Module], Message]],
        search_indices: List[int],
        memory_indices: List[int],
        contextual_indices: List[int],
        batch_agents: Dict[Module, Agent],
        available_memory: List[List[str]],
        search_results: List[List[str]],
        top_docs: List[List[Document]],
        top_memories: List[List[str]],
    ) -> List[Message]:
        """
        Add prompts to knowledge observations before calling batch act.

        :param observations:
            raw observations
        :param search_indices:
            indices in which we use search knowledge
        :param memory_indices:
            indices in which we use memories
        :param contextual_indices:
            indices in which we use an entity from the context
        :param batch_agents:
            mapping from module to agent with which to call batch act
        :param available_memory:
            memories per batch example
        :param search_results:
            search results per batch example
        :param top_docs:
            top docs corresponding to the search results
        :param top_memories:
            retrieved memories.

        :return batch_reply_knowledge:
            return batch reply from knowledge modules.
        """
        for i, all_obs in enumerate(observations):
            for module in Module:
                obs = all_obs[module]
                if module is Module.MEMORY_KNOWLEDGE and i in memory_indices:
                    memories = MemoryUtils.get_available_memory(
                        all_obs['raw'], self.dictionary
                    )
                    memories = '\n'.join(available_memory[i])
                    new_prompt = self._check_and_limit_len(
                        obs['prompt'].replace(module.opt_pre_context_tok(), memories)
                    )
                    obs.force_set('prompt', new_prompt)
                elif module is Module.SEARCH_KNOWLEDGE and i in search_indices:
                    results = (
                        f"{PROMPT.EXTERNAL_KNOWLEDGE_PREFIX}: "
                        + f'\n{PROMPT.EXTERNAL_KNOWLEDGE_PREFIX}: '.join(
                            [s.replace('\n', '') for s in search_results[i]]
                        )
                    )
                    new_prompt = self._check_and_limit_len(
                        obs['prompt'].replace(module.opt_pre_context_tok(), results)
                    )
                    obs.force_set('prompt', new_prompt)

        batch_reply_knowledge = super().batch_act_knowledge(
            observations,
            search_indices,
            memory_indices,
            contextual_indices,
            batch_agents,
            top_docs=top_docs,
            top_memories=top_memories,
        )
        # re-assign memories
        for i, reply in enumerate(batch_reply_knowledge):
            if i in memory_indices:
                text = reply[Module.MEMORY_KNOWLEDGE.message_name()].strip()
                do_split = True
                if any(
                    text.startswith(p)
                    for p in [PROMPT.PARTNER_MEMORY_PREFIX, PROMPT.SELF_MEMORY_PREFIX]
                ):
                    # no need to split the memory to find it
                    do_split = False
                true_memory = ''
                for memory in available_memory[i]:
                    raw_mem = (
                        MemoryUtils.split_prefix_memory(memory)[-1]
                        if do_split
                        else memory
                    )
                    if raw_mem == text:
                        true_memory = memory
                        break
                if true_memory:
                    reply.force_set(Module.MEMORY_KNOWLEDGE.message_name(), true_memory)
                else:
                    logging.debug(
                        f"Memory {text} is invalid, given {available_memory[i]}"
                    )
                    reply.pop(Module.MEMORY_KNOWLEDGE.message_name(), None)
                    reply.pop(f"{Module.MEMORY_KNOWLEDGE.message_name()}_docs", None)

        return batch_reply_knowledge

    def detect_and_handle_failures(
        self, module: Module, messages: List[Message], batch_reply_final: List[Message]
    ) -> List[int]:
        """
        Return indices of messages that have failures.

        :param messages:
            candidate list of Messages
        :param module:
            module in which failure may have occurred
        :param batch_reply_final:
            final batch replies

        :return indices:
            return indices of messages with failures
        """
        failed: List[int] = []
        for i, message in enumerate(messages):
            if APIUtils.is_request_failed_response(message):
                batch_reply_final[i]['failures'].extend(message['failures'])
                batch_reply_final[i][f"{module.message_name()}_failures"] = message[
                    'failures'
                ]
                failed.append(i)
        return failed

    def detect_and_handle_final_failures(
        self, messages: List[Message]
    ) -> List[Message]:
        """
        Handle failures of final dialogue replies.

        We get here if the whole pipeline collapses.
        """
        for reply in messages:
            if APIUtils.is_request_failed_response(reply):
                set_failed_reply(reply)
        return messages

    def batch_act_dialogue(
        self,
        observations: List[Dict[Union[str, Module], Message]],
        batch_reply_knowledge: List[Message],
    ) -> List[Message]:
        """
        Batch act for dialogue!

        If we have failures of knowledge conditioning, fall back to vanilla dialogue.

        :param observations:
            raw observations per module
        :param batch_reply_knowledge:
            list of knowledge replies

        :return batch_act_dialogue:
            return the dialogue replies!
        """
        # construct dialogue obs with knowledge conditioning
        dialogue_obs, dialogue_replies = [], []
        for i, reply in enumerate(batch_reply_knowledge):
            for k_module in Module.knowledge_modules():
                if not reply.get(
                    k_module.message_name()
                ) or APIUtils.METASEQ_FAIL_MESSAGE_TEXT in reply.get(
                    k_module.message_name(), ''
                ):
                    continue
                logging.debug(
                    f'{k_module.message_name()}: {reply[k_module.message_name()]}'
                )
                d_module = None
                if k_module is Module.CONTEXTUAL_KNOWLEDGE:
                    d_module = Module.CONTEXTUAL_DIALOGUE
                elif k_module is Module.MEMORY_KNOWLEDGE:
                    d_module = Module.MEMORY_DIALOGUE
                else:
                    assert k_module is Module.SEARCH_KNOWLEDGE
                    d_module = Module.SEARCH_DIALOGUE
                d_obs = Message(observations[i][d_module])
                prompt = d_obs['prompt'].replace(
                    d_module.opt_post_context_tok(),
                    f"{d_module.opt_dialogue_knowledge_prefix()}{reply[k_module.message_name()]}",
                )
                d_obs.force_set('prompt', prompt)
                dialogue_obs.append((i, d_module, d_obs))

        # check if we have failures; if so, resort to vanilla dialogue
        all_inds = set([o[0] for o in dialogue_obs])
        for i in range(len(observations)):
            if i not in all_inds:
                dialogue_obs.append(
                    (
                        i,
                        Module.VANILLA_DIALOGUE,
                        Message(observations[i][Module.VANILLA_DIALOGUE]),
                    )
                )

        if self.opt['debug_bb3']:
            DisplayUtils.display_observations({o[1]: o[2] for o in dialogue_obs})

        # check knowledge conditioning; combine observations if so
        if self.knowledge_conditioning == 'separate' or len(dialogue_obs) == 1:
            dialogue_replies = self.batch_agents[Module.SEARCH_DIALOGUE].batch_act(
                [o[-1] for o in dialogue_obs]
            )
        elif self.knowledge_conditioning == 'combined':
            logging.debug('combining all knowledge')
            combined_obs: List[Message] = []
            for obs_i in range(len(observations)):
                extra_knowledge = []
                dialogue_obs_i = [d for d in dialogue_obs if d[0] == obs_i]
                primary_obs = dialogue_obs_i[0]
                for (_, _, obs) in dialogue_obs_i[1:]:
                    extra_knowledge.append(obs['prompt'].split('\n')[-2])
                old_prompt = primary_obs[-1]['prompt'].split('\n')
                primary_obs[-1].force_set(
                    'prompt',
                    '\n'.join(old_prompt[:-1] + extra_knowledge + old_prompt[-1:]),
                )
                combined_obs.append(primary_obs)
            dialogue_replies = self.batch_agents[Module.SEARCH_DIALOGUE].batch_act(
                [o[-1] for o in combined_obs]
            )
            dialogue_obs = combined_obs
        else:
            raise NotImplementedError('Both is not implemented')

        batch_reply_dialogue = [Message({}) for _ in range(len(observations))]

        # collate dialogue replies
        for (i, d_module, _), reply in zip(dialogue_obs, dialogue_replies):
            if APIUtils.is_request_failed_response(reply):
                batch_reply_dialogue[i] = reply
                continue
            if self.knowledge_conditioning == 'combined':
                logging.debug(f'combined dialogue reply: {reply}')
                assert 'text' not in batch_reply_dialogue[i]
                batch_reply_dialogue[i]['text'] = reply['text']
                batch_reply_dialogue[i]['score'] = reply['logprobs']
                batch_reply_dialogue[i]['best_module'] = 'Combined Response'
            else:
                logging.debug(f'{d_module.message_name()}: {reply}')
            batch_reply_dialogue[i][d_module.message_name()] = reply['text']
            batch_reply_dialogue[i][f"{d_module.message_name()}_score"] = reply[
                'logprobs'
            ]

        # if separate, make sure we set text to highest scoring object
        if self.knowledge_conditioning == 'separate':
            for reply in batch_reply_dialogue:
                if APIUtils.is_request_failed_response(reply):
                    continue
                scores = []
                for module in Module.dialogue_modules():
                    scores.append(
                        (
                            module,
                            reply.get(f"{module.message_name()}_score", float('inf')),
                        )
                    )
                best_module = min(scores, key=lambda x: x[-1])[0]
                reply['text'] = reply[best_module.message_name()]
                reply['score'] = reply[f"{best_module.message_name()}_score"]
                reply['best_module'] = best_module

        return batch_reply_dialogue

    def batch_act_decision(
        self,
        observations: List[Dict[Union[str, Module], Message]],
        module: Module,
        batch_reply_final: List[Message],
    ) -> Tuple[List[Message], List[int]]:
        """
        Override super class to handle failures.
        """
        agent = self.batch_agents[module]
        batch_reply, indices = super().batch_act_decision(observations, module, agent)
        failed = self.detect_and_handle_failures(module, batch_reply, batch_reply_final)
        indices = [i for i in indices if i not in failed]
        return batch_reply, indices

    def get_opening(
        self, observations: List[Dict[Union[str, Module], Message]]
    ) -> List[Message]:
        """
        Get the opening message.

        :param observations:
            raw observations

        :return batch_act:
            return batch act from the opening dialogue agent.
        """
        module = Module.OPENING_DIALOGUE
        batch_act = [Message({'text': PROMPT.SELF_PREFIX})] * len(observations)

        def _failed_messages(replies):
            return any(
                p in o['text']
                for p in [PROMPT.SELF_PREFIX, PROMPT.PARTNER_PREFIX]
                for o in replies
            )

        retries = 0
        opening_obs = [o[module] for o in observations]
        while _failed_messages(batch_act) and retries < 3:
            batch_act = self.batch_agents[Module.OPENING_DIALOGUE].batch_act(
                opening_obs
            )
            self.batch_agents[Module.OPENING_DIALOGUE].reset()
            retries += 1
            n_mems = [min(1, len(obs['memories']) // 3) for obs in opening_obs]
            for i, o in enumerate(opening_obs):
                o.force_set('memories', random.sample(o['memories'], n_mems[i]))
        if _failed_messages(batch_act):
            for reply in batch_act:
                text = reply.pop('text')
                for p in [
                    PROMPT.SELF_MEMORY_PREFIX,
                    PROMPT.PARTNER_MEMORY_PREFIX,
                    PROMPT.SELF_PREFIX,
                    PROMPT.PARTNER_PREFIX,
                ]:
                    text = text.replace(f"{p}:", '').replace(p, '')
                reply['text'] = text

        return batch_act

    def batch_act_simple(
        self,
        observations: List[Dict[Union[str, Module], Message]],
        module: Module,
    ) -> List[Message]:
        """
        Return either vanilla batch act or opening batch act.

        :param observations:
            list of observations
        :param module:
            module to perform simple batch act with

        :return batch_act:
            return batch act from appropriate agent module.
        """
        batch_agent = self.batch_agents[module]
        if module is Module.OPENING_DIALOGUE:
            batch_act = self.get_opening(observations)
        else:
            batch_act = batch_agent.batch_act([o[module] for o in observations])

        batch_reply_final = [Message({}) for _ in range(len(observations))]
        failed = self.detect_and_handle_failures(module, batch_act, batch_reply_final)

        for i, reply in enumerate(batch_act):
            if i in failed:
                set_failed_reply(reply)
            else:
                reply[module.message_name()] = reply['text']
                reply[f"{module.message_name()}_score"] = reply.get(
                    'logprobs', float('inf')
                )
        batch_reply_final = self.collate_batch_acts(
            batch_reply_mdm=[Message({})] * len(observations),
            batch_reply_sdm=[Message({})] * len(observations),
            batch_reply_mgm_partner=[Message({})] * len(observations),
            batch_reply_mgm_self=[Message({})] * len(observations),
            batch_reply_sgm=[Message({})] * len(observations),
            batch_reply_knowledge=[Message({})] * len(observations),
            batch_reply_dialogue=batch_act,
            available_memory=[o['raw']['memories'] for o in observations],
        )
        return self.detect_and_handle_final_failures(batch_reply_final)

    def batch_act(
        self, observations: List[Dict[Union[str, Module], Message]]
    ) -> List[Message]:
        """
        Override batch act for OPT BB3.
        """
        if self.vanilla:
            return self.batch_act_simple(observations, Module.VANILLA_DIALOGUE)

        if all(
            is_opener(o['raw']['text'], o[Module.OPENING_DIALOGUE].get('memories'))
            for o in observations
        ):
            # We're opening!
            return self.batch_act_simple(observations, Module.OPENING_DIALOGUE)

        batch_reply_final = [
            Message({'id': 'BlenderBot3', 'failures': []})
            for _ in range(len(observations))
        ]
        # Step 1: determine whether we're searching or accessing memory
        available_memory = [o['raw']['memories'] for o in observations]

        batch_reply_sdm, search_indices = self.batch_act_decision(
            observations,
            Module.SEARCH_DECISION,
            batch_reply_final,
        )
        batch_reply_mdm, memory_indices = self.batch_act_decision(
            observations,
            Module.MEMORY_DECISION,
            batch_reply_final,
        )

        memory_indices = [i for i in memory_indices if available_memory[i]]
        contextual_indices = [
            i
            for i in range(len(observations))
            if (
                self.contextual_knowledge_decision is Decision.ALWAYS
                or (
                    self.contextual_knowledge_decision is Decision.COMPUTE
                    and i not in search_indices + memory_indices
                )
            )
            and self.contextual_knowledge_decision is not Decision.NEVER
        ]
        all_inds = (
            set(contextual_indices)
            .union(set(search_indices))
            .union(set(memory_indices))
        )
        if not all(i in all_inds for i in range(len(observations))):
            # something went terribly, terribly wrong. Vanilla batch act.
            return self.batch_act_simple(observations, Module.VANILLA_DIALOGUE)
        # Step 2: Generate search queries, if necessary. Also generate memories for partner
        batch_reply_mgm_partner = self.batch_act_mgm(
            observations=observations, agent=self.batch_agents[Module.MEMORY_GENERATOR]
        )
        batch_reply_sgm = self.batch_act_sgm(
            observations, search_indices, self.batch_agents[Module.SEARCH_QUERY]
        )
        search_indices = [
            i
            for i in search_indices
            if i
            not in self.detect_and_handle_failures(
                Module.SEARCH_QUERY, batch_reply_sgm, batch_reply_final
            )
        ]

        search_results, top_docs = self.get_search_results(
            batch_reply_sgm, search_indices
        )
        invalid_search_results = [i for i in search_indices if not top_docs[i]]
        contextual_indices += invalid_search_results
        search_indices = [i for i in search_indices if i not in invalid_search_results]

        # step 3: memory/knowledge/contextual access.
        # for knowledge obs, we need to fill in context with external or persona knowledge
        batch_reply_knowledge = self.batch_act_knowledge(
            observations,
            search_indices,
            memory_indices,
            contextual_indices,
            {m: self.batch_agents[m] for m in Module if m.is_knowledge()},
            available_memory,
            search_results,
            top_docs,
            [
                [Document(docid='', text=memory, title='') for memory in memories]
                for memories in available_memory
            ],
        )

        # step 4: memory/knowledge/contextual dialogue
        batch_reply_dialogue = self.batch_act_dialogue(
            observations, batch_reply_knowledge
        )
        batch_reply_mgm_self = self.batch_act_mgm(
            self_messages=[
                self.agents[Module.MEMORY_GENERATOR].observe(r)
                for r in batch_reply_dialogue
            ],
            agent=self.batch_agents[Module.MEMORY_GENERATOR],
        )

        batch_reply_final = self.collate_batch_acts(
            batch_reply_sdm,
            batch_reply_mdm,
            batch_reply_sgm,
            batch_reply_mgm_self,
            batch_reply_mgm_partner,
            batch_reply_knowledge,
            batch_reply_dialogue,
            available_memory,
        )
        for i, reply in enumerate(batch_reply_final):
            reply.force_set('id', 'BlenderBot3')
            if i in invalid_search_results:
                reply.force_set('search_result_failure', True)
            if self.opt['debug_bb3']:
                DisplayUtils.display_act(reply)

        batch_reply_final = self.detect_and_handle_final_failures(batch_reply_final)

        return batch_reply_final

    def self_observe(self, self_message: Message):
        logging.debug(f"Self-observing message: {self_message}")
        if APIUtils.is_request_failed_response(self_message):
            return

        for m, agent in self.agents.items():
            if isinstance(m, Module):
                logging.debug(f'Self-observing for module {m}')
                agent.self_observe(self_message)
        if self.vanilla:
            return
        memory_key = Module.MEMORY_GENERATOR.message_name()
        for person in ['self', 'partner']:
            memory_candidate = self_message.get(f"{memory_key}_{person}")
            if not memory_candidate:
                continue
            if MemoryUtils.is_valid_memory(
                self.memories,
                memory_candidate,
                MemoryUtils.get_memory_prefix(person, self.MODEL_TYPE),
            ):
                self.memories.append(
                    MemoryUtils.add_memory_prefix(
                        memory_candidate, person, self.MODEL_TYPE
                    )
                )
