#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import torch
from typing import Optional, List, Dict, Any

from parlai.core.agents import Agent, create_agent
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import PPLMetric
import parlai.utils.logging as logging

from projects.safety_bench.utils.wrapper_loading import register_model_wrapper

from projects.bb3.agents.module import Module
import projects.bb3.prompts as PROMPT

from projects.bb3.agents.utils import APIUtils

PROMPTS = {
    "convai2": "A conversation between two persons. Person 2's personality is described.\n\n",
    "wow": "A conversation between two persons. Person 1 is trying to learn about ",
    "none": None,  # no prompt
}


class SimplePromptHistory(object):
    SPEAKER_SELF = "Person 2"
    SPEAKER_OTHER = "Person 1"

    def __init__(self, prompt: Optional[str] = None):
        self.turns = []
        self.prompt = prompt
        self._will_clear = False

    def observe_self(self, text):
        self.turns.append(f'{self.SPEAKER_SELF}: {text}')

    def observe_other(self, text: str):
        assert text is not None
        lines = text.split("\n")
        assert len(lines) >= 1
        while lines and 'your persona' in lines[0]:
            self.turns.append(lines.pop(0))
        end_of_context = len(self.turns)
        speakers = itertools.cycle([self.SPEAKER_OTHER, self.SPEAKER_SELF])
        for speaker, line in zip(speakers, lines[::-1]):
            if '__SILENCE__' in line:
                continue
            self.turns.insert(end_of_context, f'{speaker}: {line}')

    def render_prompt(self) -> str:
        flattened = "\n".join(self.turns + [f'{self.SPEAKER_SELF}:'])
        if self.prompt:
            flattened = f'{self.prompt}{flattened}'
        return flattened

    def get_history_str(self):
        return '\n'.join(self.turns)

    def clear(self):
        self.turns.clear()
        self._will_clear = False

    def prepare_clear(self):
        self._will_clear = True

    def finish_clear(self):
        if self._will_clear:
            self.clear()


class BB3PromptHistory(SimplePromptHistory):
    def __init__(
        self,
        prompt: Optional[str] = None,
        opt: Optional[Opt] = None,
        dictionary: Optional[DictionaryAgent] = None,
    ):
        super().__init__(prompt)
        assert opt is not None
        self.add_speaker_prefixes = opt.get('add_speaker_prefixes', True)
        self.max_prompt_len = opt.get('max_prompt_len', PROMPT.MAX_PROMPT_LEN)
        self.module = Module(opt['module'])
        self.dictionary = dictionary
        self.memories_included = self.module is Module.MEMORY_DECISION and opt.get(
            'memory_decision_use_memories', False
        )
        self.one_turn_history = (
            self.module.is_one_turn_history() and not self.memories_included
        )
        self.prompt = ''
        if opt['include_prompt']:
            if opt['all_vanilla_prompt']:
                self.prompt = Module(Module.VANILLA_DIALOGUE).opt_prompt()
            else:
                self.prompt = self.module.opt_prompt()

        self.shots = self.module.opt_shots()
        if opt.get('num_shots') is not None:
            effective_shots = opt['num_shots'] if opt['num_shots'] >= 0 else 100000
            if effective_shots == 0:
                self.shots = ''
            elif effective_shots > 0:
                self.shots = (
                    '\n\n'.join(self.shots.split('\n\n')[:effective_shots]) + '\n\n'
                )
        self.final_prefix = f"{self.module.opt_final_prefix()}:"
        self.pre_context_tok = self.module.opt_pre_context_tok()
        self.post_context_tok = self.module.opt_post_context_tok()
        self.style_string = (
            f"{PROMPT.STYLE_PREFIX}: {opt['insert_style']}"
            if opt.get('insert_style')
            else ''
        )
        self.debug = opt.get('debug_bb3', False)
        if not opt.get('include_substitution_tokens', True):
            self.pre_context_tok = ''
            self.post_context_tok = ''

    def observe_self(self, text):
        if self.add_speaker_prefixes:
            super().observe_self(text)
            return

        self.turns.append(text)

    def observe_other(self, text: str):
        assert text is not None
        lines = text.split("\n")
        assert len(lines) >= 1
        while lines and any(
            p in lines[0]
            for p in [
                'your persona',
                PROMPT.SELF_MEMORY_PREFIX,
                PROMPT.PARTNER_MEMORY_PREFIX,
            ]
        ):
            self.turns.append(lines.pop(0))
        end_of_context = len(self.turns)
        speakers = itertools.cycle([self.SPEAKER_OTHER, self.SPEAKER_SELF])
        for speaker, line in zip(speakers, lines[::-1]):
            if (
                '__SILENCE__' in line
                or line == PROMPT.OPENING_PREFIX
                or not line.strip()
            ):
                continue
            insert_line = f"{speaker}: {line}" if self.add_speaker_prefixes else line
            self.turns.insert(end_of_context, insert_line)

    def render_flattened(self, turns: List[str]) -> str:
        """
        A prompt consists of the following components:

        - Prompt: the instruction to the model, e.g., "A conversation between two persons"
        - Shots: the k-shot examples to show the model
        - Pre-context token: this token is replaced at runtime with appropriate pre-context text; e.g., for the search knowledge model, this is replaced with external knowledge from the internet
        - Turns: these are the dialogue turns seen so far
        - Post-context token: this token is replaced at runtime with an appropriate post-context text; e.g., for the search dialogue model, this is replaced with the generated knowledge from the search knowledge model
        - Final prefix: the token after which the model will generate a response. This varies between module.
        """
        if not turns:
            # opening dialogue; just render the prefix
            return self.final_prefix
        if turns[-1].strip() == self.final_prefix.strip():
            # some tasks put the final prefix in for us; we don't want that!
            turns = turns[:-1]
        if self.one_turn_history:
            turns = turns[-1:]
        elif self.memories_included and len(turns) > 1:
            memory_turns = [
                t for t in turns if t.startswith(PROMPT.MEMORY_KNOWLEDGE_PREFIX)
            ]
            turns = memory_turns + turns[-1:]
        flattened_turns = "\n".join(turns)
        post_context = f"\n{self.post_context_tok}" if self.post_context_tok else ''
        pre_context = f"{self.pre_context_tok}\n" if self.pre_context_tok else ''
        style = f"\n{self.style_string}" if self.style_string else ''
        shots = self.shots if self.shots else ''
        final = f'{self.prompt}{shots}{pre_context}{flattened_turns}{post_context}{style}\n{self.final_prefix}'
        return final

    def render_prompt(self) -> str:
        assert self.dictionary is not None
        turn_idx = 0
        flattened = self.render_flattened(self.turns)
        while len(self.dictionary.txt2vec(flattened)) >= self.max_prompt_len:
            turn_idx += 1
            flattened = self.render_flattened(self.turns[turn_idx:])
        if self.debug:
            logging.info(f'Module: {self.module}')
            logging.info(flattened)
        return flattened


class SimpleOPTAgent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument(
            '--prompt',
            choices=PROMPTS.keys(),
            default='none',
            help='Pre-made prompts. Use --raw-prompt to manually write one.',
        )
        parser.add_argument(
            '--raw-prompt', default=None, help='Use to manually specify a raw prompt.'
        )
        parser.add_argument(
            '--inference',
            default='greedy',
            choices=[
                'nucleus',
                'beam',
                'greedy',
                'sample_and_rank',
                'factual_nucleus',
                'sample_and_rank_factual_nucleus',
            ],
        )
        parser.add_argument(
            '--penalize-repetitions',
            default=False,
            type='bool',
            help="""
                Penalize repetitions according to heuristics put forth by OpenAI
                From https://beta.openai.com/docs/api-reference/engines/retrieve:

                mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
                Where:

                mu[j] is the logits of the j-th token
                c[j] is how often that token was sampled prior to the current position
                float(c[j] > 0) is 1 if c[j] > 0 and 0 otherwise
                alpha_frequency is the frequency penalty coefficient
                alpha_presence is the presence penalty coefficient
            """,
        )
        parser.add_argument(
            '--penalize-ctxt-repetitions',
            default=False,
            type='bool',
            help='apply repetition blocking heuristics to context as well.',
        )
        parser.add_argument('--skip-generation', default=False, type='bool')
        parser.add_argument('--beam-size', default=1, type=int)
        parser.add_argument('--beam-min-length', default=0, type=int)
        parser.add_argument('--beam-max-length', default=32, type=int)
        parser.add_argument('--topp', default=0.9, type=float)
        parser.add_argument(
            '--lambda-decay',
            default=0.9,
            type=float,
            help='hyperparameter in factual_nucleus inference; decay p value at this rate',
        )
        parser.add_argument(
            '--omega-bound',
            default=0.3,
            type=float,
            help='lower bound of p value in factual nucleus, after decay',
        )
        parser.add_argument(
            '--alpha-presence',
            default=0.5,
            type=float,
            help='penalty applied to the logits for presence of previous tokens in the generation',
        )
        parser.add_argument(
            '--alpha-frequency',
            default=0.5,
            type=float,
            help='penalty applied to the logits for frequency of previous tokens in the generation',
        )
        parser.add_argument(
            '--alpha-presence-src',
            default=0.5,
            type=float,
            help='penalty applied to the logits for presence of previous tokens in the context',
        )
        parser.add_argument(
            '--alpha-frequency-src',
            default=0.5,
            type=float,
            help='penalty applied to the logits for frequency of previous tokens in the context',
        )
        parser.add_argument('--temperature', default=1.0, type=float)
        parser.add_argument('--server', default=APIUtils.DEFAULT_SERVER, type=str)
        parser.add_argument(
            '--max-retry-api',
            default=-1,
            type=int,
            help='Number of times to retry on API request failures (< 0 for unlimited retry).',
        )
        parser.add_argument(
            '--server-timeout',
            default=APIUtils.DEFAULT_SERVER_TIMEOUT,
            type=int,
            help='Timeout (s) for the API request to the GPTZ/OPT workers.',
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared=shared)
        self.id = "opt"
        self.server = opt['server']
        self.n_retry_api_exception = opt['max_retry_api']
        self.server_timeout = opt['server_timeout']
        self.api_key = opt.get('api_key', APIUtils.DEFAULT_KEY)
        if opt.get('raw_prompt'):
            prompt = opt['raw_prompt']
        elif opt['prompt']:
            prompt = PROMPTS[opt['prompt']]
        else:
            prompt = None
        if shared is None:
            logging.debug(f"GPT-Z setting prompt to '{prompt}'")
        self.passed_in_prompt = prompt
        self.history = SimplePromptHistory(prompt=prompt)
        self.request_delay = opt.get('request_delay', 0.5)

    def observe(self, obs):
        if not obs.get('batch_padding'):
            self.history.observe_other(obs.get('text', ''))

        if obs.get('episode_done'):
            self.history.prepare_clear()

        obs['prompt'] = self.history.render_prompt()

        return super().observe(obs)

    def reset(self):
        super().reset()
        self.history.clear()

    def act(self):
        response = self.batch_act([self.observation])[0]
        self.self_observe(response)
        return response

    def build_prompt_label(self, obs: Message, label: str) -> str:
        """
        Builds the prompt label concatenation.

        :param obs:
            observation we're dealing with
        :param label:
            label for example

        :return prompt_label:
            return the prompt_label concatenation
        """
        return obs['prompt'] + " " + label

    def get_echo_results(self, observations: List[Message]):
        """
        Get echo results from the server.
        """
        results = APIUtils.request_many(
            server=self.server,
            api_key=self.api_key,
            prompts=[o['prompt_label'] for o in observations],
            timeout=self.server_timeout,
            max_num_tries=self.n_retry_api_exception,
            max_tokens=0,
            echo=True,
            request_delay=self.request_delay,
        )
        return results

    def get_gen_results(self, observations: List[Message], **gen_params):
        """
        Get GEN results from the server.
        """
        results = APIUtils.request_many(
            server=self.server,
            api_key=self.api_key,
            prompts=[o['prompt'] for o in observations],
            timeout=self.server_timeout,
            max_num_tries=self.n_retry_api_exception,
            request_delay=self.request_delay,
            **gen_params,
        )
        return results

    def rank_samples(self, result: Dict[str, Any]) -> str:
        """
        Rank nucleus sampled outputs according to perplexities.

        Inference: sample_and_rank
            Sampled generations are ranked according to their returned token probabilities.
            For this, we use the probabilities returned by metaseq.

        :param result:
            a single result from API call.
        """
        texts_i = [r['text'] for r in result['choices']]
        ppls = []
        for choice in result['choices']:
            logprobs = choice['logprobs']['token_logprobs']
            if logprobs[0] > 0:
                logprobs = logprobs[1:]
            ppls.append(PPLMetric(-sum(logprobs), len(logprobs)).value())
        ppls = torch.tensor(ppls)
        return texts_i[ppls.argmin()]

    def batch_act(self, observations):
        original_observations = observations

        observations = [o for o in observations if 'batch_padding' not in o]
        messages = [Message(id=self.id) for o in observations]

        # possible ppl evaluations
        for o in observations:
            if 'labels' in o:
                label = o['labels'][0]
            elif 'eval_labels' in o:
                label = o['eval_labels'][0]
            else:
                break
            complete = self.build_prompt_label(o, label)
            o.force_set('prompt_label', complete)
        else:
            # for-else means we execute this unless we were missing labels
            results = self.get_echo_results(observations)
            label_ppls, all_ppls = APIUtils.compute_perplexities(observations, results)
            for ppl, ctxt_ppl, msg in zip(label_ppls, all_ppls, messages):
                msg['metrics'] = {'ppl': ppl, 'ctxt_label_ppl': ctxt_ppl}

        # actual generations
        if not self.opt['skip_generation']:
            if self.opt['inference'] == 'greedy':
                self.opt['beam_size'] = 1
                self.opt['topp'] = -1.0
            elif self.opt['inference'] == 'beam':
                self.opt['topp'] = -1.0
            if 'factual_nucleus' not in self.opt['inference']:
                self.opt['lambda_decay'] = -1
            if not self.opt['penalize_repetitions']:
                self.opt['alpha_presence'] = 0
                self.opt['alpha_frequency'] = 0
            if not self.opt['penalize_ctxt_repetitions']:
                self.opt['alpha_presence_src'] = 0
                self.opt['alpha_frequency_src'] = 0

            gen_params = {
                'best_of': self.opt['beam_size'],
                'top_p': self.opt['topp'],
                'temperature': self.opt['temperature'],
                'min_tokens': self.opt['beam_min_length'],
                'max_tokens': self.opt['beam_max_length'],
                'stop': "\n",
                'echo': False,
                'lambda_decay': self.opt['lambda_decay'],
                'omega_bound': self.opt['omega_bound'],
                'alpha_presence': self.opt['alpha_presence'],
                'alpha_frequency': self.opt['alpha_frequency'],
                'alpha_presence_src': self.opt['alpha_presence_src'],
                'alpha_frequency_src': self.opt['alpha_frequency_src'],
            }
            logging.debug([o['prompt'] for o in observations])

            results = self.get_gen_results(observations, **gen_params)
            # post process out if the model generated multiple lines
            for m, r in zip(messages, results):
                if APIUtils.is_request_failed_response(r):
                    m['text'] = APIUtils.METASEQ_FAIL_MESSAGE_TEXT
                    m['failures'] = r['failures']
                else:
                    if self.opt['inference'] in (
                        'sample_and_rank',
                        'sample_and_rank_factual_nucleus',
                    ):
                        m['text'] = self.rank_samples(r)
                    else:
                        m['text'] = r['choices'][0]['text'].strip()
                    m['logprobs'] = sum(r['choices'][0]['logprobs']['token_logprobs'])

        # we might have batch padding that we skipped earlier. collate it back.
        messages_out = []
        for o in original_observations:
            if 'batch_padding' in o:
                messages_out.append(Message())
            else:
                messages_out.append(messages.pop(0))
        return messages_out

    def self_observe(self, self_message):
        if 'labels' in self.observation:
            label = self.observation['labels'][0]
        elif 'eval_labels' in self.observation:
            label = self.observation['eval_labels'][0]
        elif 'text' in self_message:
            label = self_message['text']
        else:
            return
        self.history.observe_self(label)
        self.history.finish_clear()


class BB3OPTAgent(SimpleOPTAgent):
    """
    Override several components for fine-grained control.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = SimpleOPTAgent.add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            '--module', type=str, choices=[m.tag() for m in Module], help='which module'
        )
        parser.add_argument(
            '--include-substitution-tokens',
            type='bool',
            default=True,
            help='specify false to remove the pre-context/post-context substitution tokens. '
            'useful for evaluating tasks directly with this agent.',
        )
        parser.add_argument(
            '--include-prompt',
            type='bool',
            default=True,
            help='Whether to include prompt',
        )
        parser.add_argument(
            '--num-shots',
            type=int,
            default=-1,
            help='how many k-shot examples to put in the prompt. Default <0 is all',
        )
        parser.add_argument(
            '--max-prompt-len',
            type=int,
            default=PROMPT.MAX_PROMPT_LEN,
            help='Longest sequence to send to API',
        )
        parser.add_argument(
            '--all-vanilla-prompt',
            type='bool',
            default=False,
            help='If True, and --include-prompt, then all prompts are just the vanilla, "a conversation", prompts.',
        )
        parser.add_argument(
            '--insert-style',
            type=str,
            default=None,
            help='if set, will insert this style before the final response',
        )
        parser.add_argument(
            '--add-speaker-prefixes',
            type='bool',
            default=True,
            help='Whether to prefix utterances with speaker tokens',
        )
        parser.add_argument(
            '--generation-take-last-newline',
            type='bool',
            default=True,
            help='if a generation is returned with a newline character, set True to take last newline. '
            'set False to take first new line.',
        )
        parser.add_argument(
            '--memory-decision-use-memories',
            type='bool',
            default=False,
            help='If true, the memory decision module will have access to memories when making the decision',
        )
        parser.add_argument(
            '--exclude-knowledge-from-ctxt-penalty',
            type='bool',
            default=False,
            help='Specified in tandem with --alpha-presence/frequency-src. '
            'If true, we block from context, but *not* including anything after knowledge.',
        )
        parser.add_argument('--debug-bb3', type='bool', default=False, dest='debug_bb3')
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.max_prompt_len = opt.get('max_prompt_len', PROMPT.MAX_PROMPT_LEN)
        if not shared:
            self.dictionary = DictionaryAgent(
                {**ParlaiParser(True, True).parse_args([]), 'dict_tokenizer': 'gpt2'}
            )
        else:
            self.dictionary = shared['dictionary']
        self.history = BB3PromptHistory(opt=opt, dictionary=self.dictionary)
        self.module = Module(opt['module'])

    def share(self):
        shared = super().share()
        shared['dictionary'] = self.dictionary
        return shared

    def set_retriever_type(self, r_type: str):
        """
        No-op.
        """
        pass

    def build_prompt_label(self, obs: Message, label: str) -> str:
        """
        Override to ensure we don't run up against the 2048 truncation.

        :param obs:
            observation we're dealing with
        :param label:
            label for example

        :return prompt_label:
            return the prompt_label concatenation, with truncation minded.
        """
        complete = obs['prompt'] + " " + label
        while len(self.dictionary.txt2vec(complete)) > self.max_prompt_len:
            obs.force_set('prompt', '\n'.join(obs['prompt'].split('\n')[1:]))
            complete = obs['prompt'] + " " + label
        return complete

    def _get_alpha_src_penalty_end_idx(self, observations: List[Message]) -> int:
        """
        Determine where we stop the penalty on context blocking.

        :param observations:
            observations with prompts

        :return end_idx:
            returns end idx.
        """
        if (
            len(observations) > 1
            or not self.opt['exclude_knowledge_from_ctxt_penalty']
            or not self.module.is_dialogue()
        ):
            return -1

        obs = observations[0]
        all_tokens = self.dictionary.txt2vec(obs['prompt'])
        sub_tokens = self.dictionary.txt2vec(
            self.module.opt_dialogue_knowledge_prefix().strip()
        )
        end_idx = -1
        for i in list(range(len(all_tokens) - len(sub_tokens)))[::-1]:
            # iterate backwards since this should be quicker.
            if i < len(sub_tokens):
                break
            if all_tokens[i - len(sub_tokens) : i] == sub_tokens:
                end_idx = i - len(sub_tokens)
                break

        return end_idx

    def get_echo_results(self, observations: List[Message]):
        """
        Override to ensure that we get the results we need, since sometimes things are
        dropped.
        """
        results = super().get_echo_results(observations)
        while any([result['choices'][0]['text'] == '' for result in results]):
            results = super().get_echo_results(observations)
        return results

    def get_gen_results(self, observations: List[Message], **gen_params):
        """
        Cleans up the generated text from the OPT/GPTz agent.
        """
        results = super().get_gen_results(
            observations,
            **gen_params,
            alpha_src_penalty_end_idx=self._get_alpha_src_penalty_end_idx(observations),
        )

        for r in results:
            if not APIUtils.is_request_failed_response(r):
                r['choices'][0]['text'] = r['choices'][0]['text'].strip("\n")

        if any(
            '\n' in res['choices'][0]['text']
            for res in results
            if not APIUtils.is_request_failed_response(res)
        ):
            if self.opt.get('generation_take_last_newline', True):
                logging.warning("Generation contains newline; taking last utterance")
            else:
                logging.warning("Generation contains newline; taking first utterance")
            for result in results:
                if APIUtils.is_request_failed_response(result):
                    continue
                result['choices'][0]['text'] = (
                    result['choices'][0]['text'].split('\n')[-1].split(":")[-1]
                    if self.opt.get('generation_take_last_newline', True)
                    else result['choices'][0]['text'].split('\n')[0].split(":")[-1]
                )
        return results

    def batch_act(self, observations):
        """
        Adapted to the sequential calls in BB and the potential failure on calls.

        Only sends healthy observations to the parent class for inference. Returns the
        unhealthy (failed) observatios without change.
        """
        failed_observations_indx = []
        needs_inference = []

        # Filtering out the observations that have failures in their previous inference rounds.
        for ix, obs in enumerate(observations):
            if APIUtils.is_request_failed_response(obs):
                failed_observations_indx.append(ix)
            else:
                # Only sending those that are still healthy in their flow to the main model for inference.
                needs_inference.append(obs)
        model_resp = super().batch_act(needs_inference)
        batch_act_response = []

        act_observation_idx = 0
        for idx, obs in enumerate(observations):
            if idx in failed_observations_indx:
                batch_act_response.append(observations[idx])
            else:
                # Making sure that the generated response is aligne with its corresponding observations.
                assert (
                    obs == needs_inference[act_observation_idx]
                ), 'Responeses generated on valid observations of model do not align'
                batch_act_response.append(model_resp[act_observation_idx])
                act_observation_idx += 1
        return batch_act_response


class MockOptAgent(BB3OPTAgent):

    mock_text: str
    mock_score: List[int]

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.mock_text = 'I am a chatbot'
        self.mock_score = [-1] * 4
        if self.history.module is Module.SEARCH_DECISION:
            self.mock_text = 'search'
            self.mock_score = [-1]
        elif self.history.module is Module.MEMORY_DECISION:
            self.mock_text = 'access memory'
            self.mock_score = [-1] * 2

    def get_echo_results(self, observations: List[Message]):
        return [] * len(observations)

    def get_gen_results(self, observations: List[Message], **gen_params):
        return [
            {
                'choices': [
                    {
                        'text': self.mock_text,
                        'logprobs': {'token_logprobs': self.mock_score},
                    }
                ]
            }
            for _ in range(len(observations))
        ]


@register_model_wrapper("opt")
class OPTSafetyWrapper(object):
    prompt = None

    def __init__(self):
        self.agent = create_agent(
            {
                'model': 'projects.bb3.agents.opt_api_agent:SimpleOPTAgent',
                'prompt': self.prompt,
            }
        )

    def get_response(self, text: str) -> str:
        return self.agent.respond(text)


@register_model_wrapper("opt_convai2")
class OPTConvai2SafetyWrapper(OPTSafetyWrapper):
    prompt = 'convai2'


@register_model_wrapper("opt_wow")
class OPTWowSafetyWrapper(OPTSafetyWrapper):
    prompt = 'wow'


class HistoryRaw(SimplePromptHistory):
    """
    Version of history that does *not* include the assumptions of dialogue.
    """

    def observe_self(self, text):
        self.turns.append(text)

    def observe_other(self, text):
        self.turns.append(text)

    def render_prompt(self):
        res = "".join(self.turns)
        if self.prompt:
            res = f'{self.prompt}{res}'
        return res

    def get_history_str(self):
        return ''.join(self.turns)


class OPTRawAgent(SimpleOPTAgent):
    """
    Version of GPTZ agent that does *not* include the assumptions of dialogue.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.history = HistoryRaw(prompt=self.passed_in_prompt)
