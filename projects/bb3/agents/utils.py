#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import aiohttp
import asyncio
from enum import Enum
import json
import math
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import string
import time
from typing import List, Tuple, Optional, Dict, Any

from parlai.agents.ir_baseline.ir_baseline import score_match, MaxPriorityQueue
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.torch_generator_agent import PPLMetric
import parlai.utils.logging as logging
import projects.bb3.constants as CONST
import projects.bb3.prompts as PROMPT
from projects.bb3.agents.module import Module

try:
    import contractions
except ImportError as e:
    print('please install contractions: "pip install contractions"')
    raise e

STEMMER = PorterStemmer()

# Very limited number of stopwords:
# We don't remove you and me etc. because of their strong importance in memory records.
_STOP_WORDS = set(
    list(string.punctuation)
    + ['a', 'an', 'the', 'am', 'are', 'is', 'as']
    + [
        STEMMER.stem(adv)
        for adv in (
            'really',
            'actually',
            'up',
            'so',
            'just',
            'now',
            'how',
            'then',
            'also',
            'very',
            'too',
            'most',
            'but',
        )
    ]
)


def normal_tokenizer(
    text: str, remove_contractions: bool = True, stem=True
) -> List[str]:
    """
    Returns normalized tokens for `text`
    """
    assert isinstance(text, str), 'The only valid arg type is str.'
    text = text.strip().lower()
    if not text:
        return []

    if remove_contractions:
        text = contractions.fix(text)

    tokens = word_tokenize(text)

    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    tokens = [t for t in tokens if t not in _STOP_WORDS]

    return tokens


def no_overlap(
    data_source: List[str],
    canidate_text: str,
    non_overlap_threshold: int = 1,
    remove_contractions: bool = True,
) -> bool:
    """
    Returns True if tokenized `canidate_text` text has at least `non_overlap_threshold`
    tokens NOT overlapping with all texts in `data_source` list.

    We use this for de-duplicating (for example memory entries)
    """
    tokenized_candidate_text = set(
        normal_tokenizer(
            canidate_text,
            remove_contractions=remove_contractions,
        )
    )
    for dst in data_source:
        dst_tokens = set(
            normal_tokenizer(
                dst,
                remove_contractions=remove_contractions,
            )
        )
        n_overlap = len(dst_tokens - tokenized_candidate_text)
        logging.debug(
            f'"{dst}" AND "{canidate_text}" ---- non-overlapping tokens: {n_overlap}'
        )
        if n_overlap < non_overlap_threshold:
            logging.info(
                f'"{canidate_text}" has too much overlap with "{dst}": discarding repeated entry.'
            )
            return False

    return True


def clean_text(text: str) -> str:
    """
    Removes all special tokens from an incoming text.
    """
    for token in CONST.ALL_SPECIAL_TOKENS:
        text = text.replace(f" {token}", '')
        text = text.replace(f"{token} ", '')
        text = text.replace(token, '')

    return text


class Decision(Enum):

    ALWAYS = 'always'
    NEVER = 'never'
    COMPUTE = 'compute'


def is_opener(text: str, mems: Optional[List[str]]):
    """
    Return if message is an opener!
    """
    return (
        text == PROMPT.OPENING_PREFIX
        and mems is not None
        and isinstance(mems, list)
        and len(mems) > 0
    )


def set_failed_reply(reply: Message) -> Message:
    """
    Set proper fields in reply if failed.
    """
    reply.force_set('id', 'FallBackRoutine')
    reply.force_set('text', APIUtils.METASEQ_FAIL_MESSAGE_TEXT)
    reply.force_set('score', float('inf'))
    return reply


################
# Memory Utils #
################


class MemoryUtils:
    @staticmethod
    def is_opt_ft_mem_format(memory_text: str) -> bool:
        return (
            ':' in memory_text
            and memory_text.startswith(PROMPT.SELF_MEMORY_PREFIX)
            or memory_text.startswith(PROMPT.PARTNER_MEMORY_PREFIX)
        )

    @staticmethod
    def _is_r2c2_format(memory_text: str) -> bool:
        return (
            ':' in memory_text
            and memory_text.startswith("your persona")
            or memory_text.startswith("partner's persona")
        )

    @staticmethod
    def _is_opt_prompt_format(memory_text: str) -> bool:
        return memory_text.startswith(PROMPT.SELF_PREFIX) or memory_text.startswith(
            PROMPT.PARTNER_PREFIX
        )

    @staticmethod
    def validate_memory_format(memory_text: str):
        assert (
            MemoryUtils._is_r2c2_format(memory_text)
            or MemoryUtils._is_opt_prompt_format(memory_text)
            or MemoryUtils.is_opt_ft_mem_format(memory_text)
        ), f'Provided memory "{memory_text}" has invalid format for chatbot memory field.'

    @staticmethod
    def split_prefix_memory(memory_text: str) -> Tuple[str, str]:
        MemoryUtils.validate_memory_format(memory_text)
        try:
            prfx, mem = memory_text.split(':', 1)
        except ValueError:
            # prompt agent sometimes says things like,
            # "Person 2 is"...
            assert memory_text.startswith(PROMPT.SELF_PREFIX) or memory_text.startswith(
                PROMPT.PARTNER_PREFIX
            )
            if memory_text.startswith(PROMPT.SELF_PREFIX):
                prfx, mem = (
                    memory_text[: len(PROMPT.SELF_PREFIX)],
                    memory_text[len(PROMPT.SELF_PREFIX) + 1 :],
                )
            else:
                prfx, mem = (
                    memory_text[: len(PROMPT.PARTNER_PREFIX)],
                    memory_text[len(PROMPT.PARTNER_PREFIX) + 1 :],
                )
        return prfx.strip(), mem.strip()

    @staticmethod
    def is_valid_memory(
        chatbot_memory: List[str], new_memory: str, new_memory_prefix: str
    ) -> bool:
        """
        Return whether the memory is valid.

        It rejects new memory entry as invalid if one similar to it exists already.
        """
        if new_memory in (
            CONST.NOPERSONA,
            PROMPT.NO_MEMORY,
            '',
            APIUtils.METASEQ_FAIL_MESSAGE_TEXT,
        ):
            return False

        # Rejecting if identical memories exist already.
        if not chatbot_memory:
            # No need to dedup if there is no existing memory
            return True

        # filtering for the memories that applies to the current person (self or other)
        person_memories = [
            MemoryUtils.split_prefix_memory(mem)[1]
            for mem in chatbot_memory
            if MemoryUtils.split_prefix_memory(mem)[0] == new_memory_prefix
        ]
        if not person_memories:
            # No memory on record for this person
            return True

        return no_overlap(person_memories, new_memory)

    @staticmethod
    def get_memory_prefix(self_or_partner: str, model_type: str) -> str:
        """
        Return memory prefix.
        """
        assert self_or_partner in ['self', 'partner']
        assert model_type in ['R2C2', 'OPT']
        self_prefix = (
            'your persona' if model_type == 'R2C2' else PROMPT.SELF_MEMORY_PREFIX
        )
        partner_prefix = (
            "partner's persona"
            if model_type == 'R2C2'
            else PROMPT.PARTNER_MEMORY_PREFIX
        )
        if self_or_partner == 'self':
            return self_prefix
        else:
            return partner_prefix

    @staticmethod
    def add_memory_prefix(memory: str, self_or_partner: str, model_type: str) -> str:
        """
        Ensure that the memory has a "persona" prefix.

        :param memory:
            memory to prefix
        :param self_or_partner:
            string representing if this is a self memory or partner memory
        :param model_type:
            which model we're working with

        :return prefixed_mem:
            return a memory with the appropriate prefix.
        """
        assert self_or_partner in ['self', 'partner']
        assert model_type in ['R2C2', 'OPT']
        prefix = MemoryUtils.get_memory_prefix(self_or_partner, model_type)
        if model_type == 'R2C2':
            if not memory.startswith(prefix):
                memory = f"{prefix}: {memory}"
        elif self_or_partner == 'self':
            memory = memory.replace("Person 1", "Person 2")
            if not memory.startswith('Person'):
                memory = f"{prefix}: {memory}"
        else:
            memory = memory.replace("Person 2", "Person 1")
            if not memory.startswith('Person'):
                memory = f"{prefix}: {memory}"

        return memory

    @staticmethod
    def maybe_add_memory_prefix(
        memory: str, self_or_partner: str, model_type: str
    ) -> str:
        """
        Maybe add it if it's not there.
        """
        if not MemoryUtils.is_opt_ft_mem_format(memory):
            memory = MemoryUtils.add_memory_prefix(memory, self_or_partner, model_type)
        return memory

    @staticmethod
    def _build_query_representation(
        query: str, dictionary: DictionaryAgent
    ) -> Dict[str, Any]:
        rep = {}
        rep['words'] = {}
        words = [w for w in dictionary.tokenize(query.lower())]
        rw = rep['words']
        used = {}
        for w in words:
            rw[w] = 1.0 / (1.0 + math.log(1.0 + dictionary.freq[w]))
            used[w] = True
        rep['norm'] = math.sqrt(len(words))
        return rep

    @staticmethod
    def get_available_memory(
        observation: Message, dictionary: DictionaryAgent
    ) -> List[str]:
        """
        TFIDF-Match memories with the textual input to reduce num memories.

        :param observation:
            raw observation
        :param memories:
            memories from which to choose

        :return memories:
            return - potentially shortened - list of memories
        """
        new_memories = []
        mems = observation['memories']
        if not mems or len(mems) < 32:  # 512 / 16, assuming 16 tokens max per memory
            return mems
        mpq = MaxPriorityQueue(1000)
        query = MemoryUtils._build_query_representation(observation['text'], dictionary)
        for m in mems:
            score = score_match(query, m, 0, dictionary)
            mpq.add(m, score)
        new_memories = list(reversed(mpq))[:32]
        return new_memories


#################
# OPT API UTILS #
#################


class APIUtils:

    DEFAULT_KEY = os.environ.get("USER", "parlai")
    DEFAULT_SERVER = "DEFAULT_API_SERVER:6010"
    DEFAULT_SERVER_TIMEOUT = 600
    METASEQ_FAIL_MESSAGE_TEXT = 'METASEQ RESPONSE FAILED.'

    @staticmethod
    def is_request_failed_response(resp):
        """
        Whether the requests to Metaseq worker have failed.

        It checks this based on the existences of the failure reasons as they get
        accumulated in `_make_request` functionn calls.
        """
        return len(
            resp.get('failures', [])
        ) > 0 or APIUtils.METASEQ_FAIL_MESSAGE_TEXT in resp.get('text', '')

    @staticmethod
    async def make_request(
        session,
        server: str,
        api_key: str,
        prompt: str,
        min_tokens: int = 0,
        n: int = 1,
        max_tokens: int = 32,
        best_of: int = 1,
        top_p: float = -1,
        echo: bool = False,
        stop: Optional[str] = None,
        temperature: float = 1.0,
        num_retry_on_api_exception=-1,
        lambda_decay: float = -1,
        omega_bound: float = 0.3,
        request_delay: float = 0.5,
        alpha_presence: float = 0.0,
        alpha_frequency: float = 0.0,
        alpha_presence_src: float = 0.0,
        alpha_frequency_src: float = 0.0,
        alpha_src_penalty_end_idx: int = -1,
    ):
        data = {
            'prompt': prompt,
            'min_tokens': min_tokens,
            'max_tokens': max_tokens,
            'best_of': best_of,
            'top_p': top_p,
            'stop': stop,
            'temperature': temperature,
            'echo': echo,
            'lambda_decay': lambda_decay,
            'omega_bound': omega_bound,
            'alpha_presence': alpha_presence,
            'alpha_frequency': alpha_frequency,
            'alpha_presence_src': alpha_presence_src,
            'alpha_frequency_src': alpha_frequency_src,
            "alpha_src_penalty_end_idx": alpha_src_penalty_end_idx,
        }
        init_request_delay = request_delay
        past_exceptions = []
        while True:
            if (
                num_retry_on_api_exception >= 0
                and len(past_exceptions) > num_retry_on_api_exception
            ):
                logging.error('Reached maximum retries, returning failure message.')
                return {
                    'failures': past_exceptions,
                }
            try:
                logging.debug(f'Making request: {data}')
                headers = {'Authorization': f'Bearer {api_key}'}
                async with session.post(
                    f'{server}/completions', json=data, headers=headers
                ) as resp:
                    resp_text = await resp.text()
                    obj = json.loads(resp_text)
                    if 'error' in obj:
                        request_delay *= 2
                        logging.warning(f"Error: {obj['error']}")
                        past_exceptions.append(f"API Error: {obj['error']}")
                        logging.debug(past_exceptions[-1])
                        continue
                    debug = json.dumps(obj, sort_keys=True)
                    logging.debug(f'GPT-Z response: {debug}')
                    request_delay = init_request_delay
                    return obj
            except asyncio.TimeoutError as e:
                past_exceptions.append(
                    f'Timout a response for prompt {len(prompt)}\n{e}'
                )
                logging.warning(past_exceptions[-1])
                request_delay *= 2
            except aiohttp.client_exceptions.ClientOSError as e:
                past_exceptions.append(
                    f'Retrying a response for prompt {len(prompt)}\n{e}'
                )
                logging.warning(past_exceptions[-1])
                request_delay *= 2
            except json.decoder.JSONDecodeError as e:
                past_exceptions.append(
                    f'Got a bad response, {resp_text}. Retrying.\n{e}'
                )
                logging.debug(past_exceptions[-1])
                request_delay *= 2

            time.sleep(request_delay)

    @staticmethod
    async def async_request_many(
        server: str,
        api_key: str,
        prompts: List[str],
        timeout: Optional[int] = None,
        max_num_tries: int = -1,
        **kwargs,
    ):
        connector = aiohttp.TCPConnector(limit=0)
        timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            tasks = []
            for prompt in prompts:
                tasks.append(
                    asyncio.ensure_future(
                        APIUtils.make_request(
                            session=session,
                            server=server,
                            api_key=api_key,
                            prompt=prompt,
                            num_retry_on_api_exception=max_num_tries,
                            **kwargs,
                        )
                    )
                )
            results = await asyncio.gather(*tasks)
            return results

    @staticmethod
    def request_many(
        server,
        api_key,
        prompts: List[str],
        timeout: Optional[int] = None,
        max_num_tries: int = -1,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        return asyncio.run(APIUtils.async_request_many(server=server, api_key=api_key, prompts=prompts, timeout=timeout, max_num_tries=max_num_tries, **kwargs))  # type: ignore

    @staticmethod
    def compute_perplexities(
        observations: List[Message], results: List[Dict[str, Any]]
    ) -> Tuple[List[PPLMetric], List[PPLMetric]]:
        """
        Compute perplexities from API call.

        :param observations:
            incoming observations
        :param results:
            results from API call

        :return ppls:
            return list of perplexities
        """
        label_perplexities = []
        all_perplexities = []
        for obs, result in zip(observations, results):
            # need text offsets to figure out what comes from the prompt
            prompt_len = len(obs['prompt'])

            text_off = result['choices'][0]['logprobs']['text_offset']
            start_label = [i for i, off in enumerate(text_off) if off <= prompt_len]
            assert len(start_label) > 0
            start_label = start_label[-1]
            all_log_probs = result['choices'][0]['logprobs']['token_logprobs']
            if not all(l <= 0 for l in all_log_probs):
                logging.warning(
                    f'Out of {len(all_log_probs)} log probs, {len([l for l in all_log_probs if l > 0])} are > 0. '
                    'Clamping to 0'
                )
                all_log_probs = [min(l, 0) for l in all_log_probs]

            log_probs = all_log_probs[start_label:]
            loss = -sum(log_probs)
            label_perplexities.append(PPLMetric(loss, len(log_probs)))
            all_perplexities.append(PPLMetric(-sum(all_log_probs), len(all_log_probs)))
        return label_perplexities, all_perplexities


class DisplayUtils:
    @staticmethod
    def display_observations(observations: Dict[Module, Dict[str, Any]]):
        """
        Print the observations nicely.
        """
        for module, obs in observations.items():
            if module == 'raw':
                continue
            print(
                f"{module.message_name()}\n{'-' * 80}\n{obs.get('prompt')}\n{'-' * 80}\n{'-' * 80}"
            )

    @staticmethod
    def display_act(act: Message):
        """
        Print the observations nicely.
        """
        print("=" * 80)
        print(f"{'-' * 20} DISPLAYING ACT FIELDS {'-' * 20}")
        print("=" * 80)
        for key, val in act.items():
            if not val:
                continue
            print(f"{key}\n{'-' * 80}\n{val}\n{'-' * 80}\n{'-' * 80}")
