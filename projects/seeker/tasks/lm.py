#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Knowledge grounding task built from Language Model Data.
"""
from abc import ABC, abstractmethod
import os
from typing import Tuple, List, Optional, Any, Dict
import random

from parlai.core.message import Message
from parlai.core.metrics import (
    F1Metric,
    BleuMetric,
    RougeMetric,
    AverageMetric,
    ExactMatchMetric,
    normalize_answer,
)
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import ChunkTeacher, ChunkOutput
import parlai.tasks.wizard_of_internet.constants as CONST
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
import parlai.utils.logging as logging
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE

from projects.seeker.utils import extract_entities, remove_possible_title_from_text


class CopiedSubstringMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str]):
        if guess is None or answers is None:
            return None
        guess = normalize_answer(guess)
        for a in answers:
            if guess in normalize_answer(a):
                return CopiedSubstringMetric(1)
        return CopiedSubstringMetric(0)


NO_KNOWLEDGE_FIELD = 'no_knowledge_text'


def _path(opt: Opt, fname: str):
    root = opt['root_dir']
    return os.path.join(root, fname)


class AbstractLMChunkTeacher(ChunkTeacher, ABC):
    """
    LM Chunk Teacher, base.

    Provides functionality for subclassed teachers for extracting data from world log
    files, generated via `generate_lm_data.py`
    """

    def __init__(self, opt: Opt, shared=None) -> None:
        """
        Separate validation data from training data.
        """
        root = opt['root_dir']
        valid_indices = []
        for s in opt['validation_data_indices'].split(','):
            if s:
                valid_indices.append(int(s))
        if not valid_indices:
            logging.warning(
                "Warning: No validation indices provided, all data will be used for training"
            )
        self.data_files = sorted(
            [f for f in os.listdir(root) if f.endswith('.jsonl')],
            key=lambda x: int(x.replace('.jsonl', '').split('_')[-1]),
        )
        if DatatypeHelper.fold(opt['datatype']) in ['valid', 'test']:
            self.data_files = [
                d for i, d in enumerate(self.data_files) if i in valid_indices
            ]
        else:
            self.data_files = [
                d for i, d in enumerate(self.data_files) if i not in valid_indices
            ]
        opt['datafile'] = root
        super().__init__(opt, shared)  # type: ignore

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        ChunkTeacher.add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('LM Task Arguments')
        agent.add_argument(
            '--root-dir',
            type=str,
            default=None,
            help="Root directory with generated world logs.",
        )
        agent.add_argument(
            '--validation-data-indices',
            type=str,
            default='',
            help="comma separated list of validation data indices",
        )
        agent.add_argument(
            '--shared-knowledge-entity',
            type='bool',
            help='If True, only include examples with an entity in the knowledge that is also in the label',
            default=False,
        )
        agent.add_argument(
            '--min-knowledge-length',
            type=int,
            help='minimum length of the knowledge required to include the example. Default -1 means no min length',
            default=5,
        )
        agent.add_argument(
            '--min-knowledge-overlap',
            type=float,
            help='minimum overlap between knowledge and target sentence. Default 0 means no overlap required.',
            default=0,
        )
        agent.add_argument(
            '--skip-empty-context',
            type=bool,
            help='whether to exclude examples with empty context.',
            default=False,
        )
        agent.add_argument(
            '--exclude-retrieved-docs',
            type='bool',
            default=False,
            help='specify to not include retrieved docs in the episodes. hopefully reduces memory footprint.',
        )
        return parser

    def _get_ep_from_turns(
        self, xturns: List[Message], yturns: List[Message]
    ) -> List[Message]:
        """
        Return an episode given xturns and yturns.

        :param xturns:
            list of context turns
        :param yturns:
            list of label turns

        :return episodes:
            return built episodes
        """
        eps = []
        for xturn, yturn in zip(xturns, yturns):
            turn = {}
            # standard fields
            turn['text'] = (
                xturn.get('text')
                .strip()
                .replace(TOKEN_KNOWLEDGE, f'{TOKEN_KNOWLEDGE} ')
                .replace(TOKEN_END_KNOWLEDGE, f' {TOKEN_END_KNOWLEDGE}')
            )
            # subtract 1 to get rid of newline char
            if len(turn['text'].split('\n')) == 1:
                # no actual context here...
                turn[NO_KNOWLEDGE_FIELD] = ''
            else:
                newline_correction = int(turn['text'].endswith('\n'))
                turn[NO_KNOWLEDGE_FIELD] = turn['text'][
                    : turn['text'].index(TOKEN_KNOWLEDGE) - newline_correction
                ]
            turn['labels'] = [yturn.get('text').strip()]
            turn['episode_done'] = False
            # my fields
            if not self.opt['exclude_retrieved_docs']:
                turn[CONST.RETRIEVED_DOCS] = yturn['retrieved_docs']
                turn[CONST.RETRIEVED_DOCS_TITLES] = [
                    d.split(' / ')[0] for d in yturn['retrieved_docs']
                ]
                turn[CONST.RETRIEVED_DOCS_URLS] = ['' for _ in yturn['retrieved_docs']]
                turn[CONST.SELECTED_DOCS] = [yturn['gold_doc']]
                turn[CONST.SELECTED_DOCS_TITLES] = [yturn['gold_doc'].split(' / ')[0]]
                turn[CONST.SELECTED_DOCS_URLS] = []
                turn[CONST.SELECTED_SENTENCES] = [yturn['knowledge']]
            else:
                turn[CONST.SELECTED_DOCS_TITLES] = [yturn['gold_doc'].split(' / ')[0]]
                turn[CONST.SELECTED_SENTENCES] = [yturn['knowledge']]

            turn['search_query'] = yturn['search_query']
            turn['f1_overlap'] = yturn['f1_overlap']
            if self.passes_filters(xturn, yturn):
                eps.append(self.finalize_message(turn))
        if eps:
            eps[-1].force_set('episode_done', True)
        return eps

    def passes_filters(self, xturn: Dict[str, Any], yturn: Dict[str, Any]) -> bool:
        """
        Subject example to various filters.

        Return whether the example passes all filters.

        :param xturn:
            context turn
        :param yturn:
            target/knowledge turn

        :return passes_filters:
            return whether the example passes the filters.
        """
        passes = True
        # Example filters
        knowledge = (
            yturn['knowledge']
            .replace(TOKEN_KNOWLEDGE, '')
            .replace(TOKEN_END_KNOWLEDGE, '')
            .strip()
        )
        if passes and self.opt['skip_empty_context']:
            doc_context_sentences = [
                s for s in xturn['text'].split(TOKEN_KNOWLEDGE)[0].split('\n') if s
            ]
            passes &= (
                len(doc_context_sentences)
            ) > 1  # All docs have <doc> token as their first line
        if passes and self.opt['min_knowledge_length'] > 0:
            passes &= len(knowledge.split(' ')) >= self.opt['min_knowledge_length']
        if passes and self.opt['min_knowledge_overlap'] > 0:
            assert 0 < self.opt['min_knowledge_overlap'] <= 1
            f1 = F1Metric.compute(yturn['text'].strip(), [knowledge])
            passes &= f1.value() >= self.opt['min_knowledge_overlap']
        if passes and self.opt['shared_knowledge_entity']:
            knol_ent = extract_entities(knowledge)
            if len(knol_ent) == 0:
                passes &= False
            label_ent = extract_entities(yturn.get('text'))
            ents = set(knol_ent).intersection(label_ent)
            if len(ents) == 0:
                passes &= False
        return passes

    @abstractmethod
    def finalize_message(self, msg: Dict[str, Any]) -> Message:
        """
        Finalize message.

        Override in subclasses to set text/labels appropriately.
        """

    def get_num_samples(self, opt: Opt) -> Tuple[int, int]:
        """
        Return the number of samples.

        Returns a tuple of (num_examples, num_episodes) based on the data split.
        """
        return len(self.data_files) * 1000, len(self.data_files) * 1000

    def get_fold_chunks(self, opt: Opt) -> List[int]:
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        return list(range(len(self.data_files)))

    def load_from_chunk(self, chunk_idx: int) -> List[ChunkOutput]:
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        convs = Conversations(_path(self.opt, self.data_files[chunk_idx]))
        chunk = []
        for conv in convs:
            turns = [t for t in conv.turns if t.get('id') != 'context']
            ep = self._get_ep_from_turns(turns[::2], turns[1::2])
            if not ep:
                continue
            chunk += ep
        return chunk

    def create_message(self, queue_output: ChunkOutput, entry_idx=0) -> Message:
        """
        Given the tuple output of the queue, return an act.

        May depend on entry index if queue output is a multi-turn episode.
        """
        return queue_output


class ResponseTeacher(AbstractLMChunkTeacher):
    """
    Response teacher.

    For training the dialogue response component of SeeKeR.
    """

    def finalize_message(self, msg: Dict[str, Any]) -> Message:
        """
        Return message as is.
        """
        return Message(msg)


class ResponseNoKnowledgeTeacher(AbstractLMChunkTeacher):
    """
    A variant that removes the knowledge from the raw data.
    """

    def finalize_message(self, msg: Dict[str, Any]) -> Message:
        """
        Remove the knowledge component of the message.
        """
        msg['text'] = msg[NO_KNOWLEDGE_FIELD]
        return Message(msg)


class KnowledgeTeacher(AbstractLMChunkTeacher):
    """
    Knowledge Teacher.

    Input is Context. Target is selected __knowledge__.
    """

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        """
        Various F1 metrics for the generated model response.
        """
        if not model_response.get('text'):
            # No response generated by model.
            return

        resp = model_response['text']
        # F1 metric over the *selected* knowledge.
        self.metrics.add(
            'knowledge_f1_docs',
            F1Metric.compute(resp, teacher_action[CONST.SELECTED_DOCS]),
        )
        self.metrics.add(
            'knowledge_f1_sentences',
            F1Metric.compute(resp, teacher_action[CONST.SELECTED_SENTENCES]),
        )

        # F1 Metrics over the *retrieved* docs.
        self.metrics.add(
            'f1_retrieved_docs',
            F1Metric.compute(resp, ' '.join(teacher_action[CONST.RETRIEVED_DOCS])),
        )
        self.metrics.add(
            'max_f1_retrieved_docs',
            F1Metric.compute(resp, teacher_action[CONST.RETRIEVED_DOCS]),
        )

        selected_doc_senetences = teacher_action[CONST.SELECTED_DOCS][0].split('\n')
        all_doc_senetences = []
        for doc in teacher_action[CONST.RETRIEVED_DOCS]:
            all_doc_senetences.extend(doc.split('\n'))

        self.metrics.add(
            'exact_copied_sentences', ExactMatchMetric.compute(resp, all_doc_senetences)
        )
        self.metrics.add(
            'max_substring_copied_sentences',
            CopiedSubstringMetric.compute(resp, all_doc_senetences),
        )
        self.metrics.add(
            'max_substring_copied_docs',
            CopiedSubstringMetric.compute(resp, teacher_action[CONST.RETRIEVED_DOCS]),
        )
        self.metrics.add(
            'substring_copied_docs',
            CopiedSubstringMetric.compute(
                resp, [''.join(teacher_action[CONST.RETRIEVED_DOCS])]
            ),
        )
        self.metrics.add(
            'max_f1_selected_docs_senetences',
            F1Metric.compute(resp, selected_doc_senetences),
        )
        self.metrics.add(
            'max_f1_docs_senetences', F1Metric.compute(resp, all_doc_senetences)
        )

        # N-gram matching metrics
        for k in range(1, 5):  # 1..4
            self.metrics.add(
                f'max_bleu_selected_docs_senetences-{k}',
                BleuMetric.compute(resp, selected_doc_senetences, k),
            )

        r1, r2, rL = RougeMetric.compute_many(resp, selected_doc_senetences)
        self.metrics.add('max_rouge_selected_docs_senetences_1', r1)
        self.metrics.add('max_rouge_selected_docs_senetences_2', r2)
        self.metrics.add('max_rouge_selected_docs_senetences_L', rL)

    def finalize_message(self, msg: Dict[str, Any]) -> Message:
        """
        Remove knowledge component of message, and move selected sentence to target.
        """
        msg['text'] = msg[NO_KNOWLEDGE_FIELD]
        msg['labels'] = [
            ' '.join(msg[CONST.SELECTED_SENTENCES])
            .replace(TOKEN_KNOWLEDGE, '')
            .replace(TOKEN_END_KNOWLEDGE, '')
            .strip()
        ]
        return Message(msg)


class SearchQueryTeacher(KnowledgeTeacher):
    """
    Search Query teacher.

    Given an input context, predict the title of the document from which the "gold"
    knowledge is derived.
    """

    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        KnowledgeTeacher.add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group("Selected Doc Titles Arguments")
        agent.add_argument(
            "--remove-possible-title",
            type=bool,
            default=True,
            help="Remove the length of characters from the beginning of the text that exactly match the title.",
        )
        agent.add_argument(
            "--random-trim-min-length",
            type=int,
            default=-1,
            help=(
                "If text is longer than this length (in chars), randomly trim from the beginning "
                "such that the text is at least this long (default -1, no trimming)."
            ),
        )
        return parser

    def __init__(self, opt, shared=None):
        self._rem_possible_title = opt['remove_possible_title']
        self._random_trim_min_len = opt['random_trim_min_length']
        super().__init__(opt, shared=shared)

    def _clean_text(self, txt, label):
        PRFX = '<doc>'
        if txt.startswith(PRFX):
            # removing the extra white-space character after the PRFX too.
            txt = txt[len(PRFX) + 1 :]

        if self._rem_possible_title:
            txt = remove_possible_title_from_text(txt, label)

        if self._random_trim_min_len > 0 and len(txt) > self._random_trim_min_len:
            trim_point_idx = random.randint(0, len(txt) - self._random_trim_min_len)
            txt = txt[trim_point_idx:]

        return txt

    def finalize_message(self, msg: Dict[str, Any]) -> Message:
        assert (
            len(msg[CONST.SELECTED_DOCS_TITLES]) == 1
        ), 'Only 1 doc must be selected by this teacher.'

        label = msg[CONST.SELECTED_DOCS_TITLES][0]
        msg['text'] = self._clean_text(msg[NO_KNOWLEDGE_FIELD], label)
        msg['labels'] = [label]
        return Message(msg)


class DefaultTeacher(ResponseTeacher):
    pass
