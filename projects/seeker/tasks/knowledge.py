#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
SeeKeR Knowledge Tasks.
"""
import os
import json
from typing import Optional, List

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import MultiTaskTeacher
import parlai.tasks.convai2.agents as convai2
import parlai.tasks.blended_skill_talk.agents as bst
import parlai.tasks.empathetic_dialogues.agents as ed
import parlai.tasks.msc.agents as msc
import parlai.tasks.wizard_of_internet.agents as woi
import parlai.tasks.wizard_of_wikipedia.agents as wow
from parlai.tasks.wizard_of_wikipedia.build import build as wow_build_data
import parlai.tasks.squad.agents as squad
import parlai.tasks.ms_marco.agents as ms_marco
import parlai.tasks.triviaqa.agents as triviaqa
import parlai.tasks.natural_questions.agents as nq

import parlai.utils.logging as logging

import projects.seeker.tasks.mutators  # type: ignore


class WoiKnowledgeTeacher(woi.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(self.get_special_mutators())
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "WoiKnowledgeTeacher"

    def get_special_mutators(self) -> List[str]:
        return [
            'flatten',
            'woi_filter_no_passage_used',
            'woi_checked_sentence_as_label',
            'woi_chunk_retrieved_docs',
            'woi_dropout_retrieved_docs',
            'woi_filter_selected_knowledge_in_retrieved_docs',
        ]


class WowKnowledgeTeacher(wow.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['add_missing_turns'] = 'all'
        mutators = '+'.join(self.get_special_mutators())
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "WowKnowledgeTeacher"

    def get_special_mutators(self) -> List[str]:
        return [
            'flatten',
            'wow_filter_no_passage_used',
            'wow_checked_sentence_as_label',
            'wow_to_woi',
            'woi_chunk_retrieved_docs',
            'woi_dropout_retrieved_docs',
            'woi_filter_selected_knowledge_in_retrieved_docs',
            'add_selected_sentences_mutator',
        ]


class MsMarcoKnowledgeTeacher(ms_marco.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(self.get_special_mutators())
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "MsMarcoKnowledgeTeacher"

    def get_special_mutators(self) -> List[str]:
        return [
            'ms_marco_filter_has_answer',
            'ms_marco_create_fid_docs',
            'ms_marco_find_selected_sentence_for_knowledge',
            'ms_marco_to_woi',
            'woi_chunk_retrieved_docs',
            'add_selected_sentences_mutator',
        ]


class SquadKnowledgeTeacher(squad.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(self.get_special_mutators())
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "SquadKnowledgeTeacher"

    def get_special_mutators(self) -> List[str]:
        return [
            'squad_to_woi',
            'woi_chunk_retrieved_docs',
            'add_selected_sentences_mutator',
        ]


class TriviaQAKnowledgeTeacher(triviaqa.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(self.get_special_mutators())
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "TriviaQAKnowledgeTeacher"

    def get_special_mutators(self) -> List[str]:
        return [
            'triviaqa_to_woi',
            'woi_chunk_retrieved_docs',
            'add_selected_sentences_mutator',
        ]


class NQKnowledgeTeacher(nq.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(self.get_special_mutators())
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "NQKnowledgeTeacher"

    def get_special_mutators(self) -> List[str]:
        return [
            'nq_to_woi',
            'woi_chunk_retrieved_docs',
            'add_selected_sentences_mutator',
        ]


class NQOpenKnowledgeTeacher(nq.NaturalQuestionsOpenTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(self.get_special_mutators())
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "NQOpenKnowledgeTeacher"

    def get_special_mutators(self) -> List[str]:
        return [
            'nqopen_to_woi',
            'woi_chunk_retrieved_docs',
            'add_selected_sentences_mutator',
        ]


class NQOpenDialoguesKnowledgeTeacher(NQOpenKnowledgeTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = "NQOpenDialoguesKnowledgeTeacher"

    def setup_data(self, fold):
        data = super().setup_data(fold)
        # Find matching wow dialogue.
        wow_build_data(self.opt)
        wow_filename = os.path.join(
            self.opt['datapath'], 'wizard_of_wikipedia', 'train.json'
        )
        with open(wow_filename, 'r') as f:
            wow_data = json.load(f)
        wow_topics = set([e['chosen_topic'].lower() for e in wow_data])

        def try_find_matching_topic(q):
            for topic in wow_topics:
                if (
                    f' {topic.lower()} ' in q.lower()
                    or f' {topic.lower()} ' in q.lower()
                ):
                    return topic
            return None

        def context_for_topic(topic):
            context = []
            for ex in wow_data:
                if ex['chosen_topic'].lower() == topic:
                    context = [d['text'] for d in ex['dialog']]
                    break
            # Make sure that the dialogue doesn't end on a question.
            while context and context[-1].strip().endswith('?'):
                context.pop()
            return context

        for ex, done in data:
            question = ex['text']
            topic = try_find_matching_topic(question)
            if not topic:
                continue
            context = context_for_topic(topic)
            if not context:
                continue
            ex['history'] = '\n'.join(context)
            yield ex, done


def get_dialogue_task_mutators(opt: Opt) -> str:
    """
    Set the mutators appropriately for the dialogue tasks.
    """
    mutators = '+'.join(
        ['flatten', 'extract_entity_for_knowledge_model', 'skip_retrieval_mutator']
    )
    if opt.get('mutators'):
        mutators = '+'.join([mutators, opt['mutators']])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2KnowledgeTeacher(convai2.NormalizedTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = self.get_special_mutators(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)
        self.id = 'Convai2KnowledgeTeacher'

    def get_special_mutators(self, opt):
        return get_dialogue_task_mutators(opt)


class EDKnowledgeTeacher(ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = self.get_special_mutators(opt)
        super().__init__(opt, shared)
        self.id = 'EDKnowledgeTeacher'

    def get_special_mutators(self, opt):
        return get_dialogue_task_mutators(opt)


class BSTKnowledgeTeacher(bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = self.get_special_mutators(opt)
        super().__init__(opt, shared)
        self.id = 'BSTKnowledgeTeacher'

    def get_special_mutators(self, opt):
        return get_dialogue_task_mutators(opt)


class MSCKnowledgeTeacher(msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = self.get_special_mutators(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)
        self.id = 'MSCKnowledgeTeacher'

    def get_special_mutators(self, opt):
        return get_dialogue_task_mutators(opt)


class MSCKnowledgeOverlapTeacher(msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten',
                'msc_find_selected_sentence_knowledge',
                'add_retrieved_documents_mutator',
                'skip_retrieval_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['include_session1'] = False
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = 'MSCKnowledgeOverlapTeacher'


class KnowledgeTeacher(MultiTaskTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        WoiKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        WowKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        MsMarcoKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        SquadKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        TriviaQAKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        NQKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        NQOpenKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        NQOpenDialoguesKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        Convai2KnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        EDKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        BSTKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        MSCKnowledgeTeacher.add_cmdline_args(parser, partial_opt)
        MSCKnowledgeOverlapTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        tasks = [
            f"projects.seeker.tasks.knowledge:{teacher}"
            for teacher in [
                'WoiKnowledgeTeacher',
                'WowKnowledgeTeacher',
                'MsMarcoKnowledgeTeacher',
                'SquadKnowledgeTeacher',
                'TriviaQAKnowledgeTeacher',
                'NQKnowledgeTeacher',
                'NQOpenKnowledgeTeacher',
                'NQOpenDialoguesKnowledgeTeacher',
                'Convai2KnowledgeTeacher',
                'EDKnowledgeTeacher',
                'BSTKnowledgeTeacher',
                'MSCKnowledgeTeacher',
                'MSCKnowledgeOverlapTeacher',
            ]
        ]
        opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)


class DefaultTeacher(KnowledgeTeacher):
    pass
