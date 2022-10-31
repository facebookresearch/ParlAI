#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Optional, List
from types import MethodType
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import random
import spacy
import torch

from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.agents import Agent
from parlai.core.metrics import F1Metric
from parlai.agents.rag.args import setup_rag_args
from parlai.agents.bart.bart import BartAgent
from parlai.core.metrics import normalize_answer
from parlai.tasks.wizard_of_wikipedia.agents import (
    TOKEN_NOCHOSEN,
)

from parlai.tasks.wizard_of_wikipedia.agents import (
    TOKEN_KNOWLEDGE,
    TOKEN_END_KNOWLEDGE,
)

STOP_WORDS = stopwords.words('english')


def load_opt_from_file(opt_file):
    if not opt_file.endswith('.opt'):
        opt_file += '.opt'
    return Opt.load(opt_file)


def wow_get_batch_context(self, batch, orig_fun=None):
    """
    Set the beam context for n-gram context blocking specific for WoW data.

    For WoW, we don't want to consider the knowledge in the input for the context beam
    blocking. That's why we mask it out here.
    """
    ctxts = orig_fun(batch)
    knowledge_start_id = self.dict.txt2vec(TOKEN_KNOWLEDGE)
    knowledge_end_id = self.dict.txt2vec(TOKEN_END_KNOWLEDGE)

    def mask_ctxttensor_between_sublists(
        ctxts: torch.Tensor, sub1: List[int], sub2: List[int]
    ) -> torch.Tensor:
        """
        Generate a mask that masks out the context between sub1 and sub2.
        """
        mask = []
        for ctxt in ctxts:
            mask_idxs = []
            should_copy = False
            idx_pointer = 0
            id_to_match = sub1
            for j, token in enumerate(ctxt.cpu().numpy()):
                if token == id_to_match[idx_pointer]:
                    idx_pointer += 1
                    if idx_pointer == 1 and id_to_match == sub1:
                        mask_idxs.append([j])
                    elif idx_pointer >= len(id_to_match):
                        should_copy = id_to_match == sub1
                        idx_pointer = 0
                        id_to_match = sub2 if (id_to_match == sub1) else sub1
                        mask_idxs[-1].append(j)
                    else:
                        mask_idxs[-1].append(j)
                elif should_copy:
                    mask_idxs[-1].append(j)
                elif idx_pointer > 0:
                    idx_pointer = 0
                    del mask_idxs[-1]
            mask.append(
                [
                    0 if idx in [i for sl in mask_idxs for i in sl] else 1
                    for idx in range(len(ctxt))
                ]
            )
        return torch.LongTensor(mask).to(ctxts.device)

    ctxts *= mask_ctxttensor_between_sublists(
        ctxts, knowledge_start_id, knowledge_end_id
    )
    return ctxts


def find_supporting_sentence(question: str, answer: str, docs: List[str]) -> str:
    """
    Finds the supporting sentence for the answer in the docs.
    """
    # Remove the title of the documents.
    for i, doc in enumerate(docs):
        if ' | ' in doc:
            docs[i] = '. '.join(doc.split(' | ')[1:])
    concat_docs = '. '.join(docs)
    sentences = sent_tokenize(concat_docs)

    # Sort sentences according to recall with the answer and question.
    sorted_sentences = sorted(
        sentences,
        key=lambda sentence: (
            F1Metric._prec_recall_f1_score(
                normalize_answer(answer).split(), normalize_answer(sentence).split()
            )[0],
            F1Metric._prec_recall_f1_score(
                normalize_answer(question).split(), normalize_answer(sentence).split()
            )[0],
        ),
        reverse=True,
    )

    return sorted_sentences[0]


def extract_entities(
    sentence, pos=('PROPN', 'NOUN'), use_named_entities=True, use_noun_chunks=True
):
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    results = []
    if pos:
        for token in doc:
            if token.pos_ in pos:
                results.append(token)
    if use_named_entities:
        for ent in doc.ents:
            results.append(ent)
    if use_noun_chunks:
        for chunk in doc.noun_chunks:
            if chunk.text.lower() not in STOP_WORDS:
                results.append(chunk)
    results = list(set([r.text for r in results]))
    return results


def extract_knowledge(txt: str) -> List[str]:
    if not txt or not txt.split():
        return []
    entities = extract_entities(txt)
    return [e.lower() for e in (entities if entities else txt.split())]


def knowledge_from_dialogue_response(dialogue_response: str) -> str:
    """
    Get a knowledge response based on the dialogue response.

    We use a random entity from the dialogue response as knowledge. If there is no
    entity present, we use a random word. If there are no words present, we use
    TOKEN_NOCHOSEN.
    """
    knowledge_options = extract_knowledge(dialogue_response)
    if not knowledge_options:
        return TOKEN_NOCHOSEN
    return random.choice(knowledge_options)


class StackedKnowledgeDialogueAgent(Agent):
    """
    Stacked model that generates first the knowledge, and then the dialogue response.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('StackedKnowledgeDialogueAgent options')

        additional_agent_parser = ParlaiParser(add_parlai_args=False)
        BartAgent.add_cmdline_args(additional_agent_parser)
        setup_rag_args(additional_agent_parser)

        for action in additional_agent_parser._actions:
            key = action.option_strings[-1]
            type = action.type

            for prefix in [
                'krm',  # knowledge response model
                'drm',  # dialogue response model
                'drmrw',  # dialogue response model rag-wiki
                'drmnk',  # dialogue response model no-knowledge
            ]:
                agent.add_argument(
                    f'--{prefix}-{key.strip("-")}',
                    type=type,
                    required=False,
                )

        agent.add_argument(
            '--knowledge-response-model-path',
            type=str,
            default=''
            'wow_knowledge_response_generation_sweep_1_Tue_Aug_17_1822/9aa/model',
            help='Model used to generate the knowledge response.',
        )
        agent.add_argument(
            '--dialogue-response-model-path',
            type=str,
            default='',
            help='Model used to generate the dialogue response.',
        )
        agent.add_argument(
            '--dialogue-response-no-knowledge-model-path',
            type=str,
            default='',
            help='Model used to generate the dialogue response without knowledge.',
        )
        agent.add_argument(
            '--dialogue-response-rag-wiki-model-path',
            type=str,
            default='',
            help='Model used to generate the dialogue response with Wiki knowledge.',
        )
        agent.add_argument(
            '--use-supporting-sentence-as-knowledge',
            type=bool,
            default=False,
            help='Instead of using the knowledge response directly to condition the dialogue'
            ' model, we search for the top supporting sentence and use this.',
        )
        agent.add_argument(
            '--beam-filter-for-knowledge-response',
            type=bool,
            default=False,
            help='Try to pick a beam that contains the knowledge response.',
        )
        agent.add_argument(
            '--beam-filter-questions',
            type=bool,
            default=False,
            help='Try to pick a beam that does not contain a question mark.',
        )
        agent.add_argument(
            '--beam-filter-self-references',
            type=bool,
            default=False,
            help='Try to pick a beam that does not contain self references like "I" and "me".',
        )
        agent.add_argument(
            '--beam-disregard-knowledge-for-context-blocking',
            type=bool,
            default=True,
            help='If True disregard the knowledge input for the context blocking.',
        )
        agent.add_argument(
            '--add-fixed-confidence',
            type=int,
            default=-1,
            help='Add a fixed confidence score of the knowledge response.',
        )
        agent.add_argument(
            '--add-confidence-as-str',
            type=bool,
            default=False,
            help='If we add a confidence score to the KRM, we add it as a str.',
        )
        return parser

    def __init__(self, opt, shared=None):
        self.id = 'StackedKnowledgeDialogueAgent'
        self._construct_opts(opt)

        self.knowledge_agent = None
        self.dialogue_agent = None
        self.dialogue_agent_no_knowledge = None
        self.dialogue_agent_rag_wiki = None

        if not shared:
            self._init_knowledge_model()
            self._init_dialogue_models()
        else:
            if 'knowledge_agent_share' in shared:
                self.knowledge_agent = create_agent_from_shared(
                    shared['knowledge_agent_share']
                )
            if 'dialogue_agent_share' in shared:
                self.dialogue_agent = create_agent_from_shared(
                    shared['dialogue_agent_share']
                )
            if 'dialogue_agent_no_knowledge_share' in shared:
                self.dialogue_agent = create_agent_from_shared(
                    shared['dialogue_agent_no_knowledge_share']
                )
            if 'dialogue_agent_rag_wiki_share' in shared:
                self.dialogue_agent = create_agent_from_shared(
                    shared['dialogue_agent_rag_wiki_share']
                )

        self.shared = shared
        super().__init__(opt, shared)

    @property
    def has_no_knowledge_dialogue_model(self):
        return (
            self.opts['init']['dialogue_response_no_knowledge_model_path']
            and self.opts['init']['dialogue_response_no_knowledge_model_path'] != 'None'
        )

    @property
    def has_rag_wiki_dialogue_model(self):
        return (
            self.opts['init']['dialogue_response_rag_wiki_model_path']
            and self.opts['init']['dialogue_response_rag_wiki_model_path'] != 'None'
        )

    def _agent_opt(
        self, filename, specific_override_args, general_override_args, **kwargs
    ):
        opt = load_opt_from_file(filename)
        opt['override'] = {}
        blocklist_general = ['model', 'model_file', 'init_model']
        standard_override_args = {
            'skip_generation': False,
            'inference': 'beam',
            'beam_block_ngram': 3,
            'beam_context_block_ngram': -1,
            'beam_size': 3,
        }
        general_override_args = {
            **general_override_args,
            **standard_override_args,
            **kwargs,
        }

        # Remove the prefix for the model for the specific override args.
        specific_override_args = {
            '_'.join(k.split('_')[1:]): v for k, v in specific_override_args.items()
        }

        # Specific for --indexer-type.
        if 'indexer_type' in specific_override_args:
            # TODO: do we also need to overwrite path_to_index?
            pass

        override_args = {**general_override_args, **specific_override_args}

        for k, v in override_args.items():
            if k not in blocklist_general and k in opt:
                opt['override'][k] = v

        return opt

    def _construct_opts(self, opt):
        self.opts = {}
        self.opts['init'] = opt
        override_opts = defaultdict(dict)
        for k, v in opt['override'].items():
            if k.startswith('krm_'):
                if v is not None:
                    override_opts['knowledge_agent'][k] = v
            elif k.startswith('drm_'):
                if v is not None:
                    override_opts['dialogue_agent'][k] = v
            elif k.startswith('drmnk_'):
                if v is not None:
                    override_opts['dialogue_agent_no_knowledge'][k] = v
            elif k.startswith('drmrw_'):
                if v is not None:
                    override_opts['dialogue_agent_rag_wiki'][k] = v
            else:
                override_opts['general'][k] = v
        self.opts['override'] = override_opts

        if opt['knowledge_response_model_path'] and opt[
            'knowledge_response_model_path'
        ] not in ['oracle']:
            self.opts['knowledge_agent'] = self._agent_opt(
                filename=opt['knowledge_response_model_path'],
                specific_override_args=override_opts['knowledge_agent'],
                general_override_args=override_opts['general'],
            )
        self.opts['dialogue_agent'] = self._agent_opt(
            filename=opt['dialogue_response_model_path'],
            specific_override_args=override_opts['dialogue_agent'],
            general_override_args=override_opts['general'],
        )
        if self.has_no_knowledge_dialogue_model:
            self.opts['dialogue_agent_no_knowledge'] = self._agent_opt(
                filename=opt['dialogue_response_no_knowledge_model_path'],
                specific_override_args=override_opts['dialogue_agent_no_knowledge'],
                general_override_args=override_opts['general'],
            )
        if self.has_rag_wiki_dialogue_model:
            self.opts['dialogue_agent_rag_wiki'] = self._agent_opt(
                filename=opt['dialogue_response_rag_wiki_model_path'],
                specific_override_args=override_opts['dialogue_agent_rag_wiki'],
                general_override_args=override_opts['general'],
            )

    def share(self):
        shared = super().share()
        shared['knowledge_agent_share'] = self.knowledge_agent.share()
        shared['dialogue_agent_share'] = self.dialogue_agent.share()
        if self.has_no_knowledge_dialogue_model:
            shared[
                'dialogue_agent_no_knowledge_share'
            ] = self.dialogue_agent_no_knowledge.share()
        if self.has_rag_wiki_dialogue_model:
            shared[
                'dialogue_agent_rag_wiki_share'
            ] = self.dialogue_agent_rag_wiki.share()
        return shared

    def _init_knowledge_model(self):
        # Initialize knowledge agent.
        if 'knowledge_agent' in self.opts:
            self.knowledge_agent = create_agent(
                self.opts['knowledge_agent'], requireModelExists=True
            )
            print('Options for Knowledge Response Agent')
            self.knowledge_agent.opt.log()
        elif self.opts['init']['knowledge_response_model_path'] == 'oracle':
            self.knowledge_agent = OracleKnowledgeAgent(self.opts['init'])

    def _init_dialogue_models(self):
        ## Init dialogue models.

        # Initialize dialogue agent that uses the predicted knowledge.
        self.dialogue_agent = create_agent(
            self.opts['dialogue_agent'], requireModelExists=True
        )
        # Monkey patch the get_batch_context to ignore the knowledge for
        # beam-context-blocking.
        if self.opts['init']['beam_disregard_knowledge_for_context_blocking']:
            orig_fun = self.dialogue_agent._get_batch_context
            self.dialogue_agent._get_batch_context = MethodType(
                lambda self, batch: wow_get_batch_context(
                    self, batch, orig_fun=orig_fun
                ),
                self.dialogue_agent,
            )

        print('Options for Dialogue Response Agent')
        self.dialogue_agent.opt.log()

        # Initialize dialogue agent that doesn't use knowledge.
        if self.has_no_knowledge_dialogue_model:
            self.dialogue_agent_no_knowledge = create_agent(
                self.opts['dialogue_agent_no_knowledge'], requireModelExists=True
            )

        # Initialize dialogue agent that uses RAG with Wiki.
        if self.has_rag_wiki_dialogue_model:
            self.dialogue_agent_rag_wiki = create_agent(
                self.opts['dialogue_agent_rag_wiki'], requireModelExists=True
            )

    def dialogue_reply(self, agent, observation):
        return self.batch_dialogue_reply(agent, [observation])[0]

    def batch_dialogue_reply(self, agent, observations):
        dialogue_observations = []
        # Observation for the dialogue model.
        for obs in observations:
            dialogue_observation = agent.observe(obs)
            agent.self_observe(dialogue_observation)
            dialogue_observations.append(dialogue_observation)

        return agent.batch_act(dialogue_observations)

    def generate_knowledge_observation(self, knowledges: List[str], observations):
        # Adjust the observation texts.
        knowledge_infused_observations = deepcopy(observations)
        for obs, knowledge in zip(knowledge_infused_observations, knowledges):
            if 'text' not in obs:
                continue
            text = obs.pop('text')

            if self.opts['init']['add_fixed_confidence'] >= 0:
                confidence = self.opts['init']['add_fixed_confidence']
                if self.opts['init']['add_confidence_as_str']:
                    confidence = {
                        0: 'low',
                        5: 'medium',
                        10: 'high',
                    }[confidence] + ' confidence'
                text += f'\n{TOKEN_KNOWLEDGE} {confidence}: {knowledge} {TOKEN_END_KNOWLEDGE}'
            else:
                text += f'\n{TOKEN_KNOWLEDGE} {knowledge} {TOKEN_END_KNOWLEDGE}'
            obs['text'] = text
        return knowledge_infused_observations

    def batch_act(self, observations):
        knowledge_agent_observations = [o['knowledge_agent'] for o in observations]
        raw_observations = [o['raw'] for o in observations]

        # Get the knowledge replies.
        if self.knowledge_agent is None:
            self._init_knowledge_model()
        batch_reply_knowledge = self.knowledge_agent.batch_act(
            knowledge_agent_observations
        )

        if (
            'top_docs' in batch_reply_knowledge[0]
            and self.opts['init']['use_supporting_sentence_as_knowledge']
        ):
            # The knowledge agent is a rag-style model. Instead of the actual knowledge
            # response, we will use the best matching sentence from the retrieved docs
            # as the knowledge conditioning.
            for i, reply_knowledge in enumerate(batch_reply_knowledge):
                reply_knowledge['support_sentence'] = find_supporting_sentence(
                    question=raw_observations[i]['text'],
                    answer=reply_knowledge['text'],
                    docs=reply_knowledge['top_docs'],
                )

        if self.dialogue_agent is None:
            self._init_dialogue_models()

        if self.dialogue_agent_no_knowledge:
            batch_reply_dialogue_no_knowledge = self.batch_dialogue_reply(
                self.dialogue_agent_no_knowledge, raw_observations
            )
        if self.dialogue_agent_rag_wiki:
            batch_reply_dialogue_rag_wiki = self.batch_dialogue_reply(
                self.dialogue_agent_rag_wiki, raw_observations
            )

        knowledge_infused_observations = self.generate_knowledge_observation(
            knowledges=[
                reply_knowledge.get('text', '')
                for reply_knowledge in batch_reply_knowledge
            ],
            observations=raw_observations,
        )
        batch_reply_dialogue = self.batch_dialogue_reply(
            self.dialogue_agent, knowledge_infused_observations
        )

        batch_reply_dialogue_knowledge_sentence = None
        if (
            self.opts['init']['use_supporting_sentence_as_knowledge']
            and 'support_sentence' in batch_reply_knowledge[0]
        ):
            knowledge_sentence_infused_observations = (
                self.generate_knowledge_observation(
                    knowledges=[
                        reply_knowledge.get('support_sentence', '')
                        for reply_knowledge in batch_reply_knowledge
                    ],
                    observations=raw_observations,
                )
            )
            batch_reply_dialogue_knowledge_sentence = self.batch_dialogue_reply(
                self.dialogue_agent, knowledge_sentence_infused_observations
            )

        for i in range(len(batch_reply_dialogue)):
            if batch_reply_knowledge and len(batch_reply_knowledge) > i:
                batch_reply_dialogue[i]['knowledge_response'] = batch_reply_knowledge[
                    i
                ].get('text', '')
                if 'support_sentence' in batch_reply_knowledge[i]:
                    batch_reply_dialogue[i]['support_sentence'] = batch_reply_knowledge[
                        i
                    ].get('support_sentence', '')
            if (
                self.dialogue_agent_no_knowledge
                and batch_reply_dialogue_no_knowledge
                and len(batch_reply_dialogue_no_knowledge) > i
            ):
                batch_reply_dialogue[i][
                    'text_no_knowledge'
                ] = batch_reply_dialogue_no_knowledge[i].get('text', '')

            if (
                self.dialogue_agent_rag_wiki
                and batch_reply_dialogue_rag_wiki
                and len(batch_reply_dialogue_rag_wiki) > i
            ):
                batch_reply_dialogue[i][
                    'text_rag_wiki'
                ] = batch_reply_dialogue_rag_wiki[i].get('text', '')

            if (
                batch_reply_dialogue_knowledge_sentence
                and len(batch_reply_dialogue_knowledge_sentence) > i
            ):
                batch_reply_dialogue[i][
                    'text_knowledge_sentence'
                ] = batch_reply_dialogue_knowledge_sentence[i].get('text', '')

        [
            self._filter_beams(
                reply=reply,
                filter_for_knowledge=self.opts['init'][
                    'beam_filter_for_knowledge_response'
                ],
                filter_questions=self.opts['init']['beam_filter_questions'],
                filter_self_references=self.opts['init']['beam_filter_self_references'],
            )
            for reply in batch_reply_dialogue
        ]

        return batch_reply_dialogue

    def _filter_beams(
        self,
        reply,
        filter_for_knowledge: bool = True,
        filter_questions: bool = False,
        filter_self_references: bool = False,
    ):
        knowledge = normalize_answer(reply['knowledge_response'])
        self_references = [
            'I live',
            'I love',
            ' me ',
            'my favorite',
            'My favorite',
            'do you know',
            'I have',
            'I like ',
            'My ',
        ]
        question_words = [
            'who',
            'when',
            'where',
            'what',
            'do you',
            'are you',
        ]

        def filter_fn(text: str) -> bool:
            normalized_text = normalize_answer(text)
            if filter_for_knowledge and knowledge not in normalized_text:
                return False
            if filter_questions and (
                '?' in text or any([qw in normalized_text for qw in question_words])
            ):
                return False
            if filter_self_references and any([ref in text for ref in self_references]):
                return False
            return True

        if not (
            'text' in reply
            and 'beam_texts' in reply
            and 'knowledge_response' in reply
            and len(reply['beam_texts']) > 1
        ):
            return

        beam_texts = [
            text
            for text, _ in sorted(reply['beam_texts'], key=lambda x: x[1], reverse=True)
        ]
        # print('\t' + '\n\t'.join(beam_texts[:10]))
        for text in [reply['text']] + beam_texts:
            if filter_fn(text):
                reply.force_set('text', text)
                return

    def self_observe(self, self_message: Message) -> None:
        # Hack: Feed back the final dialogue response to the knowledge model.
        # This is why we need to make sure that --mutators flatten.
        self.knowledge_agent.self_observe(self_message)

    def observe(self, observation):
        # Delete unused keys.
        for key in ['label_candidates', 'knowledge']:
            if key in observation:
                del observation[key]

        label_key = 'eval_labels' if 'eval_labels' in observation else 'labels'
        if 'nqopen' in self.opts['init']['task'].lower():
            knowledge_target = observation.get(label_key, '')
            if isinstance(knowledge_target, tuple):
                knowledge_target = '\t'.join(knowledge_target)
            observation['knowledge_target'] = knowledge_target
            observation['dialogue_response'] = ''
        else:
            observation['dialogue_response'] = observation.get(label_key, '')
            observation['knowledge_response'] = observation.get('checked_sentence', '')

        if self.knowledge_agent is None:
            self._init_knowledge_model()

        observations = {
            'raw': deepcopy(observation),
            'knowledge_agent': self.knowledge_agent.observe(observation),
        }
        self.observations = observations
        return observations

    def act(self):
        """
        Call batch_act with the singleton batch.
        """
        response = self.batch_act([self.observations])[0]
        self.self_observe(response)
        return response


class OracleKnowledgeAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'OracleKnowledgeAgent'

    def get_knowledge(self, obs):
        labels_kword = 'labels' if 'train' in self.opt['datatype'] else 'eval_labels'
        if 'wizardofwikipedia' in self.opt['task'].lower().replace('_', ''):
            return obs.get('checked_sentence', '')
        if 'wizardofinternet' in self.opt['task'].lower().replace('_', ''):
            knowledge = obs.get('__selected-sentences__', '')
            if isinstance(knowledge, list):
                knowledge = '\n'.join(knowledge)
            return knowledge
        elif (
            'nqopen' in self.opt['task'].lower()
            or 'natural_questions' in self.opt['task'].lower()
        ):
            return obs.get(labels_kword, '')
        elif 'LightTeacherPlus' in self.opt['task']:
            if labels_kword not in obs or not obs[labels_kword]:
                return ''
            labels = obs[labels_kword]
            if not isinstance(labels, str):
                labels = labels[0]
            return knowledge_from_dialogue_response(labels)
        elif 'SummaryQA' in self.opt['task']:
            return obs.get(labels_kword, '')
        else:
            raise NotImplementedError(f'Task "{self.opt["task"]}" is not known.')

    def self_observe(self, obs):
        pass

    def batch_act(self, observations):
        return [self.act(obs) for obs in observations]

    def act(self, obs=None):
        if not obs:
            obs = self.observation
        if obs is None:
            return {'text': 'Nothing to repeat yet.'}
        reply = {}
        reply['id'] = self.getID()
        knowledge = self.get_knowledge(obs)
        if not isinstance(knowledge, str):
            knowledge = random.choice(knowledge)
        reply['text'] = knowledge
        return Message(reply)
