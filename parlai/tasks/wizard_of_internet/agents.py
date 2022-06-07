#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import jsonlines
import os
from parlai.core.message import Message
from tqdm import tqdm
from parlai.core.metrics import F1Metric
from parlai.core.params import ParlaiParser, Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.data import DatatypeHelper
import parlai.utils.logging as logging
import parlai.tasks.wizard_of_internet.constants as CONST
from .build import build
import parlai.tasks.wizard_of_internet.mutators  # noqa: F401


def get_dtype(opt):
    return DatatypeHelper.fold(opt.get('datatype', 'train'))


def _path(opt):
    build(opt)
    dpath = os.path.join(opt['datapath'], CONST.DATASET_NAME)
    dtype = get_dtype(opt)
    return os.path.join(dpath, f'{dtype}.jsonl')


def get_single_val_from_dict(dialog_json):
    """
    Extracting the single dialogue in the JSON.
    """
    assert len(dialog_json) == 1
    return next(iter(dialog_json.values()))


def parse_agent_message(message_dict, agent_type):
    return {
        CONST.SPEAKER_ID: agent_type,
        CONST.MESSAGE_TEXT: message_dict[CONST.MESSAGE_TEXT],
    }


def parse_apprentice_message(message_dict):
    return parse_agent_message(message_dict, CONST.APPRENTICE)


def wizard_message_has_selection(message):
    sel_docs = message[CONST.CONTEXT][CONST.SELECTED_CONTENTS]
    if len(sel_docs) < 2:
        return False
    return not sel_docs[0][0]


def parse_wizard_message(message_dict, doc_lines_delim):
    def get_knowledge(msg_d):
        knowledge = {
            CONST.RETRIEVED_DOCS: [],
            CONST.RETRIEVED_SENTENCES: [],
            CONST.RETRIEVED_DOCS_URLS: [],
            CONST.RETRIEVED_DOCS_TITLES: [],
            CONST.SELECTED_DOCS: [],
            CONST.SELECTED_DOCS_URLS: [],
            CONST.SELECTED_DOCS_TITLES: [],
            CONST.SELECTED_SENTENCES: [],
        }
        docs = msg_d[CONST.CONTEXT][CONST.CONTENTS]
        selections = msg_d[CONST.CONTEXT][CONST.SELECTED_CONTENTS]

        # Checking the option that agents choose if there is no selected sentence
        has_selection = wizard_message_has_selection(msg_d)

        for doc_ind, doc in enumerate(docs):
            doc_lines = []
            doc_lines_selection = selections[doc_ind + 1]
            doc_selected = False
            for line_ind, line in enumerate(doc['content']):
                doc_lines.append(line)
                if has_selection and doc_lines_selection[line_ind]:
                    doc_selected = True
                    knowledge[CONST.SELECTED_SENTENCES].append(line)
            full_doc = doc_lines_delim.join(doc_lines)
            knowledge[CONST.RETRIEVED_SENTENCES].extend(doc_lines)
            knowledge[CONST.RETRIEVED_DOCS].append(full_doc)
            knowledge[CONST.RETRIEVED_DOCS_TITLES].append(doc['title'])
            knowledge[CONST.RETRIEVED_DOCS_URLS].append(doc['url'])
            if doc_selected:
                knowledge[CONST.SELECTED_DOCS_TITLES].append(doc['title'])
                knowledge[CONST.SELECTED_DOCS].append(full_doc)
                knowledge[CONST.SELECTED_DOCS_URLS].append(doc['url'])

        if not knowledge[CONST.RETRIEVED_DOCS]:
            knowledge[CONST.RETRIEVED_DOCS] = [CONST.NO_RETRIEVED_DOCS_TOKEN]
            knowledge[CONST.RETRIEVED_DOCS_URLS] = [CONST.NO_URLS]
            knowledge[CONST.RETRIEVED_DOCS_TITLES] = [CONST.NO_TITLE]

        if not knowledge[CONST.SELECTED_DOCS]:
            knowledge[CONST.SELECTED_DOCS] = [CONST.NO_SELECTED_DOCS_TOKEN]
            knowledge[CONST.SELECTED_DOCS_URLS] = [CONST.NO_URLS]
            knowledge[CONST.SELECTED_DOCS_TITLES] = [CONST.NO_TITLE]
            knowledge[CONST.SELECTED_SENTENCES] = [CONST.NO_SELECTED_SENTENCES_TOKEN]

        return knowledge

    d = parse_agent_message(message_dict, CONST.WIZARD)
    if message_dict[CONST.ACTION] == CONST.ACTION_WIZARD_TO_APPRENTICE:
        d.update(get_knowledge(message_dict))
    return d


def parse_search_results(message_dict, delim='; '):
    d = {CONST.SPEAKER_ID: CONST.SEARCH_AGENT}
    d[CONST.SEARCH_RESULTS] = message_dict[CONST.CONTEXT][CONST.CONTENTS]
    all_title = [
        f'({i+1}) {doc["title"]}' for i, doc in enumerate(d[CONST.SEARCH_RESULTS])
    ]
    d[CONST.MESSAGE_TEXT] = delim.join(all_title)
    return d


def remove_retrieved_docs_from_message(message: Message):
    message.force_set(CONST.RETRIEVED_DOCS, [CONST.NO_RETRIEVED_DOCS_TOKEN])
    message.force_set(CONST.RETRIEVED_DOCS_URLS, [CONST.NO_URLS])
    message.force_set(CONST.RETRIEVED_DOCS_TITLES, [CONST.NO_TITLE])


def remove_selected_docs_from_message(message: Message):
    message.force_set(CONST.SELECTED_DOCS, [CONST.NO_SELECTED_DOCS_TOKEN])
    message.force_set(CONST.SELECTED_SENTENCES, [CONST.NO_SELECTED_SENTENCES_TOKEN])
    message.force_set(CONST.SELECTED_DOCS_URLS, [CONST.NO_URLS])
    message.force_set(CONST.SELECTED_DOCS_TITLES, [CONST.NO_TITLE])


class WizardOfInternetBaseTeacher(DialogTeacher):
    """
    Base Teacher for Wizard of Internet tasks.

    This teachers loads full conversations and all the actions that happens
    during a full dialogue. Teachers that are drived from this class are
    responsible for slicing data and selecting the actions that they need.
    NOTE: Do NOT use this directly, use its children.
    """

    def __init__(self, opt: Opt, shared=None):
        opt = deepcopy(opt)
        self.datatype = get_dtype(opt)
        opt['datafile'] = _path(opt)
        self.include_persona = opt.get('include_persona', CONST.INCLUDE_PERSONA_DEFAULT)
        self.skip_empty_text = opt.get(
            'skip_empty_text', CONST.SKIP_ON_EMPTY_TEXT_DEFAULT
        )
        self.text_flatten_delimeter = opt.get('delimiter', '\n')
        self.docs_delim = opt.get('docs_delimiter', '\n')
        self.docs_titles_delimeter = opt.get('docs_title_delimiter', '\n')
        self.doc_lines_delim = opt.get('doc_lines_delimiter', '\n')
        self.id = 'WizInternetBase'
        super().__init__(opt, shared=shared)

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        arg_group = parser.add_argument_group('Wizard Base Dialog Teacher arguments')
        arg_group.add_argument(
            '--include-persona',
            type='bool',
            default=CONST.INCLUDE_PERSONA_DEFAULT,
            help='Whether to include the apprentice persona in text',
        )
        arg_group.add_argument(
            '--skip-empty-text',
            type='bool',
            default=CONST.SKIP_ON_EMPTY_TEXT_DEFAULT,
            help='Whether to skip response to empty messages (may happen if persona is not included)',
        )
        return parser

    def _load_data(self, datafile):
        logging.info(f'Loading data from {datafile} ...')
        with jsonlines.open(datafile, 'r') as fin:
            for dialog_json in tqdm(fin):
                yield self._get_episode_examples(dialog_json)

    def _get_episode_examples(self, dialog_json):
        data = get_single_val_from_dict(dialog_json)
        persona = data[CONST.PERSONA]
        output = defaultdict(list)
        msg_history = data[CONST.DIALOG_HIST]
        for msg_ind, message in enumerate(msg_history):
            d = {CONST.PERSONA: persona, CONST.TOTAL_CONVERSATION_INDEX: msg_ind}
            action = message[CONST.ACTION]

            # Separating the actions
            if action == CONST.ACTION_APPRENTICE_TO_WIZARD:
                d.update(parse_apprentice_message(message))
            elif action == CONST.ACTION_WIZARD_TO_APPRENTICE:
                # TODO: must avoid having these in the dataset in the first place
                assert message[CONST.MESSAGE_TEXT], 'Empty message text'

                if (  # Getting the search query that Wizard used for this utterance, if any
                    msg_ind > 1
                    and msg_history[msg_ind - 2][CONST.ACTION]
                    == CONST.ACTION_WIZARD_TO_SEARCH_AGENT
                ):
                    d[CONST.SEARCH_QUERY] = msg_history[msg_ind - 2][CONST.MESSAGE_TEXT]
                    # The last search query was the last one before sending the response
                    output[CONST.ACTION_WIZARD_TO_SEARCH_AGENT][-1][1][
                        CONST.IS_LAST_SEARCH_QUERY
                    ] = True
                else:
                    d[CONST.SEARCH_QUERY] = CONST.NO_SEARCH_QUERY_USED

                d.update(
                    parse_wizard_message(message, doc_lines_delim=self.doc_lines_delim)
                )
                if wizard_message_has_selection(message):
                    # Wizard had selection for this utterance, thus its a knowledge piece
                    output[CONST.ACTION_WIZARD_DOC_SELECTION].append((msg_ind, d))
            elif action == CONST.ACTION_WIZARD_TO_SEARCH_AGENT:
                d.update(
                    parse_wizard_message(message, doc_lines_delim=self.doc_lines_delim)
                )
                d[CONST.IS_SEARCH_QUERY] = True
            elif action == CONST.ACTION_SEARCH_AGENT_TO_WIZARD:
                # TODO: remove assert in the final version
                assert (
                    msg_history[msg_ind - 1][CONST.ACTION]
                    == CONST.ACTION_WIZARD_TO_SEARCH_AGENT
                )

                d.update(parse_search_results(message, self.docs_titles_delimeter))
                # Getting the send text (query) from the latest Wizard search
                d[CONST.SEARCH_QUERY] = output[CONST.ACTION_WIZARD_TO_SEARCH_AGENT][-1][
                    1
                ][CONST.MESSAGE_TEXT]

            # TODO: remove on the final version
            assert 'id' in d, str(message)

            # Getting current actions's previous message/action
            if (
                action == CONST.ACTION_APPRENTICE_TO_WIZARD
                and output[CONST.ACTION_WIZARD_TO_APPRENTICE]
            ):
                d[CONST.PARTNER_PREVIOUS_MESSAGE] = output[
                    CONST.ACTION_WIZARD_TO_APPRENTICE
                ][-1]
            elif output[CONST.ACTION_APPRENTICE_TO_WIZARD]:
                d[CONST.PARTNER_PREVIOUS_MESSAGE] = output[
                    CONST.ACTION_APPRENTICE_TO_WIZARD
                ][-1]

            output[action].append((msg_ind, d))
            output[CONST.ACTION_ALL].append(d)

        return output

    def create_parlai_message(self, dict_message: Dict):
        parlai_msg = Message(
            {
                CONST.SPEAKER_ID: dict_message[CONST.SPEAKER_ID],
                CONST.LABELS: [dict_message[CONST.MESSAGE_TEXT]],
            }
        )
        prv_msg = dict_message.get(CONST.PARTNER_PREVIOUS_MESSAGE)
        if prv_msg:
            parlai_msg[CONST.MESSAGE_TEXT] = prv_msg[1][CONST.MESSAGE_TEXT]
        else:
            parlai_msg[CONST.MESSAGE_TEXT] = ''
        return parlai_msg

    @abstractmethod
    def _teacher_action_type(self) -> str:
        """
        Is this for a Wizard or Apprentice Dialogue agent.
        """

    def additional_message_content(self, parlai_message: Message, action: Dict):
        """
        Children of this class may override this method to add extra content to message.

        It adds components from the original `action` (which is a regular dict) to the
        ParlAI message object `parlai_message`
        """
        pass

    def _opening_message_text(self, parlai_message: Message, action: Dict):
        """
        Handles the first message if this agent is has the opening message.
        """
        if not self.include_persona:
            return

        persona = action[CONST.PERSONA]
        curr_text = parlai_message[CONST.MESSAGE_TEXT]
        if curr_text:
            new_text = f'{persona}{self.text_flatten_delimeter}{curr_text}'
        else:
            new_text = persona

        parlai_message.force_set(CONST.MESSAGE_TEXT, new_text)

    def _should_skip(self, message: Message):
        if not self.skip_empty_text:
            return False
        return not message[CONST.MESSAGE_TEXT].strip()

    def setup_data(self, datafile) -> Message:
        for message, episode_started in self.teacher_setup_data(datafile):
            if not self._should_skip(message):
                yield message, episode_started

    def teacher_setup_data(self, datafile) -> Message:
        for data in self._load_data(datafile):
            started = True
            for idx, (_, act) in enumerate(data[self._teacher_action_type()]):
                parlai_msg = self.create_parlai_message(act)
                if idx == 0 and self.include_persona:
                    self._opening_message_text(parlai_msg, act)
                self.additional_message_content(parlai_msg, act)
                yield parlai_msg, started
                started = False


###############################################################
#                                                             #
# Dialog Teachers                                             #
#                                                             #
###############################################################


class ApprenticeDialogTeacher(WizardOfInternetBaseTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared=shared)
        self.id = 'WizInternetApprenticeTeacher'

    def _teacher_action_type(self) -> str:
        return CONST.ACTION_APPRENTICE_TO_WIZARD


class WizardDialogTeacher(WizardOfInternetBaseTeacher):
    def __init__(self, opt, shared=None):
        self.prepend_gold_knowledge = opt.get('prepend_gold_knowledge')
        self.gold_knowledge_delimiter = opt.get('gold_knowledge_delimiter', '\n')
        self.add_skip_search_if_gold_prepended = opt.get(
            'add_skip_search_if_gold_prepended'
        )
        super().__init__(opt, shared=shared)
        self.id = 'WizInternetWizardTeacher'

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        arg_group = parser.add_argument_group('Wizard Dialog Knowledge arguments')
        arg_group.add_argument(
            '--prepend-gold-knowledge',
            type='bool',
            default=False,
            help='If true, prepend text with checked sentences',
        )
        arg_group.add_argument(
            '--add-skip-search-if-gold-prepended',
            type='bool',
            default=False,
            help='If true, add skip search field when prepending text with checked sentences',
        )
        return parser

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        if not isinstance(teacher_action, Message):
            teacher_action = Message(teacher_action)
        if not isinstance(model_response, Message):
            model_response = Message(model_response)
        if (
            (
                teacher_action[CONST.SELECTED_SENTENCES][0]
                == CONST.NO_SELECTED_SENTENCES_TOKEN
            )
            or (model_response.is_padding())
            or ('text' not in model_response)
        ):
            # Has NOT selected knowledge or a is batch padding message
            return

        # F1 metric over the *selected* knowledge.
        resp = model_response['text']
        self.metrics.add(
            'knowledge_f1_docs',
            F1Metric.compute(resp, [' '.join(teacher_action[CONST.SELECTED_DOCS])]),
        )
        self.metrics.add(
            'knowledge_f1_max_docs', F1Metric.compute(resp, CONST.SELECTED_DOCS)
        )
        self.metrics.add(
            'knowledge_f1_sentences',
            F1Metric.compute(
                resp, [' '.join(teacher_action[CONST.SELECTED_SENTENCES])]
            ),
        )
        self.metrics.add(
            'knowledge_f1_max_sentences',
            F1Metric.compute(resp, CONST.SELECTED_SENTENCES),
        )

        # F1 Metrics over the *retrieved* docs.
        self.metrics.add(
            'knowledge_f1_max_retrieved_sentences',
            F1Metric.compute(resp, teacher_action[CONST.RETRIEVED_SENTENCES]),
        )
        self.metrics.add(
            'knowledge_f1_max_retrieved_docs',
            F1Metric.compute(resp, teacher_action[CONST.RETRIEVED_DOCS]),
        )

    def _teacher_action_type(self) -> str:
        return CONST.ACTION_WIZARD_TO_APPRENTICE

    def additional_message_content(self, parlai_message: Message, action: Dict):
        for item_key in (
            CONST.RETRIEVED_DOCS,
            CONST.RETRIEVED_SENTENCES,
            CONST.RETRIEVED_DOCS_URLS,
            CONST.RETRIEVED_DOCS_TITLES,
            CONST.SELECTED_DOCS,
            CONST.SELECTED_DOCS_URLS,
            CONST.SELECTED_DOCS_TITLES,
            CONST.SELECTED_SENTENCES,
            CONST.SEARCH_QUERY,
        ):
            parlai_message[item_key] = action[item_key]

    def teacher_setup_data(self, datafile) -> Message:
        for message, episode_started in super().teacher_setup_data(datafile):
            if self.prepend_gold_knowledge:
                text = message[CONST.MESSAGE_TEXT]
                gold_knowledge = self.gold_knowledge_delimiter.join(
                    message[CONST.SELECTED_SENTENCES]
                )
                message.force_set(
                    CONST.MESSAGE_TEXT,
                    (
                        f'{CONST.KNOWLEDGE_TOKEN} {gold_knowledge} {CONST.END_KNOWLEDGE_TOKEN}'
                        f' {self.gold_knowledge_delimiter} {text}'
                    ),
                )
                if self.add_skip_search_if_gold_prepended:
                    message[CONST.SKIP_SEARCH] = True
            yield message, episode_started


class WizardDialogGoldKnowledgeTeacher(WizardDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared=shared)
        self.id = 'WizardDialogGoldKnowledgeTeacher'

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser.set_params(prepend_gold_knowledge=True)
        return parser


class WizardDialogGoldKnowledgeNoDocsTeacher(WizardDialogGoldKnowledgeTeacher):
    """
    Prepends gold (selected knowledge) to the context, and removes the retrieved docs.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared=shared)
        self.id = 'WizardDialogGoldKnowledgeNoDocsTeacher'

    def additional_message_content(self, parlai_message: Message, action: Dict):
        super().additional_message_content(parlai_message, action)
        remove_retrieved_docs_from_message(parlai_message)


class DefaultTeacher(WizardDialogTeacher):
    pass


###############################################################
#                                                             #
# Search and Knowledge Teachers                               #
#                                                             #
###############################################################


class BaseSQKnowledgeTeacher(WizardOfInternetBaseTeacher):
    """
    Parent class for knowledge and search query generation teachers.

    We need this teacher because of the actions related to these teacher, that is search
    and knowledge selection, happens as a side track to the main conversation.
    Therefore, we do not want to include the history of the messages emitted by these
    agents in the conversatin history.

    Note: this is an abstract class and is not intended for direct use in a task.
    """

    def __init__(self, opt, shared=None):
        self.dialog_history = opt.get('dialog_history', CONST.DIALOG_HIST_DEFAULT)
        super().__init__(opt, shared=shared)
        self.id = 'BaseKnowledgeTeacher'

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        arg_group = parser.add_argument_group(
            'Base Search Query and Knowledge arguments'
        )
        arg_group.add_argument(
            '--dialog-history',
            type=str,
            choices=[CONST.HISTORY_TYPE.FULL, CONST.HISTORY_TYPE.ONLY_LAST],
            default=CONST.DIALOG_HIST_DEFAULT,
            help='Full dialogue history or the only the last previous message',
        )
        return parser

    def get_message_history(self, dialog_data: Dict, curr_idx: int) -> List[str]:
        message_hist = []
        for act in dialog_data[CONST.ACTION_ALL]:
            if act[CONST.SPEAKER_ID] in (
                CONST.WIZARD,
                CONST.APPRENTICE,
            ) and not act.get(CONST.IS_SEARCH_QUERY, False):
                if act[CONST.TOTAL_CONVERSATION_INDEX] > curr_idx:
                    break
                message_hist.append(act[CONST.MESSAGE_TEXT])

        if self.dialog_history == CONST.HISTORY_TYPE.ONLY_LAST:
            message_hist = [message_hist[-1]]

        return self.text_flatten_delimeter.join(message_hist)

    def teacher_setup_data(self, datafile) -> Message:
        for data in self._load_data(datafile):
            for idx, act in data[self._teacher_action_type()]:
                parlai_msg = self.create_parlai_message(act)
                if self.dialog_history == CONST.HISTORY_TYPE.FULL:
                    parlai_msg.force_set(
                        CONST.MESSAGE_TEXT, self.get_message_history(data, idx)
                    )
                self._opening_message_text(parlai_msg, act)
                self.additional_message_content(parlai_msg, act)
                yield parlai_msg, True


class SearchQueryTeacher(BaseSQKnowledgeTeacher):
    def __init__(self, opt, shared=None):
        self.only_last_search_query = opt.get(
            'only_last_search_query', CONST.ONLY_LAST_QUERY_DEFAULT
        )
        super().__init__(opt, shared=shared)
        self.id = 'SearchQueryGenerationTeacher'

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        arg_group = parser.add_argument_group('Search Query Teacher')
        arg_group.add_argument(
            '--only-last-search-query',
            type='bool',
            default=CONST.ONLY_LAST_QUERY_DEFAULT,
            help='Whether to include only the last search before sending the response.',
        )
        return parser

    def _teacher_action_type(self) -> str:
        return CONST.ACTION_WIZARD_TO_SEARCH_AGENT

    def additional_message_content(self, parlai_message: Message, action: Dict):
        parlai_message[CONST.IS_LAST_SEARCH_QUERY] = action.get(
            CONST.IS_LAST_SEARCH_QUERY, False
        )

    def teacher_setup_data(self, datafile) -> Message:
        for message, _ in super().teacher_setup_data(datafile):
            if self.only_last_search_query and not message[CONST.IS_LAST_SEARCH_QUERY]:
                continue
            yield message, True


class BaseKnowledgeTeacher(BaseSQKnowledgeTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared=shared)
        self.id = 'KnowledgeGenerationTeacher'

    def _teacher_action_type(self) -> str:
        return CONST.ACTION_WIZARD_DOC_SELECTION

    @abstractmethod
    def _knowledge_piece(self):
        """
        Determines the pieces of knowledge (selected content) to retrieve.

        This may be the enitre document, selected sentences or document titles.
        """

    def additional_message_content(self, parlai_message: Message, action: Dict):
        for item_key in (
            CONST.SELECTED_DOCS,
            CONST.SELECTED_DOCS_TITLES,
            CONST.SELECTED_SENTENCES,
        ):
            parlai_message[item_key] = action[item_key]

    def teacher_setup_data(self, datafile) -> Message:
        for message, _ in super().teacher_setup_data(datafile):
            message.force_set(CONST.LABELS, message[self._knowledge_piece()])
            yield message, True


class GoldKnowledgeTeacher(BaseKnowledgeTeacher):
    def _knowledge_piece(self):
        return CONST.SELECTED_SENTENCES


class GoldDocsTeacher(BaseKnowledgeTeacher):
    def _knowledge_piece(self):
        return CONST.SELECTED_DOCS


class GoldDocTitlesTeacher(BaseKnowledgeTeacher):
    def _knowledge_piece(self):
        return CONST.SELECTED_DOCS_TITLES
