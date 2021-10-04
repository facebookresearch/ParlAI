#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiles the final dataset (a json file) from the this Mephisto crowdsourcing task.

Example use:

    python compile_results.py --task-name wizard-of-internet --output-folder=/dataset/wizard-internet
"""

from typing import Dict, Union
import parlai.utils.logging as logging
from parlai.crowdsourcing.projects.wizard_of_internet.wizard_internet_blueprint import (  # noqa: F401
    WIZARD_INTERNET_PARLAICHAT_BLUEPRINT,
)
from parlai.crowdsourcing.utils.analysis import AbstractResultsCompiler
from mephisto.abstractions.blueprint import AgentState

# Roles of the task
WIZARD = 'Wizard'
APPRENTICE = 'Apprentice'
SEARCH_AGENT = 'SearchAgent'

# Constant names with many reuse
CONTENTS = 'contents'
SELECTED_CONTENTS = 'selected_contents'
DIALOG_HISTORY_KEY = 'dialog_history'


def is_persona_form_response(message):
    k1 = 'task_data'
    k2 = 'form_responses'
    return message and k1 in message and k2 in message[k1]


def get_human_sender(message):
    sender = message['id']
    if sender in (WIZARD, APPRENTICE, SEARCH_AGENT):
        return sender


def remove_keys_from_dict(dictionary_data, keys):
    for rm_key in keys:
        del dictionary_data[rm_key]
    return dictionary_data


def chat_interruption(message):
    for k in ('requested_finish', 'MEPHISTO_is_submit'):
        if k in message and message[k]:
            return True
    return False


class ChatMessage:
    """
    Container for keeping the content of one interaction between agents.
    """

    def __init__(self, message_dict: dict) -> None:
        self._message = message_dict
        self._sender = get_human_sender(self._message)

    def _format_action(self, receiver):
        return f'{self._sender} => {receiver}'

    def get_action(self):
        if self._sender == APPRENTICE:
            return self._format_action(WIZARD)

        if self._sender == SEARCH_AGENT:
            return self._format_action(WIZARD)

        # Must be WIZARD
        k = 'is_search_query'
        if k in self._message and self._message[k]:
            return self._format_action(SEARCH_AGENT)
        else:
            return self._format_action(APPRENTICE)

    def get_text(self):
        if self._sender == SEARCH_AGENT:
            return ''
        return self._message.get('text', '')

    def get_context(self):
        if self._sender == APPRENTICE:
            return {}

        context_data = self._message.get('task_data', None)

        if self._sender == WIZARD:
            if not context_data:
                return {}
            else:
                if 'form_responses' in context_data:
                    return {'Persona': context_data['form_responses'][0]['response']}
                return {
                    CONTENTS: context_data.get('text_candidates', ''),
                    SELECTED_CONTENTS: context_data.get('selected_text_candidates', ''),
                }
        # Must be SEARCH_AGENT
        return {CONTENTS: context_data['search_results']}

    def compile_message(self):
        d = dict()
        d['action'] = self.get_action()
        d['text'] = self.get_text()
        d['context'] = self.get_context()
        return d


class WizardOfInternetResultsCompiler(AbstractResultsCompiler):
    """
    Compiles the results of Wizard of Internet crowdsourcing task into a json dataset.
    """

    def is_unit_acceptable(self, unit_data):
        # Depending on the situation (in practice) we may be able to salvage incomplete data.
        # Here, we only keep completed and approved ones (discarding the rest).
        return unit_data['status'] in (
            AgentState.STATUS_ACCEPTED,
            AgentState.STATUS_APPROVED,
        )

    def format_chat_data(self, dialog_history):
        data_dict = {'apprentice_persona': '', DIALOG_HISTORY_KEY: []}

        for message in dialog_history:
            if message['id'] == 'PersonaAgent':
                data_dict['apprentice_persona'] = message['task_data'][
                    'apprentice_persona'
                ]
                continue

            if is_persona_form_response(message) or not get_human_sender(message):
                continue

            if chat_interruption(message):
                # There was interruption (could be a clean Finish)
                # Ignoring the rest of messages
                break

            else:
                compiled_message = ChatMessage(message).compile_message()
                data_dict[DIALOG_HISTORY_KEY].append(compiled_message)

        return data_dict

    def compile_results(self) -> Dict[str, Dict[str, Union[dict, str]]]:

        logging.info('Retrieving task data from Mephisto.')
        task_units_data = self.get_task_data()
        logging.info(f'Data for {len(task_units_data)} units loaded successfully.')

        results = dict()
        for work_unit in task_units_data:
            assignment_id = work_unit['assignment_id']

            data = work_unit['data']
            agent_name = data['agent_name']
            if agent_name == 'Wizard':
                # Only collecting Wizard side data. Because it contains
                # data that is needed from the apprentice side too.
                formatted_chat_data = self.format_chat_data(data['messages'])
                results[assignment_id] = formatted_chat_data

        logging.info(f'{len(results)} dialogues compiled.')
        return results


if __name__ == '__main__':
    parser_ = WizardOfInternetResultsCompiler.setup_args()
    args = parser_.parse_args()
    opt = {
        'task_name': args.task_name,
        'results_format': 'json',
        'output_folder': args.output_folder,
    }
    wizard_data_compiler = WizardOfInternetResultsCompiler(opt)
    wizard_data_compiler.compile_and_save_results()
