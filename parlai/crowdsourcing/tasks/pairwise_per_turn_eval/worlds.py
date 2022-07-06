#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import parlai.utils.logging as logging
from typing import Any, Dict, Optional

from parlai.core.agents import create_agent_from_shared
from parlai.core.message import Message
from parlai.core.worlds import validate
from parlai.crowdsourcing.tasks.model_chat.constants import ONBOARD_SUCCESS
from parlai.crowdsourcing.tasks.model_chat.utils import Compatibility
from parlai.crowdsourcing.tasks.model_chat.worlds import (
    BaseModelChatWorld,
    ModelChatWorld,
    ModelChatOnboardWorld,
)
from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker
from parlai.crowdsourcing.utils.mturk import get_mturk_id_from_mephisto_wrapper
from parlai.crowdsourcing.tasks.pairwise_per_turn_eval.bot_agent import (
    PerTurnEvalTurkLikeAgent,
)


class PerTurnEvalWorld(ModelChatWorld):
    def __init__(self, opt, agent, bots, context_info: Optional[dict] = None):

        # num_turns turns for a single side, and really it appears to be
        # (num_turns + 1) * 2 total b/c of the "Hi!" and first bot utterance

        # TODO: this logic is very close to that of BaseModelChatWorld.__init__(). Can
        #  any of this be deduplicated?

        super(BaseModelChatWorld, self).__init__(opt, agent)

        num_turns = opt['num_turns']
        max_resp_time = opt['max_resp_time']

        self.opt = opt
        self.bots = bots

        self.task_turn_idx = 0
        self.num_turns = num_turns

        self.dialog = []
        self.tag = f'conversation_id {agent.mephisto_agent.db_id}'
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.check_acceptability = opt['check_acceptability']
        self.acceptability_checker = AcceptabilityChecker()
        self.block_qualification = opt['block_qualification']

        self.final_chat_data = None
        # TODO: remove this attribute once chat data is only stored in the Mephisto
        #  TaskRun for this HIT (see .get_custom_task_data() docstring for more
        #  information)

        # below are timeout protocols
        self.max_resp_time = max_resp_time  # in secs
        logging.info(
            f'Creating {self.__class__.__name__} for tag {self.tag} with {num_turns} turns.'
        )

        if context_info is not None:
            self.context_info = context_info
            self.personas = [
                self.context_info['persona_1_strings'],
                self.context_info['persona_2_strings'],
            ]
        else:
            self.context_info = {}
            self.personas = None

    def __add_problem_data_to_utterance(self, p, turn_idx: int):
        """
        Attach problem data to the bot's prior utterance, given by turn_idx.
        """
        logging.info(p)
        assert (
            self.dialog[turn_idx]['agent_idx'] == 1
        ), 'Problem data must be attached to a bot utterance.'
        assert (
            'problem_data' not in self.dialog[turn_idx]
        ), "Don't overwrite existing problem data!"
        self.dialog[turn_idx]['problem_data'] = p

    def parley(self):
        """
        The main function that controls the logic of the task. Uses self.task_turn_idx
        to control the sequence of the conversation.

        Specifically, when self.task_turn_idx is even, we know that the bots just gave
        their potential responses, and that it is the human's turn to choose one of the
        responses and give a justification value.

        When self.task_turn_idx is odd, we know that the human just chose one of the
        bots' responses, and now needs to respond to that response.

        self.task_turn_idx is initially 0, and during _run_initial_turn() the UI is
        redrawn to have the human select between the bots' responses. Then,
        self.task_turn_idx is incremented to 1.

        During self.agent.observe(), the UI is redrawn for the following human input,
        and during self.agent.act(), the code awaits the human input.
        """

        logging.info(
            f'{self.__class__.__name__}:{self.tag}: is at task_turn_idx '
            f'{self.task_turn_idx}, with {self.num_turns} pairs of turns needed...'
        )

        if self.task_turn_idx == 0:
            self._run_initial_turn()
            self.task_turn_idx += 1
            return

        logging.info(
            f'{self.__class__.__name__}:{self.tag}: About to act with task turn idx: '
            f'{self.task_turn_idx}'
        )

        # At this point, we know that the human now needs to respond to the bot's
        # response that the human just chose

        # We retrieve information regarding the human's choice and justification using
        # self.agent.act()

        human_choose_bot_response_act = self.agent.act(timeout=self.max_resp_time)
        human_choose_bot_response_act = Message(
            Compatibility.maybe_fix_act(human_choose_bot_response_act)
        ).json_safe_payload()

        logging.info(
            f'Got act for human, act was: {human_choose_bot_response_act} and '
            f'self.task_turn_idx: {self.task_turn_idx}.'
        )

        accepted_bot_response = human_choose_bot_response_act['task_data'][
            'accepted_bot_response'
        ]
        accepted_bot_id = human_choose_bot_response_act['task_data']['accepted_bot_id']
        accepted_bot_justification_value = human_choose_bot_response_act['task_data'][
            'justification_value'
        ]

        not_accepted_bot_response = human_choose_bot_response_act['task_data'][
            'not_accepted_bot_response'
        ]
        not_accepted_bot_id = human_choose_bot_response_act['task_data'][
            'not_accepted_bot_id'
        ]

        # We have both bots observe the accepted bot's response so that the conversation
        # history stays the same

        self.bots[0].observe(accepted_bot_response)
        self.bots[1].observe(accepted_bot_response)

        task_data = {}

        accepted_bot_utterance_data = {
            'text': accepted_bot_response['text'].split('<br>')[0],
            'id': accepted_bot_id,
        }
        not_accepted_bot_utterance_data = {
            'text': not_accepted_bot_response['text'].split('<br>')[0],
            'id': not_accepted_bot_id,
        }
        bot_utterance_data = {
            'agent_idx': 1,
            'accepted_bot_data': accepted_bot_utterance_data,
            'not_accepted_bot_data': not_accepted_bot_utterance_data,
            'human_choice': accepted_bot_id,
            'human_justification': accepted_bot_justification_value,
        }
        self.dialog.append(bot_utterance_data)

        self._postprocess_acts(acts=None, agent_idx=0)

        # All logic and processing for this step has now been done, so we do
        # self.agent.observe() to send the accepted response back to the frontend to
        # display and update task turn index, as well as await for the next action,
        # which is the human typing their response

        task_data['task_turn_idx'] = self.task_turn_idx
        # The UI will ask the human to respond to the chosen bot response
        self.agent.observe(
            {'text': accepted_bot_response['text'], 'task_data': task_data}
        )

        # Make self.task_turn_idx even now
        self.task_turn_idx += 1

        # Check for whether 6 pairs of turns has been done, since the last message of a
        # conversation should always be the bot's response

        if (
            human_choose_bot_response_act is not None
            and human_choose_bot_response_act.get('task_data', {}).get('finished')
            is not None
        ):
            self.chat_done = True
            # agent ends chat after exceeding minimum number of turns

            # Bot has just responded. Any problem data received now will be
            # regarding this bot's utterance

            # Get the final chat data
            self.final_chat_data = self.get_final_chat_data()

            # Soft-block the worker if there were acceptability violations
            acceptability_violations = self.final_chat_data['acceptability_violations'][
                0
            ]
            if acceptability_violations is not None and acceptability_violations != '':
                logging.info(
                    f'**NOTE** Acceptability violations detected: '
                    f'{acceptability_violations}'
                )
                # Grant the failed qualification
                self.agent.mephisto_agent.get_worker().grant_qualification(
                    self.block_qualification, 1
                )

            return

        logging.info(
            f'[human agent] self.task_turn_idx: {self.task_turn_idx}, self.dialog is: '
            f'{self.dialog}'
        )

        logging.info(
            f'Got act for human, act was: {human_choose_bot_response_act} and '
            f'self.task_turn_idx: {self.task_turn_idx}.'
        )

        # At this point, we know that the human now needs to respond to the bot's
        # response that the human just chose

        # We retrieve information regarding the human's response using self.agent.act()

        human_response_act = self.agent.act(timeout=self.max_resp_time)

        # Again, we have both bots observe the human response so that the conversation
        # history stays the same
        self.bots[0].observe(validate(human_response_act))
        self.bots[1].observe(validate(human_response_act))

        # Check that the models' conversation histories are the same
        bot_1_history = self.bots[0].model_agent.history.history_strings
        bot_2_history = self.bots[1].model_agent.history.history_strings

        assert (
            bot_1_history == bot_2_history
        ), f"The two bots' conversation histories are different.\nBot 1 history: {bot_1_history}\nBot 2 history: {bot_2_history}"

        # After the bots have observed the human response, it's time for them to produce
        # their response to the human using self.bots.act()

        bot_1_response = self.bots[0].act()
        bot_1_response = Compatibility.maybe_fix_act(bot_1_response)

        bot_2_response = self.bots[1].act()
        bot_2_response = Compatibility.maybe_fix_act(bot_2_response)

        # We display the result to the frontend randomly so there is no selection bias.
        # Also, we attach our result to task_data to send arbitrary data to the frontend

        if random.random() > 0.5:
            task_data = {
                'top_bot_data': {
                    'top_bot_id': self.bots[0].worker_id,
                    'top_bot_response': bot_1_response,
                },
                'bottom_bot_data': {
                    'bottom_bot_id': self.bots[1].worker_id,
                    'bottom_bot_response': bot_2_response,
                },
                'task_turn_idx': self.task_turn_idx,
            }
        else:
            task_data = {
                'top_bot_data': {
                    'top_bot_id': self.bots[1].worker_id,
                    'top_bot_response': bot_2_response,
                },
                'bottom_bot_data': {
                    'bottom_bot_id': self.bots[0].worker_id,
                    'bottom_bot_response': bot_1_response,
                },
                'task_turn_idx': self.task_turn_idx,
            }

        human_utterance_data = {
            'agent_idx': 0,
            # Get rid of annotations HTML if it's the bot response
            'text': human_response_act['text'].split('<br>')[0],
            'id': human_response_act['id']
            if 'id' in human_response_act
            else 'NULL_ID',  # Person1 or Polyencoder
        }

        self.dialog.append(human_utterance_data)

        # Human has just responded. Any problem data received now will be regarding the
        # bot's prior utterance
        p = human_response_act['task_data'].get('problem_data_for_prior_message')
        if p is not None:
            turn_idx = -2
            # Attach the problem data to the second-to-last utterance, since the last
            # utterance is what the human just said
            self.__add_problem_data_to_utterance(p, turn_idx=turn_idx)

        self._postprocess_acts(acts=None, agent_idx=0)

        task_data['task_turn_idx'] = self.task_turn_idx

        # All logic and processing for this step has now been done, so we do
        # self.agent.observe() to send the two bots' responses back to the frontend to
        # display and update task turn index, as well as await for the next action,
        # which is the human choosing from the two responses and providing a
        # justification value

        # The UI will ask the human to choose between two bot responses and give a
        # justification
        logging.info(f'*** self.task_turn_idx: {self.task_turn_idx} ***')
        self.agent.observe({'text': '', 'task_data': task_data})

        # Make self.task_turn_idx odd now
        self.task_turn_idx += 1

        logging.info(
            f'[bot agent] self.task_turn_idx: {self.task_turn_idx}, self.dialog is: '
            f'{self.dialog}'
        )

    def get_final_chat_data(self) -> Dict[str, Any]:
        """
        Add relevant fields to the final chat data.
        """

        if self.check_acceptability:
            human_messages, violation_types = self._prepare_acceptability_checking()
            violations_string = self.acceptability_checker.check_messages(
                messages=human_messages,
                is_worker_0=False,
                violation_types=violation_types,
            )
        else:
            violations_string = None

        data = {
            'dialog': self.dialog,
            'workers': [get_mturk_id_from_mephisto_wrapper(self.agent)],
            'bad_workers': [],
            'acceptability_violations': (violations_string,),
            'hit_ids': [self.agent.mephisto_agent.task_run_id],
            'assignment_ids': [self.agent.mephisto_agent.assignment_id],
            'task_description': {
                'annotations_config': self.opt['annotations_config'],
                'model_1_nickname': self.bots[0].worker_id,
                'model_1_file': self.bots[0].model_agent.opt.get('model_file'),
                'model_1_opt': self.bots[0].model_agent.opt,
                'model_2_nickname': self.bots[1].worker_id,
                'model_2_file': self.bots[1].model_agent.opt.get('model_file'),
                'model_2_opt': self.bots[1].model_agent.opt,
            },
        }
        # 'bad_workers' is for compatibility. Before, it was only non-empty if a
        # worker abandoned, returned, etc. a HIT, but now we don't even save chat
        # data in that case
        if self.check_acceptability:
            data['acceptability_violations'] = (violations_string,)
            # Make a tuple for compatibility with a human/human conversation in
            # which we check both sides for acceptability

        context_data = {
            'personas': self.personas,
            'context_dataset': self.context_info.get('context_dataset'),
            'person1_seed_utterance': self.context_info.get('person1_seed_utterance'),
            'person2_seed_utterance': self.context_info.get('person2_seed_utterance'),
            'additional_context': self.context_info.get('additional_context'),
        }
        data.update(context_data)
        return data

    def _run_initial_turn(self) -> None:
        """
        Run the initial turn for both the human and the bot.

        Optionally show the bot its persona. If we are in Meena-like conversation mode
        show "Hi!" to the human and the bot and let the bot respond accordingly.

        Check parley() function for more information on the main logic.
        """
        control_msg = {"episode_done": False}

        if self.opt['include_persona']:
            # The Bot agent
            # We add the personas and 1/3 of the time WoW topic as the
            # first utterance in the history.
            # Previously for BST task, we also had a big first utterance
            # that gave instructions. Removing that for this task.
            persona_strings = [s.strip() for s in self.personas[1]]
            persona_utterance = self._get_persona_utterance(
                persona_strings=persona_strings,
                context_dataset=self.context_info['context_dataset'],
                additional_context=self.context_info['additional_context'],
                is_bot=True,
            )
            message = control_msg.copy()
            message['text'] = persona_utterance
            # The bot seeing its persona does not count as a "turn"
            self.bots[0].observe(validate(message), increment_turn=False)
            self.bots[1].observe(validate(message), increment_turn=False)

        if self.opt['conversation_start_mode'] == 'hi':
            logging.info('[Displaying "Hi!" only as per Meena task.]')
            if self.personas is not None:
                human_persona_strings = [s.strip() for s in self.personas[0]]
            else:
                human_persona_strings = ['', '']
            human_first_msg = {
                'episode_done': False,
                'id': self.agent.id,
                'text': 'Hi!',
                'fake_start': True,
                'agent_idx': 0,
                'task_data': {
                    'human_persona_string_1': human_persona_strings[0],
                    'human_persona_string_2': human_persona_strings[1],
                    'prompt_instruction': self.opt['task_question'],
                },
            }
            for k, v in control_msg.items():
                human_first_msg[k] = v

            # The first message is always "Hi", so we have both bots observe the message

            self.dialog.append(human_first_msg)
            self.agent.observe(validate(human_first_msg))
            self.bots[0].observe(validate(human_first_msg))
            self.bots[1].observe(validate(human_first_msg))

            bot_1_response = self.bots[0].act()
            bot_1_response = Compatibility.maybe_fix_act(bot_1_response)

            bot_2_response = self.bots[1].act()
            bot_2_response = Compatibility.maybe_fix_act(bot_2_response)

            if random.random() > 0.5:
                task_data = {
                    'top_bot_data': {
                        'top_bot_id': self.bots[0].worker_id,
                        'top_bot_response': bot_1_response,
                    },
                    'bottom_bot_data': {
                        'bottom_bot_id': self.bots[1].worker_id,
                        'bottom_bot_response': bot_2_response,
                    },
                    'task_turn_idx': self.task_turn_idx,
                }
            else:
                task_data = {
                    'top_bot_data': {
                        'top_bot_id': self.bots[1].worker_id,
                        'top_bot_response': bot_2_response,
                    },
                    'bottom_bot_data': {
                        'bottom_bot_id': self.bots[0].worker_id,
                        'bottom_bot_response': bot_1_response,
                    },
                    'task_turn_idx': self.task_turn_idx,
                }

            # Need an initial human's observe to observe the two choices from the bot
            self.agent.observe({'text': '', 'task_data': task_data})

        else:
            raise ValueError(
                f"Conversation start mode {self.opt['conversation_start_mode']} "
                f"not recognized!"
            )

    def shutdown(self):

        if self.chat_done:
            self.opt['run_statistics'][
                f'{self.bots[0].worker_id}:{self.bots[1].worker_id}'
            ] += 1
            logging.info(
                'Runs completed per model: '
                + ', '.join(
                    f'{model}: {count:d}'
                    for model, count in self.opt['run_statistics'].items()
                )
            )

        self.agent.shutdown()


def make_onboarding_world(opt, agent):
    return ModelChatOnboardWorld(opt, agent)


def validate_onboarding(data):
    """
    Check the contents of the data to ensure they are valid.
    """
    logging.info(f"Validating onboarding data {data}")
    messages = data['outputs']['messages']
    if len(messages) == 0:
        return False
    status_message = messages[-2]
    if status_message is None:
        return False
    final_status = status_message.get('final_status')
    return final_status == ONBOARD_SUCCESS


# TODO: find a better way to avoid duplicating this from model_chat world.py
def get_bot_worker(opt: Dict[str, Any], model_name: str) -> PerTurnEvalTurkLikeAgent:
    """
    Return a bot agent.

    Agent behaves like a crowdsource worker but actually wraps around a dialogue model.
    """
    semaphore = opt['semaphore']
    shared_bot_agents = opt['shared_bot_agents']
    num_turns = opt['num_turns']
    bot_agent = create_agent_from_shared(shared_bot_agents[model_name])
    bot_worker = PerTurnEvalTurkLikeAgent(
        opt,
        model_name=model_name,
        model_agent=bot_agent,
        num_turns=num_turns,
        semaphore=semaphore,
    )
    return bot_worker


def make_world(opt, agents):
    # Extract important components from opt
    statistics_condition = opt['statistics_condition']
    context_generator = opt['context_generator']

    agents[0].agent_id = "Worker"

    # Get context: personas, previous utterances, etc.
    if context_generator is not None:
        context_info = context_generator.get_context()
    else:
        context_info = None

    # Decide on a bot to use
    run_statistics = opt['run_statistics']
    with statistics_condition:
        remaining_counts_needed = [
            (m, c - run_statistics[m]) for (m, c) in opt['conversations_needed'].items()
        ]
        remaining_counts_needed.sort(reverse=True, key=lambda x: x[1])
        model_name = remaining_counts_needed[0][0]
        logging.info(f'Remaining conversation counts needed: {remaining_counts_needed}')
        logging.info(f'Choosing the "{model_name}" pair of models for the bot.')

    model_1, model_2 = model_name.split(":")

    bot_worker_1 = get_bot_worker(opt=opt, model_name=model_1)
    bot_worker_2 = get_bot_worker(opt=opt, model_name=model_2)

    logging.info(f'context info: {context_info}')

    # Creating a world for a pair of models
    return PerTurnEvalWorld(
        opt,
        agent=agents[0],
        bots=(bot_worker_1, bot_worker_2),
        context_info=context_info,
    )


def get_world_params():
    return {"agent_count": 1}
