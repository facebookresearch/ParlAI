#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from parlai.core.agents import create_agent
from parlai.utils.strings import normalize_reply

# from model_override_configs import MODEL_OVERRIDE_CONFIGS
from parlai_internal.mturk.tasks.q_function.original.constants import (
    AGENT_1,
    CHECKBOXES_CONFIG,
)
from parlai_internal.mturk.tasks.q_function.original.utils import Compatibility


class TurkLikeAgent:
    """
    Will act like a Turker but actually contains a bot agent.
    """

    def __init__(self, opt, model_name, model_agent, num_turns, semaphore=None):
        self.opt = opt
        self.model_agent = model_agent
        self.id = AGENT_1
        self.num_turns = num_turns
        self.turn_idx = 0
        self.semaphore = semaphore
        self.worker_id = model_name
        self.hit_id = 'none'
        self.assignment_id = 'none'
        self.some_agent_disconnected = False
        self.hit_is_abandoned = False
        self.hit_is_returned = False
        self.disconnected = False
        self.hit_is_expired = False

    @staticmethod
    def construct_checkbox_html(turn_idx):
        css_style = 'margin-right:15px;'
        checkboxes_html = """<br><br><span style="font-style:italic;">Does this comment from your partner have any problems? (Check all that apply)<br>"""
        for c in CHECKBOXES_CONFIG:
            checkboxes_html += f"""<input type="checkbox" id="checkbox_{c["value"]}_{turn_idx}" name="checkbox_group_{turn_idx}" /><span style={css_style}>{c["name"]}</span>"""
        checkboxes_html += f'<br><br><div id="explanation_{turn_idx}"></div>'
        return checkboxes_html

    def act(self, timeout=None):
        _ = timeout  # The model doesn't care about the timeout
        if self.semaphore:
            with self.semaphore:
                act_out = self.model_agent.act()
        else:
            act_out = self.model_agent.act()

        checkboxes_html = TurkLikeAgent.construct_checkbox_html(self.turn_idx)

        if 'dict_lower' in self.opt and not self.opt['dict_lower']:
            # model is cased so we don't want to normalize the reply like below
            final_message_text = act_out['text']
        else:
            normalized_act_text = normalize_reply(act_out['text'])
            final_message_text = normalized_act_text + checkboxes_html

        if self.turn_idx >= self.num_turns * 2:
            radio_css_style = 'margin-left:5px;margin-right:15px;'
            radio_buttons_html = ''
            for i in range(1, 6):
                radio_buttons_html += f"""<input type="radio" id="radio_rating_{i}" name="radio_final_rating_group" value="{i}" /><span style={radio_css_style}>{i}</span>"""
            exceeds_min_turns = f"""<br><br><div>{self.num_turns} chat turns finished! Please rate your partner on a scale of 1-5, how much would you enjoy talking to this partner over the course of a long conversation? (1: not at all, 5: a lot)</div>
            {radio_buttons_html}
            <br>Then, please click the "Done" button to end the chat."""
            final_message_text += exceeds_min_turns
            act_out = Compatibility.backward_compatible_force_set(
                act_out, 'exceed_min_turns', True
            )

        act_out = Compatibility.backward_compatible_force_set(
            act_out, 'text', final_message_text
        )
        assert ('episode_done' not in act_out) or (not act_out['episode_done'])
        self.turn_idx += 1
        return {**act_out, 'episode_done': False, 'checked_radio_name_id': ''}

    def observe(self, observation):
        """
        Need to protect the observe also with a semaphore b/c of retnref where the
        retriever act() actually happens in the agent's observe()

        ALso to make sure history vec during bot-human chat collection is consistent with
        that of model training and interative mode from interative.py, a new flag first_turn_version is added:
        if first_turn_version = v1 (default) model_agent.observe is called twice in worlds.py when self.task_turn_idx ==1
            1st call: model_agent.observe({persona_utterance})
            2nd call: model_agent.observe({the actual conversation openning such as Hi! from Meena})

        if first_turn_version = v2: model_agent will only observe concatenated string (persona+text) when self.task_turn_idx ==1
            1st call: SKIP in worlds.py
            2nd call: model_agent.observe(): model_agent.observe({persona_utterance+'\n'+first_human_message})
        """
        print(
            f'OBSERVE self.turn_idx is {self.turn_idx} and observation is: {observation}'
        )
        new_ob = copy.deepcopy(observation)
        # if first_turn_version = v2: concatenate persona_text and text if persona_text is not None
        if (
            observation.get('first_turn_version', 'v1') == 'v2'
            and observation.get('persona_text', None) is not None
            and observation.get('text', None) is not None
        ):
            new_ob['text'] = '\n'.join([new_ob['persona_text'], new_ob['text']])

        if self.semaphore:
            with self.semaphore:
                self.model_agent.observe(new_ob)
        else:
            self.model_agent.observe(new_ob)
        print(
            f'in observe - self.turn_idx: {self.turn_idx}, observation["text"]: {new_ob["text"]}'
        )

        if observation['id'] != 'SYSTEM':
            # The first utterance for the bot is not seen. It includes a control
            # message for the left pane and the persona utterance + WoW topic
            # (unless opt["include_persona"] is False), so don't want
            # to increment the turn_idx there; human speaks first and says Hi!
            self.turn_idx += 1

    def shutdown(self):
        pass

    def reset(self):
        self.model_agent.reset()

    @staticmethod
    def get_bot_agents(opt: dict, active_models: list, datapath: str, no_cuda=False):
        model_overrides = {
            'datatype': 'valid',  # So we don't have to load the optimizer
            'encode_candidate_vecs': True,  # For pulling from fixed list cands
            'interactive_mode': True,
            'model_parallel': True
        }
        if no_cuda:
            # If we load many models at once, we have to keep it on CPU
            model_overrides['no_cuda'] = no_cuda
        else:
            print(
                'WARNING: MTurk task has no_cuda FALSE. Models will run on GPU. Will not work if loading many models at once.'
            )

        # Get the model nicknames from common folder and use them to load opts
        # from file, and add options specified in MODEL_CONFIGS
        base_model_folder = opt.get('base_model_folder', None)
        models_available = []
        for obj in os.listdir(base_model_folder):
            if os.path.isdir(os.path.join(base_model_folder, obj)):
                models_available.append(obj)
        print(
            f'Found {len(models_available)} models available for Mturk task in {base_model_folder}: {models_available}'
        )

        all_model_opts = {}
        print(f'Active models to use are: {active_models}')
        for model_nickname in models_available:
            model_path = os.path.join(base_model_folder, model_nickname, 'model')

            if model_nickname not in active_models:
                print(
                    f'Skipping available model because not in active models list: {model_nickname}.'
                )
                continue

            model_overrides_copy = copy.deepcopy(model_overrides)
            opt = {'model_file': model_path, 'override': model_overrides_copy}
            all_model_opts[model_nickname] = opt

        active_model_opt_dicts = {m: all_model_opts[m] for m in active_models}

        print(
            f'Got {len(list(active_model_opt_dicts.keys()))} active models with keys: {active_model_opt_dicts.keys()}.'
        )
        shared_bot_agents = {}
        for model_name, model_opt in active_model_opt_dicts.items():
            print('\n\n--------------------------------')
            print(f'model_name: {model_name}, opt_dict: {model_opt}')
            copied_opt_dict = copy.deepcopy(model_opt)
            model_agent = create_agent(model_opt, requireModelExists=True)

            # have to check that the options are set properly
            for k, v in copied_opt_dict.items():
                if k != 'override':
                    assert model_agent.opt[k] == v

            shared_bot_agents[model_name] = model_agent.share()
        return shared_bot_agents
