#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os

import parlai.utils.logging as logging
from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.utils.strings import normalize_reply
from parlai.mturk.tasks.turn_annotations.constants import AGENT_1
from parlai.mturk.tasks.turn_annotations.utils import (
    Compatibility,
    construct_annotations_html,
)


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

    def act(self, timeout=None):
        _ = timeout  # The model doesn't care about the timeout
        if self.semaphore:
            with self.semaphore:
                act_out = self.model_agent.act()
        else:
            act_out = self.model_agent.act()

        annotations_html = construct_annotations_html(
            annotations_intro=self.opt['annotations_intro'],
            annotations_config=self.opt['annotations_config'],
            turn_idx=self.turn_idx,
        )

        if 'dict_lower' in self.opt and not self.opt['dict_lower']:
            # model is cased so we don't want to normalize the reply like below
            final_message_text = act_out['text']
        else:
            normalized_act_text = normalize_reply(act_out['text'])
            final_message_text = normalized_act_text + annotations_html

        if self.turn_idx >= self.num_turns * 2:
            radio_css_style = 'margin-left:5px;margin-right:15px;'
            radio_buttons_html = ''
            for i in range(1, 6):
                radio_buttons_html += f"""<input type="radio" id="radio_rating_{i}" name="radio_final_rating_group" value="{i}" /><span style={radio_css_style}>{i}</span>"""
            final_scoring_question = self.opt['final_rating_question']
            exceeds_min_turns = f"""<br><br><div>{self.num_turns} chat turns finished! {final_scoring_question}</div>
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

    def observe(self, observation, increment_turn: bool = True):
        """
        Need to protect the observe also with a semaphore for composed models where an
        act() may be called within an observe()
        """
        print(
            f'{self.__class__.__name__}: In observe() before semaphore, self.turn_idx is {self.turn_idx} and observation is {observation}'
        )
        new_ob = copy.deepcopy(observation)
        if self.semaphore:
            with self.semaphore:
                self.model_agent.observe(new_ob)
        else:
            self.model_agent.observe(new_ob)
        print(
            f'{self.__class__.__name__}: In observe() AFTER semaphore, self.turn_idx: {self.turn_idx}, observation["text"]: {new_ob["text"]}'
        )

        if increment_turn:
            self.turn_idx += 1

    def shutdown(self):
        pass

    def reset(self):
        self.model_agent.reset()

    @staticmethod
    def get_bot_agents(opt: dict, active_models: list, no_cuda=False):
        model_overrides = {
            'datatype': 'valid',  # So we don't have to load the optimizer
            'encode_candidate_vecs': True,  # For pulling from fixed list cands
            'interactive_mode': True,
            'model_parallel': opt['task_model_parallel'],
        }
        if no_cuda:
            # If we load many models at once, we have to keep it on CPU
            model_overrides['no_cuda'] = no_cuda
        else:
            logging.warn(
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
        for model_nickname in active_models:
            model_overrides_copy = copy.deepcopy(model_overrides)
            model_path = os.path.join(base_model_folder, model_nickname, 'model')
            if os.path.isfile(model_path):
                model_opt = {'model_file': model_path, 'override': model_overrides_copy}
            else:
                model_opt_path = model_path + '.opt'
                print(
                    f'Model file for model {model_nickname} does not exist! Instead, '
                    f'loading opt from {model_opt_path}.'
                )
                model_opt = Opt.load(model_opt_path)
                if 'override' not in model_opt:
                    model_opt['override'] = {}
                model_opt['override'].update(model_overrides_copy)
            all_model_opts[model_nickname] = model_opt

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
