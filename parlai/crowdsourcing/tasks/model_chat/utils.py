#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
import datetime
import json
import os
import random
import threading
import time
import unittest
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from parlai.core.message import Message
from parlai.core.metrics import Metric
from parlai.core.params import ParlaiParser
from parlai.core.loader import load_task_module
from parlai.crowdsourcing.utils.tests import AbstractParlAIChatTest
from parlai.tasks.blended_skill_talk.agents import ContextGenerator


class Compatibility(object):
    """
    Class to address backward compatibility issues with older ParlAI models.
    """

    @staticmethod
    def backward_compatible_force_set(act, key, value):
        if isinstance(act, Message):
            act.force_set(key, value)
        elif isinstance(act, dict):
            act[key] = value
        else:
            raise Exception(f'Unknown type of act: {type(act)}')
        return act

    @staticmethod
    def maybe_fix_act(incompatible_act):
        if 'id' not in incompatible_act:
            new_act = Compatibility.backward_compatible_force_set(
                incompatible_act, 'id', 'NULL_ID'
            )
            return new_act
        return incompatible_act

    @staticmethod
    def serialize_bot_message(bot_message):
        if 'metrics' in bot_message:
            metric_report = bot_message['metrics']
            bot_message['metrics'] = {
                k: v.value() if isinstance(v, Metric) else v
                for k, v in metric_report.items()
            }
        return bot_message


class ImageStack:
    """
    Represents a stack of images to run through.

    Each element of the stack contains a list of the workers who have seen the given
    image for a given model. The stack ensures that no worker will see the same image
    twice.
    """

    def __init__(self, opt):

        # Input params
        self.num_images = opt['num_images']
        self.models = opt['models']
        self.evals_per_combo = opt.get('evals_per_image_model_combo', 1)

        # Paths
        self.save_folder = opt['stack_folder']
        self.save_name = 'stack.json'
        self.backup_save_folder = os.path.join(self.save_folder, '_stack_backups')
        for folder in [self.save_folder, self.backup_save_folder]:
            os.makedirs(folder, exist_ok=True)
        self.save_path = os.path.join(self.save_folder, self.save_name)

        # Saving params
        self.save_stack_interval = 60
        self.last_save_time = time.time()
        self.save_lock = threading.RLock()
        self.next_image_lock = threading.RLock()

        # Things that will be defined later
        self.stack = None

        self.pointer = self.build_or_load_stack()

        self.conditionally_save_stack()

    def load_stack(self) -> int:
        print(f'[ Loading stack from file... {self.save_path}]')
        with open(self.save_path, 'r') as f:
            self.stack = json.load(f)

        pointer = self.get_pointer()

        # Check that the number of images is the same as before
        if len(self.stack) != self.num_images:
            raise ValueError(
                f'The loaded stack has {len(self.stack):d} images instead of the '
                f'desired {self.num_images:d}!'
            )

        # Make sure that the set of models is correct (i.e. in case we are loading in an
        # older obsolete version of the stack)
        if set(self.stack[0].keys()) == set(self.models):
            return pointer
        else:
            input_ = input(
                '\n\nWARNING: the currently saved stack has a different set of test '
                'cases than what is currently being used. Do you want to back up this '
                'stack file and stretch the stack to fit the new set of models? '
                '(y/n) '
            )
            if input_.lower().strip() == 'y':
                self.save_stack_backup()
                return self.stretch_stack()
            else:
                raise ValueError('Mismatch in set of models in stack!')

    def get_pointer(self) -> int:
        """
        Return the index of the first entry in the stack that needs more conversations.
        """
        pointer = 0
        found = False
        while not found:
            if self._need_more_convos(self.stack[pointer]):
                found = True
            else:
                pointer += 1
        return pointer

    def stretch_stack(self) -> int:
        """
        "Stretch" the stack to handle the current set of models.

        The goal is to preserve as many existing stack entries as possible while
        matching the set of models in the stack with the new set of models in
        self.models:
        - (1) All stack entries belonging to models that are still in self.models will
        be kept
        - (2) All models not in self.models will be removed from the stack
        - (3) All models in self.models not in the stack will be added to the stack

        Return the new pointer value.
        """

        # Stretch the stack
        existing_models = set(self.stack[0].keys())
        new_models = set(self.models)
        models_to_add = new_models.difference(existing_models)
        models_to_remove = existing_models.difference(new_models)
        print('\nStarting to stretch the stack.')
        print('Models to add: ', models_to_add)
        print('Models to remove: ', models_to_remove)
        models_to_add_list = sorted(list(models_to_add))
        for stack_idx, orig_workers_by_model in enumerate(self.stack):
            surviving_workers_by_model = {
                model: workers
                for model, workers in orig_workers_by_model.items()
                if model in new_models
            }
            new_workers_by_model = {model: [] for model in models_to_add_list}
            self.stack[stack_idx] = {
                **surviving_workers_by_model,
                **new_workers_by_model,
            }
            assert set(self.stack[stack_idx]) == new_models

        pointer = self.get_pointer()

        return pointer

    def conditionally_save_stack(self):
        if time.time() - self.last_save_time > self.save_stack_interval:
            self.save_stack()

    def save_stack(self):
        """
        Save the stack to its regular location.

        Mark down the save time.
        """
        self._save_stack_to_path(self.save_path)
        self.last_save_time = time.time()

    def save_stack_backup(self):
        """
        Save a backup copy of the stack to a path with a datetime suffix.
        """
        suffix = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        backup_path = os.path.join(
            self.backup_save_folder, f'{self.save_name}.{suffix}'
        )
        self._save_stack_to_path(backup_path)

    def _save_stack_to_path(self, path: str):
        """
        Save stack to the specified path.
        """
        with self.save_lock:
            print(f'Saving all data to {path}.')
            data = json.dumps(self.stack)
            with open(path, 'w') as f:
                f.write(data)

    def _need_more_convos(self, workers_by_model: Dict[str, list]) -> bool:
        """
        Returns True if, for the given image, we need at least 1 more conversation with
        any of the models that we're testing.
        """
        return any(
            len(workers) < self.evals_per_combo for workers in workers_by_model.values()
        )

    def build_stack(self) -> int:
        print('[ Building stack... ]')
        self.stack = [
            {model: [] for model in self.models} for _ in range(self.num_images)
        ]
        return 0  # The pointer starts at 0

    def build_or_load_stack(self) -> int:
        # Check if this stack has been partially completed
        if os.path.isfile(self.save_path):
            return self.load_stack()
        else:
            return self.build_stack()

    def _get_stack_entry(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Return a stack entry if the input index is less than the length of the stack;
        return None otherwise.
        """
        if idx < len(self.stack):
            return self.stack[idx]
        else:
            return None

    def get_next_image(self, worker: str) -> Tuple[int, str, bool]:
        """
        Returns the image name, persona strings, model name, etc. for the next HIT.

        Finds an image that we don't currently have enough conversations for, ensuring
        that the given worker will not have had a conversation employing this image
        before, with any model. Returns the index of the given image, the name of the
        model with which to have a conversation, and a flag indicating whether there are
        no more image pairs to show this worker.
        """
        with self.next_image_lock:
            no_more_work = False

            # Find the next entry in the stack that needs more workers
            workers_by_model = self._get_stack_entry(self.pointer)
            while workers_by_model is not None and not self._need_more_convos(
                workers_by_model
            ):
                self.pointer += 1
                print(f'Pointer at {self.pointer}')
                workers_by_model = self._get_stack_entry(self.pointer)

            # Find the next entry in the stack that the worker hasn't completed before
            worker_pointer = self.pointer
            while workers_by_model is not None and (
                any(worker in workers for workers in workers_by_model.values())
                or not self._need_more_convos(workers_by_model)
            ):
                print(f'Pointer for worker {worker} at {self.pointer}')
                worker_pointer += 1
                workers_by_model = self._get_stack_entry(worker_pointer)

            # Deal with the case in which no entry is suitable for the worker
            if workers_by_model is None:
                print(f'WARNING: getting a random stack for worker {worker}.')
                worker_pointer = random.randrange(len(self.stack))
                workers_by_model = self.stack[worker_pointer]
                no_more_work = True
                # We'll want to assign this worker a qualification to prevent more work

            self.conditionally_save_stack()

            # Pick out a model for this worker, among the ones that we need more
            # conversations for
            available_models = [
                model
                for model, workers in workers_by_model.items()
                if len(workers) < self.evals_per_combo
            ]
            if len(available_models) == 0:
                print(
                    f'WARNING: no more convos needed for any model for '
                    f'{worker_pointer:d}. Picking a random model for worker '
                    f'{worker}.'
                )
                available_models = list(workers_by_model.keys())
            print(f'Available models: ' + ', '.join(available_models))
            chosen_model = random.choice(available_models)
            print(
                f'Retrieving stack {worker_pointer:d} for worker {worker} and test '
                f'case {chosen_model}.'
            )
            workers_by_model[chosen_model].append(worker)

            return worker_pointer, chosen_model, no_more_work

    def remove_worker_from_stack(self, worker: str, stack_idx: int):
        if any(worker in workers for workers in self.stack[stack_idx].values()):
            removed = False
            print(f'Removing worker {worker} from stack {stack_idx:d}.')
            for this_models_workers in self.stack[stack_idx].values():
                if worker in this_models_workers:
                    this_models_workers.remove(worker)
                    removed = True
            assert removed is True
            if stack_idx < self.pointer:
                print(f'Moving pointer from {self.pointer:d} to {stack_idx:d}.')
                self.pointer = stack_idx
        else:
            raise ValueError(f'Worker {worker} not found in stack {stack_idx:d}!')


class AbstractModelChatTest(AbstractParlAIChatTest, unittest.TestCase):
    """
    Abstract test class for testing model chat code.
    """

    def _remove_non_deterministic_keys(self, actual_state: dict) -> dict:

        # Remove non-deterministic keys from each message
        for message in actual_state['outputs']['messages']:
            for field in ['update_id', 'timestamp']:
                if field in message:
                    del message[field]

        # TODO: in `self._check_output_key()`, there is other logic for ignoring
        #  keys with non-deterministic values. Consolidate all of that logic here!
        custom_data = self._get_custom_data(actual_state)
        # Delete keys that will change depending on when/where the test is run
        for key in ['model_file']:
            del custom_data['task_description'][key]
        for key in ['datapath', 'dict_file', 'model_file', 'parlai_home', 'starttime']:
            if key in custom_data['task_description']['model_opt']:
                del custom_data['task_description']['model_opt'][key]
        for key in ['model_file']:
            if key in custom_data['task_description']['model_opt']['override']:
                del custom_data['task_description']['model_opt']['override'][key]

        return actual_state

    def _filter_agent_state_data(self, agent_state: dict) -> dict:
        """
        Remove agent state messages that do not contain text or final chat data and are
        thus not useful for testing the crowdsourcing task.
        """
        filtered_messages = [
            m
            for m in agent_state['outputs']['messages']
            if 'text' in m or 'final_chat_data' in m
        ]
        filtered_agent_state = {
            'inputs': agent_state['inputs'],
            'outputs': {**agent_state['outputs'], 'messages': filtered_messages},
        }
        return filtered_agent_state

    def _get_custom_data(self, actual_state: dict) -> dict:
        """
        Return the custom task data (without making a copy).

        The last message contains the custom data saved by the model-chat task code.
        """
        return actual_state['outputs']['messages'][-1]['WORLD_DATA']['custom_data']

    def _check_output_key(self, key: str, actual_value: Any, expected_value: Any):
        """
        Special logic for handling the 'final_chat_data' key.
        """
        if key == 'final_chat_data':
            self._check_final_chat_data(
                actual_value=actual_value, expected_value=expected_value
            )
        else:
            super()._check_output_key(
                key=key, actual_value=actual_value, expected_value=expected_value
            )

    def _check_final_chat_data(
        self, actual_value: Dict[str, Any], expected_value: Dict[str, Any]
    ):
        """
        Check the actual and expected values of the final chat data.

        TODO: this is hard to maintain. It'd be better to just delete the non-deterministic keys from actual_value beforehand, inside self._remove_non_deterministic_keys().
        """
        for key_inner, expected_value_inner in expected_value.items():
            if key_inner == 'dialog':
                assert len(actual_value[key_inner]) == len(expected_value_inner)
                for actual_message, expected_message in zip(
                    actual_value[key_inner], expected_value_inner
                ):
                    clean_actual_message = {
                        k: v for k, v in actual_message.items() if k != 'update_id'
                    }
                    clean_expected_message = {
                        k: v for k, v in expected_message.items() if k != 'update_id'
                    }
                    self.assertDictEqual(
                        clean_actual_message,
                        clean_expected_message,
                        f'The following dictionaries are different: {clean_actual_message} and {clean_expected_message}',
                    )
            elif key_inner == 'task_description':
                for (key_inner2, expected_value_inner2) in expected_value_inner.items():
                    if key_inner2 == 'model_file':
                        pass
                        # The path to the model file depends on the random
                        # tmpdir
                    elif key_inner2 == 'model_opt':
                        keys_to_ignore = [
                            'datapath',
                            'dict_file',
                            'model_file',
                            'override',
                            'parlai_home',
                            'starttime',
                        ]
                        # These paths depend on the random tmpdir and the host
                        # machine
                        for (
                            key_inner3,
                            expected_value_inner3,
                        ) in expected_value_inner2.items():
                            if key_inner3 in keys_to_ignore:
                                pass
                            else:
                                self.assertEqual(
                                    actual_value[key_inner][key_inner2][key_inner3],
                                    expected_value_inner3,
                                    f'Error in key {key_inner3}!',
                                )
                    else:
                        self.assertEqual(
                            actual_value[key_inner][key_inner2],
                            expected_value_inner2,
                            f'Error in key {key_inner2}!',
                        )
            else:
                self.assertEqual(
                    actual_value[key_inner],
                    expected_value_inner,
                    f'Error in key {key_inner}!',
                )


def get_context_generator(
    override_opt: Optional[Dict[str, Any]] = None,
    task: Optional[str] = 'blended_skill_talk',
    **kwargs,
) -> ContextGenerator:
    """
    Return an object to return BlendedSkillTalk-style context info (personas, etc.).
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    if override_opt is not None:
        argparser.set_params(**override_opt)
    opt = argparser.parse_args([])
    task_module = load_task_module(task)
    context_generator_class = getattr(task_module, 'ContextGenerator', None)
    context_generator = context_generator_class(opt, datatype='test', seed=0, **kwargs)
    # We pull from the test set so that the model can't regurgitate
    # memorized conversations
    return context_generator


def get_image_src(
    image: Optional[Image.Image] = None, path: Optional[str] = None
) -> str:
    """
    Given an image or the path to an image, return a string of the encoded image that
    can be used as the src field in an HTML img tag.
    """
    if image is None:
        image = Image.open(path)
    rgb_image = image.convert('RGB')
    buffered = BytesIO()
    rgb_image.save(buffered, format='JPEG')
    encoded = str(base64.b64encode(buffered.getvalue()).decode('ascii'))
    image_src = 'data:image/jpeg;base64,' + encoded
    return image_src
