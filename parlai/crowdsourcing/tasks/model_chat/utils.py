#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
import random
import threading
import time
from typing import Any, Dict, Optional, Tuple

from parlai.core.message import Message
from parlai.core.metrics import Metric


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
    Stack of images and contexts to run through.

    Each element of the stack contains the image to show, context information such as
    persona strings and BlendedSkillTalk-style seed utterances, and a list of the
    workers who have seen the given pairing of image+context for a given model. Stack
    ensures that no worker will see the same image+context twice.
    """

    def __init__(self, opt):

        # Input params
        self.models = opt['models']
        self.evals_per_combo = opt['evals_per_image_model_combo']

        # Paths
        self.images_and_contexts_path = opt['images_and_contexts_path']
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

        # Make sure that the set of models is correct (i.e. in case we are loading in an
        # older obsolete version of the stack)
        if set(self.stack[0]['workers_by_model'].keys()) == set(self.models):
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
        (1) All stack entries belonging to models that are still in self.models will
          be kept
        (2) All models not in self.models will be removed from the stack
        (3) All models in self.models not in the stack will be added to the stack

        Return the new pointer value.
        """

        # Stretch the stack
        existing_models = set(self.stack[0]['workers_by_model'].keys())
        new_models = set(self.models)
        models_to_add = new_models.difference(existing_models)
        models_to_remove = existing_models.difference(new_models)
        print('\nStarting to stretch the stack.')
        print('Models to add: ', models_to_add)
        print('Models to remove: ', models_to_remove)
        models_to_add_list = sorted(list(models_to_add))
        for stack_entry in self.stack:
            orig_workers_by_model = stack_entry['workers_by_model']
            surviving_workers_by_model = {
                model: workers
                for model, workers in orig_workers_by_model.items()
                if model in new_models
            }
            new_workers_by_model = {model: [] for model in models_to_add_list}
            stack_entry['workers_by_model'] = {
                **surviving_workers_by_model,
                **new_workers_by_model,
            }
            assert set(stack_entry['workers_by_model']) == new_models

        pointer = self.get_pointer()

        return pointer

    def conditionally_save_stack(self):
        if time.time() - self.last_save_time > self.save_stack_interval:
            self.save_stack()

    def save_stack(self):
        """
        Save the stack to its regular location. Mark down the save time.
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

    def _need_more_convos(self, image_info: Dict[str, Any]) -> bool:
        """
        Returns True if, for the given pairing of image+context (`image_info`), we
        need at least 1 more conversation with any of the models that we're testing.
        """
        return any(
            len(workers) < self.evals_per_combo
            for workers in image_info['workers_by_model'].values()
        )

    def build_stack(self) -> int:
        print('[ Building stack from original file... ]')
        with open(self.images_and_contexts_path, 'r') as f:
            image_names_to_image_info = json.load(f)

        self.stack = []
        for image_name, image_info in image_names_to_image_info.items():
            self.stack.append(
                {
                    'image_filename': image_name,
                    'workers_by_model': {model: [] for model in self.models},
                    **image_info,
                }
            )

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

    def get_next_image(self, worker: str) -> Tuple[int, Dict[str, Any], str, bool]:
        """
        Returns the image name, persona strings, model name, etc. for the next HIT.

        Finds a pairing of image+context that we don't currently have enough
        conversations for, ensuring that the given worker will not have had a
        conversation employing this image+context before using any model. Returns
        the index of the given input+context, the context info itself, the name of the
        model under which to have a conversation, and a flag indicating whether
        there are no more image+context pairs to show this worker.
        """
        with self.next_image_lock:
            no_more_work = False

            # Find the next entry in the stack that needs more workers
            image_info = self._get_stack_entry(self.pointer)
            while image_info is not None and not self._need_more_convos(image_info):
                self.pointer += 1
                print(f'Pointer at {self.pointer}')
                image_info = self._get_stack_entry(self.pointer)

            # Find the next entry in the stack that the worker hasn't completed before
            worker_pointer = self.pointer
            while image_info is not None and (
                any(
                    worker in workers
                    for workers in image_info['workers_by_model'].values()
                )
                or not self._need_more_convos(image_info)
            ):
                print(f'Pointer for worker {worker} at {self.pointer}')
                worker_pointer += 1
                image_info = self._get_stack_entry(worker_pointer)

            # Deal with the case in which no entry is suitable for the worker
            if image_info is None:
                print(f'WARNING: getting a random stack for worker {worker}.')
                worker_pointer = random.randrange(len(self.stack))
                image_info = self.stack[worker_pointer]
                no_more_work = True
                # We'll want to assign this worker a qualification to prevent more work

            self.conditionally_save_stack()

            # Pick out a model for this worker, among the ones that we need more
            # conversations for
            available_models = [
                model
                for model, workers in image_info['workers_by_model'].items()
                if len(workers) < self.evals_per_combo
            ]
            if len(available_models) == 0:
                print(
                    f'WARNING: no more convos needed for any model for '
                    f'{worker_pointer:d}. Picking a random model for worker '
                    f'{worker}.'
                )
                available_models = list(image_info['workers_by_model'].keys())
            print(f'Available models: ' + ', '.join(available_models))
            chosen_model = random.choice(available_models)
            print(
                f'Retrieving stack {worker_pointer:d} for worker {worker} and test '
                f'case {chosen_model}.'
            )
            image_info['workers_by_model'][chosen_model].append(worker)

            return worker_pointer, image_info, chosen_model, no_more_work

    def remove_worker_from_stack(self, worker: str, stack_idx: int):
        if any(
            worker in workers
            for workers in self.stack[stack_idx]['workers_by_model'].values()
        ):
            removed = False
            print(f'Removing worker {worker} from stack {stack_idx:d}.')
            for this_models_workers in self.stack[stack_idx][
                'workers_by_model'
            ].values():
                if worker in this_models_workers:
                    this_models_workers.remove(worker)
                    removed = True
            assert removed is True
            if stack_idx < self.pointer:
                print(f'Moving pointer from {self.pointer:d} to {stack_idx:d}.')
                self.pointer = stack_idx
        else:
            raise ValueError(f'Worker {worker} not found in stack {stack_idx:d}!')
