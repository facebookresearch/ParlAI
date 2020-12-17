#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


class ContextStack:
    # TODO: revise
    """
    Stack of contexts to run through.

    Each element of the stack contains the image to show, context information such as
    persona strings and BST-style seed utterances, and a list of the workers who have
    seen the given pairing of image+context for a given test case. Stack ensures that no
    worker will see the same image+context twice.
    """

    def __init__(self, opt):

        # Input params
        self.test_cases = list(TEST_CASES_TO_TOP_LEVEL_OPTS.keys())
        self.is_local = opt['is_local']
        self.version_num = opt['version_num']
        self.evals_per_context = opt['evals_per_context']

        # Paths
        save_dir = opt['save_dir']
        if self.is_local:
            self.save_folder = os.path.join(save_dir, 'local')
            self.save_name = 'image_and_context_stack.json'
        else:
            self.save_folder = save_dir
            self.save_name = f'image_and_context_stack_v{self.version_num}.json'
        self.backup_save_folder = os.path.join(self.save_folder, '_stack_backups')
        for folder in [self.save_folder, self.backup_save_folder]:
            os.makedirs(folder, exist_ok=True)
        self.save_path = os.path.join(self.save_folder, self.save_name)

        # Saving params
        self.save_stack_interval = 60
        self.last_save_time = time.time()
        self.save_lock = threading.RLock()
        self.next_context_lock = threading.RLock()

        # Things that will be defined later
        self.stack = None

        self.pointer = self.build_or_load_stack()

        self.conditionally_save_stack()

    def load_stack(self) -> int:
        print(f'[ Loading stack from file... {self.save_path}]')
        with open(self.save_path, 'r') as f:
            self.stack = json.load(f)

        pointer = self.get_pointer()

        # Make sure that the test cases are correct (i.e. in case we are loading in an
        # older obsolete version of the stack)
        if set(self.stack[0]['workers_by_test_case'].keys()) == set(self.test_cases):
            return pointer
        else:
            input_ = input(
                '\n\nWARNING: the currently saved stack has a different set of test '
                'cases than what is currently being used. Do you want to back up this '
                'stack file and stretch the stack to fit the new set of test cases? '
                '(y/n) '
            )
            if input_.lower().strip() == 'y':
                self.save_stack_backup()
                return self.stretch_stack()
            else:
                raise ValueError('Mismatch in set of test cases in stack!')

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
        "Stretch" the stack to handle the current set of test cases.

        The goal is to preserve as many existing stack entries as possible while
          matching the set of test cases in the stack with the new set of test cases in
          self.test_cases:
        (1) All stack entries belonging to test cases that are still in self.test_cases
          will be kept
        (2) All test cases not in self.test_cases will be removed from the stack
        (3) All test cases in self.test_cases not in the stack will be added to the
          stack

        Return the new pointer value.
        """

        # Stretch the stack
        existing_test_cases = set(self.stack[0]['workers_by_test_case'].keys())
        new_test_cases = set(self.test_cases)
        test_cases_to_add = new_test_cases.difference(existing_test_cases)
        test_cases_to_remove = existing_test_cases.difference(new_test_cases)
        print('\nStarting to stretch the stack.')
        print('Test cases to add: ', test_cases_to_add)
        print('Test cases to remove: ', test_cases_to_remove)
        test_cases_to_add_list = sorted(list(test_cases_to_add))
        for stack_entry in self.stack:
            orig_workers_by_test_case = stack_entry['workers_by_test_case']
            surviving_workers_by_test_case = {
                test_case: workers
                for test_case, workers in orig_workers_by_test_case.items()
                if test_case in new_test_cases
            }
            new_workers_by_test_case = {
                test_case: [] for test_case in test_cases_to_add_list
            }
            stack_entry['workers_by_test_case'] = {
                **surviving_workers_by_test_case,
                **new_workers_by_test_case,
            }
            assert set(stack_entry['workers_by_test_case']) == new_test_cases

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

    def _need_more_convos(self, context_info: Dict[str, Any]) -> bool:
        """
        Returns True if, for the given pairing of image+context (`context_info`), we
        need at least 1 more conversation with any of the test cases that we're testing.
        """
        return any(
            len(workers) < self.evals_per_context
            for workers in context_info['workers_by_test_case'].values()
        )

    def build_stack(self) -> int:
        print('[ Building stack from original file... ]')
        data_file = {
            0: IMAGES_AND_CONTEXTS_PATH,
            1: IMAGES_AND_CONTEXTS_PATH,
            2: IMAGES_AND_CONTEXTS_PATH,
        }.get(self.version_num, IMAGES_AND_CONTEXTS_PATH)
        with open(data_file, 'r') as f:
            image_names_to_context_info = json.load(f)

        self.stack = []
        for image_name, context_info in image_names_to_context_info.items():
            self.stack.append(
                {
                    'image_filename': image_name,
                    'workers_by_test_case': {
                        test_case: [] for test_case in self.test_cases
                    },
                    **context_info,
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

    def get_next_context(self, worker: str) -> Tuple[int, Dict[str, Any], str, bool]:
        """
        Returns the image name, persona strings, test case name, etc. for the next HIT.

        Finds a pairing of image+context that we don't currently have enough
        conversations for, ensuring that the given worker will not have had a
        conversation employing this image+context before using any test case. Returns
        the index of the given input+context, the context info itself, the name of the
        test case under which to have a conversation, and a flag indicating whether
        there are no more image+context pairs to show this worker.
        """
        with self.next_context_lock:
            no_more_work = False

            # Find the next entry in the stack that needs more workers
            context_info = self._get_stack_entry(self.pointer)
            while context_info is not None and not self._need_more_convos(context_info):
                self.pointer += 1
                print(f'Pointer at {self.pointer}')
                context_info = self._get_stack_entry(self.pointer)

            # Find the next entry in the stack that the worker hasn't completed before
            worker_pointer = self.pointer
            while context_info is not None and (
                any(
                    worker in workers
                    for workers in context_info['workers_by_test_case'].values()
                )
                or not self._need_more_convos(context_info)
            ):
                print(f'Pointer for worker {worker} at {self.pointer}')
                worker_pointer += 1
                context_info = self._get_stack_entry(worker_pointer)

            # Deal with the case in which no entry is suitable for the worker
            if context_info is None:
                print(f'WARNING: getting a random stack for worker {worker}.')
                worker_pointer = random.randrange(len(self.stack))
                context_info = self.stack[worker_pointer]
                no_more_work = True
                # We'll want to assign this worker a qualification to prevent more work

            self.conditionally_save_stack()

            # Pick out a test case for this worker, among the ones that we need more
            # conversations for
            available_test_cases = [
                test_case
                for test_case, workers in context_info['workers_by_test_case'].items()
                if len(workers) < self.evals_per_context
            ]
            if len(available_test_cases) == 0:
                print(
                    f'WARNING: no more convos needed for any test case for '
                    f'{worker_pointer:d}. Picking a random test case for worker '
                    f'{worker}.'
                )
                available_test_cases = list(context_info['workers_by_test_case'].keys())
            print(f'Available test cases: ' + ', '.join(available_test_cases))
            chosen_test_case = random.choice(available_test_cases)
            print(
                f'Retrieving stack {worker_pointer:d} for worker {worker} and test '
                f'case {chosen_test_case}.'
            )
            context_info['workers_by_test_case'][chosen_test_case].append(worker)

            return worker_pointer, context_info, chosen_test_case, no_more_work

    def remove_worker_from_stack(self, worker: str, stack_idx: int):
        if any(
            worker in workers
            for workers in self.stack[stack_idx]['workers_by_test_case'].values()
        ):
            removed = False
            print(f'Removing worker {worker} from stack {stack_idx:d}.')
            for this_test_cases_workers in self.stack[stack_idx][
                'workers_by_test_case'
            ].values():
                if worker in this_test_cases_workers:
                    this_test_cases_workers.remove(worker)
                    removed = True
            assert removed is True
            if stack_idx < self.pointer:
                print(f'Moving pointer from {self.pointer:d} to {stack_idx:d}.')
                self.pointer = stack_idx
        else:
            raise ValueError(f'Worker {worker} not found in stack {stack_idx:d}!')
