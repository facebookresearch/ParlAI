#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.messenger.tasks.qa_data_collection.worlds import QADataCollectionWorld
from parlai.messenger.core.worlds import SimpleMessengerOverworld as MessengerOverworld
import os
import importlib
import shared_utils as utils
import time
import datetime
from concurrent import futures
import logging


class MessengerWorldRunner:
    """A worker class that launches overworld and taskworld
    jobs based on an inputted module file"""

    def __init__(
        self,
        opt,
        world_path,
        max_workers,
        manager,
        is_debug=False,
    ):
        self._world_module = utils.get_world_module(world_path)
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.initialized = False
        self.debug = is_debug
        self._log("Found world module: {}".format(self._world_module))
        opt["is_debug"] = is_debug

        def _is_done_initializing(fut):
            if fut.result():
                print(fut.result())
            if self.debug:
                print("DEBUG: Call to `module_initialize` has completed...")
            self.initialized = True

        if hasattr(self._world_module, "module_initialize"):
            self._log("Initializing world module...")
            # perform any module intialization steps
            init_fn = self._world_module.module_initialize
            self.init_fut = self.executor.submit(
                init_fn,
                opt,
                manager,
            )
            self.init_fut.add_done_callback(_is_done_initializing)
        else:
            self._log("World module does not have `module initialize` function")
            self.initialized = True

        self.manager = manager  # need access to manager for dumping world state
        self.system_done = False
        self.opt = opt
        self.tasks = {}  # task ID to task

    def _log(self, text):
        if self.debug:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{} DEBUG (FBMessengerWorldRunner): {}".format(time, text))

    def shutdown(self, force=True):
        """Shutdown the world runner.

        If another worker is available, we do not force shutdown: instead,
        we iterate through the worlds one by one and gather state to transfer
        to a new worker.
        """
        for _, task in self.tasks.items():
            if task.world is not None:
                task.world.shutdown()

        self.system_done = True  # this forces worlds to stop executing parley
        self._log("Executor shutting down.")
        self.executor.shutdown()
        self._log("Shutdown complete.")

    def run_world(self, task, world_name, agents):
        """Import the module specified by the world_name and run the world
        until completion.

        Args:
            world_name: string. The name of the world in the module file
            agents: list. A list of agents that should be in the world.
            task_state: JSON. Previous task state if the worker died during
                the execution of this world.

        Returns:
            ret_val: last output of world's parley function. Return None if ERROR
            status: the status of the job. Return None if ok, the exception if ERROR
        """
        ret_val = None
        utils.print_and_log(logging.DEBUG, f'running world {world_name}')
        world_generator = utils.get_world_fn_attr(
            self._world_module, world_name, "generate_world"
        )
        world = world_generator(self.opt, agents)
        utils.print_and_log(logging.DEBUG, f'actual world: {world}')
        task.world = world
        while not world.episode_done() and not self.system_done:
            ret_val = world.parley()
            time.sleep(0.3)
        world.shutdown()
        utils.print_and_log(logging.DEBUG, f'ret val: {ret_val}')
        world_data = world.data if hasattr(world, "data") else ""
        return ret_val, world_data, task.state_transfer

    def launch_task_world(
        self,
        task_name,
        world_name,
        agents,
    ):
        """Launch a task world. Return the job's future. Note that the return
        value of the world is returned unused: any relevant data which should
        be sent back should be in the agents' data dictionary.

        Args:
            task_name: string. the name of the job thread
            world_name: string. the name of the task world in the module file
            agents: list. the list of agents to install in the world
        """
        task = utils.TaskState(task_name, world_name, agents)
        self.tasks[task_name] = task

        def _world_fn():
            print("Starting task {}...".format(task_name))
            return self.run_world(task, world_name, agents)

        fut = self.executor.submit(_world_fn)
        task.future = fut
        return fut

    def launch_overworld(
        self,
        task_name,
        overworld_name,
        onboard_map,
        agent,
        world_type=None,
    ):
        """Launch an overworld and a subsequent onboarding world. Return the
        job's future

        Args:
            task_name: string. the name of the job thread
            overworld_name: string. the name of the overworld in the module file
            onboard_map: map. a mapping of overworld return values to the names
                         of onboarding worlds in the module file.
            agent: The agent to run the overworld with
            world_type: if the worker died during execution of an onboarding
                world, this is the return value from the overworld
        """
        task = utils.TaskState(
            task_name,
            overworld_name,
            [agent],
            is_overworld=True,
            world_type=world_type,
        )
        self.tasks[task_name] = task
        utils.print_and_log(logging.DEBUG,'launching overworld')
        def _world_function():
            if task.world_type is None:
                utils.print_and_log(logging.DEBUG,'getting world type')
                world_type, _, overworld_exit = self.run_world(
                    task,
                    overworld_name,
                    [agent],
                )
                utils.print_and_log(logging.DEBUG,f'world_type: {world_type}')
                if world_type is None:
                    return None, overworld_exit
                task.world_type = world_type
            else:
                world_type = task.world_type
                overworld_exit = False

            # perform onboarding
            onboard_type = onboard_map.get(world_type)
            utils.print_and_log(logging.DEBUG,f'going to onboard: {onboard_type}')
            onboard_exit = False
            if onboard_type:
                _, _, onboard_exit = self.run_world(
                    task,
                    onboard_type,
                    [agent],
                )

            # silent exit: if the worker dies during the execution of the
            # overworld or onboarding world, silent_exit is True IFF the
            # world has an 'offload_state' function that allows the state
            # to be transferred; otherwise it is False
            utils.print_and_log(logging.DEBUG,f'done with onboard, returning {world_type}')

            silent_exit = overworld_exit or onboard_exit
            self.manager.add_agent_to_pool(
                self.manager.get_agent_state(agent.id), world_type
            )
            return world_type, silent_exit

        fut = self.executor.submit(_world_function)
        task.future = fut
        return fut
