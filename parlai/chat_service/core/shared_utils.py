#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import uuid
import yaml
import importlib
from collections import namedtuple

# Sleep constants
THREAD_SHORT_SLEEP = 0.1
THREAD_MEDIUM_SLEEP = 0.3
# ThrottlingException might happen if we poll too frequently
THREAD_MTURK_POLLING_SLEEP = 10

logger = None
logging_enabled = True
debug = True
log_level = logging.ERROR

if logging_enabled:
    logging.basicConfig(
        filename=str(time.time()) + '.log',
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG,
    )
    logger = logging.getLogger('mturk')


def set_log_level(new_level):
    global log_level
    log_level = new_level


def set_is_debug(is_debug):
    global debug
    debug = is_debug


def print_and_log(level, message, should_print=False):
    if logging_enabled and level >= log_level:
        logger.log(level, message)
    if should_print or debug:  # Always print message in debug mode
        print(message)


def generate_event_id(worker_id):
    """
    Return a unique id to use for identifying a packet for a worker.
    """
    return '{}_{}'.format(worker_id, uuid.uuid4())


class TaskState:
    """
    Wrapper for an agent running on a Worker.
    """

    def __init__(
        self, task_name, world_name, agents, is_overworld=False, world_type=None
    ):
        self.task_name = task_name
        self.world_name = world_name
        self.agents = agents
        self.is_overworld = is_overworld  # (bool): overworld or task world
        self.world_type = world_type  # name of the task world returned by the overworld

        self.future = None
        self.world = None  # world object


WorldConfig = namedtuple(
    "WorldConfig",
    [
        "world_name",
        "onboarding_name",
        "task_name",
        "max_time_in_pool",
        "agents_required",
        "backup_task",
    ],
)


def get_world_module(world_path):
    """
    Import the module specified by the world_path.
    """
    run_module = None
    try:
        run_module = importlib.import_module(world_path)
    except Exception as e:
        print("Could not import world file {}".format(world_path))
        raise e
    return run_module


def get_world_fn_attr(world_module, world_name, fn_name, raise_if_missing=True):
    """
    Import and return the function from world.

    :param world_module:
        module. a python module encompassing the worlds
    :param world_name:
        string. the name of the world in the module
    :param fn_name:
        string. the name of the function in the world
    :param raise_if_missing:
        bool. if true, raise error if function not found

    :return:
        the function, if defined by the world.
    """
    result_fn = None
    try:
        DesiredWorld = getattr(world_module, world_name)
        result_fn = getattr(DesiredWorld, fn_name)
    except Exception as e:
        if raise_if_missing:
            print("Could not find {} for {}".format(fn_name, world_name))
            raise e
    return result_fn


def get_eligibility_fn(world_module, world_name):
    """
    Get eligibility function for a world.

    :param world_module:
        module. a python module encompassing the worlds
    :param world_name:
        string. the name of the world in the module

    :return:
        the eligibility function if available, else None
    """
    return get_world_fn_attr(
        world_module, world_name, 'eligibility_function', raise_if_missing=False
    )


def get_assign_roles_fn(world_module, world_name):
    """
    Get assign roles function for a world.

    :param world_module:
        module. a python module encompassing the worlds
    :param world_name:
        string. the name of the world in the module

    :return:
        the assign roles function if available, else None
    """
    return get_world_fn_attr(
        world_module, world_name, 'assign_roles', raise_if_missing=False
    )


def default_assign_roles_fn(agents):
    """
    Assign agent role.

    Default role assignment.

    :param:
        list of agents
    """
    for i, a in enumerate(agents):
        a.disp_id = f'Agent_{i}'


def parse_configuration_file(config_path):
    """
    Read the config file for an experiment to get ParlAI settings.

    :param config_path:
        path to config

    :return:
        parsed configuration dictionary
    """
    result = {}
    result["configs"] = {}
    with open(config_path) as f:
        cfg = yaml.load(f.read())
        # get world path
        result["world_path"] = cfg.get("world_module")
        if not result["world_path"]:
            raise ValueError("Did not specify world module")
        result["overworld"] = cfg.get("overworld")
        if not result["overworld"]:
            raise ValueError("Did not specify overworld")
        result["max_workers"] = cfg.get("max_workers")
        if not result["max_workers"]:
            raise ValueError("Did not specify max_workers")
        result["task_name"] = cfg.get("task_name")
        if not result["task_name"]:
            raise ValueError("Did not specify task name")
        task_world = cfg.get("tasks")
        if task_world is None or len(task_world) == 0:
            raise ValueError("task not in config file")
        # get task file
        for task_name, configuration in task_world.items():
            if "task_world" not in configuration:
                raise ValueError("{} does not specify a task".format(task_name))
            result["configs"][task_name] = WorldConfig(
                world_name=task_name,
                onboarding_name=configuration.get("onboard_world"),
                task_name=configuration.get("task_world"),
                max_time_in_pool=configuration.get("timeout") or 300,
                agents_required=configuration.get("agents_required") or 1,
                backup_task=configuration.get("backup_task"),
            )
        # get world options, additional args
        result["world_opt"] = cfg.get("opt", {})
        result["additional_args"] = cfg.get("additional_args", {})

    return result
