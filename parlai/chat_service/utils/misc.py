#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Miscellaneous utils for chat services.
"""
import importlib
import math
import re
from enum import Enum

THREAD_MEDIUM_SLEEP = 0.3


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


class SafetyDetectionResult(Enum):
    """
    Result of identfying offensive language in a response.

    SAFE:       the message is safe
    BLOCKLIST:  the message contains a word from the blocklist
    UNSAFE:     the message is deemed unsafe by the safety classifier
    """

    SAFE = 0
    BLOCKLIST = 1
    UNSAFE = 2


class ReportResult(Enum):
    """
    Result of filing a report.

    FAILURE:    a player timed out while reporting, or it was an accidental report
    BLOCK:      a player is blocked, for having been reported > 1 times
    SUCCESS:    a successful report
    BOT:        the offending agent was the bot
    """

    FAILURE = 0
    BLOCK = 1
    SUCCESS = 2
    BOT = 3


class UploadImageResult(Enum):
    """
    Result of uploading an image.

    SUCCESS:        user successfully uploaded an image
    OBJECTIONABLE:  the image contains objectionable content
    ERROR:          there was an error
    """

    SUCCESS = 0
    OBJECTIONABLE = 1
    ERROR = 2


class PersonalInfoDetector(object):
    """
    Detects whether a string contains any of the following personal information
    datapoints using regular expressions:

    - credit card
    - phone number
    - email
    - SSN
    """

    def __init__(self):
        self.credit_card_regex = r"((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])"
        self.email_regex = (
            r"([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*"
            + r"[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"
        )
        self.phone_number_regex = (
            r"\D?(\d{0,3}?)\D{0,2}(\d{3})?\D{0,2}(\d{3})\D?(\d{4})$"
        )
        self.ssn_regex = r"^\d{3}-\d{2}-\d{4}$"

    def detect_all(self, text):
        contains = {}
        contains["credit_card"] = self.detect_credit_card(text)
        contains["email"] = self.detect_email(text)
        contains["phone_number"] = self.detect_phone_number(text)
        contains["ssn"] = self.detect_ssn(text)
        return contains

    def txt_format_detect_all(self, text):
        contains = self.detect_all(text)
        contains_personal_info = False
        txt = "We believe this text contains the following personal " + "information:"
        for k, v in contains.items():
            if v != []:
                contains_personal_info = True
                txt += f"\n- {k.replace('_', ' ')}: {', '.join([str(x) for x in v])}"
        if not contains_personal_info:
            return ""
        return txt

    def detect_credit_card(self, text):
        return re.findall(self.credit_card_regex, text)

    def detect_email(self, text):
        text = text.lower()
        return re.findall(self.email_regex, text)

    def detect_phone_number(self, text):
        phones = re.findall(self.phone_number_regex, text)
        edited = []
        for tup in phones:
            edited.append("".join(list(tup)))
        return edited

    def detect_ssn(self, text):
        return re.findall(self.ssn_regex, text)


class DictFrequencies:
    """
    Dict freqs.
    """

    def __init__(self, freqs):
        self.freqs = freqs
        self.N = sum(freqs.values())
        self.V = len(freqs)
        self.logNV = math.log(self.N + self.V)
