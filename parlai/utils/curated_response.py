#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random


"""
Functions to generate safe curated response that repalce an Agent's main response.
"""

####################################################################################
#
#   Init prompt: initial message to start a chit-chat.

STARTING_PROMPT_MESSAGES = (
    "Hey I am here. Let's chat!",
    "Hey, how are you doing today?",
    "Hello, how is your day going so far?",
    "Hi, I really like watchig movies, how about you?",
    "Hi, are you into sports?",
    "Have you seen any interesting movies recently?",
)


def generate_init_prompot():
    return random.choice(STARTING_PROMPT_MESSAGES)


####################################################################################
#
# Safe response
#
# A fall back method to avoid discussing sensetive and/or inappropriate topics.
# This may be used to generate a safe response that substitues the main response from an agent,
# in case the main response was flagged as inappropriate.

# The generated responses consist of three parts.
# 1-AVOID: Avoiding to continue the conversation about that topic, for example, "I don't know"
# 2-TEMPLATE: a template to start talking about a new topic, for example, " have you heard about "
# 3-TOPIC: the next topic to steer the conversation towards, for example, "new marvel movie"


AVOID = (
    "I don't know about that.",
    "I am not sure about that.",
    "I don't find that topic interesting.",
    "Let's talk about something else.",
)

TEMPLATE = (
    "Have you heard anything about",
    "I was recently read about",
    "I am really excited about",
    "I know many people are intereseted in",
    "I think it is excting to talk about",
)

TOPICS = (
    "new marvel movies.",
    "tonight game.",
    "next year Oscar.",
    "traveling to exotic destinations.",
    "read new books.",
    "recent news about space travel.",
    "scientific advacements in AI.",
    "celeberaties. How is you favorite?",
    "music.",
    "art.",
)


def generate_safe_response():
    resp_parts = []
    for part_pool in (AVOID, TEMPLATE, TOPICS):
        resp_parts.append(random.choice(part_pool))
    return " ".join(resp_parts)
