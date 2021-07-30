#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Possible roles of an agent during task
NO_ROLE = 0
WIZARD = 1
APPRENTICE = 2
IN_TRAINING = 10
WIZARD_IN_TRAINING = WIZARD + IN_TRAINING
APPRENTICE_IN_TRAINING = APPRENTICE + IN_TRAINING
# The role_id to role_name mapping
ROLE_NAMES = {WIZARD: 'Wizard', APPRENTICE: 'Apprentice'}

# The keys to get agent qualification data from opt.
SAVED_DATA_WORKER_KEY = "worker"
SAVED_DATA_IS_WIZARD_KEY = "is_wizard"
SAVED_DATA_ROLE_QUALIFICATION_DATA_KEY = "qualification_dict"
ROLE_QUALIFICATION_NAME_KEY = "role_qname"

# OnBoardingSteps
# Make sure these number are consistent with OnboardingSteps,
# as they are defined in the SidePane.jsx frontend file.
# TODO: combine the definitions into one source of truth.
ONBOARDING_STEPS = {
    "NOT_ONBOARDING": 0,
    "CHAT_INTERFACE": 1,
    "TRY_SEARCH": 2,
    "PERSONA_WIZARD": 3,
    "PERSONA_APPRENTICE": 4,
    "WAITING": 10,
}

# Name of (bot)agents involved in the task world
ONBOARDING_AGENT = "OnboardingBot"
PERSONA_AGENT = "PersonaAgent"
SEARCH_AGENT = "SearchAgent"
COORDINATOR_AGENT = "Coordinator"


# The wait time in seconds to allow the agents read the instructions during the onboarding.
# After this, we allow them to continue after a small action (for example, type anything).
# The keys are the onboarding tutorial step; values are the wait times corresponding to that.
TUTORIAL_WAIT_TIMES = {"chat-interface": 1, "persona": 2, "knowledge": 2}

# Constants for checking onboarding work quality
WORKER_REJECT_REASON = "reason_to_reject"
MIN_AVG_CHAR_LENGTH_UTTERANCES = 10
MIN_AVG_WORD_LENGTH_UTTERANCES = 5
MIN_NUM_SEARCH_ONBOARDING = 2
MIN_NUM_SELECTED_SENTENCES_ONBOARDING = 2

# Long messages
ONBOARDING_WELCOME = (
    "Welcome onboard!\n"
    "Here you will have an engaging, "
    "knowledgeable chat with another person. "
    "This is the chat interface you will be using.\n"
    "Our interactive tutorial introduces you to the main task. "
    "If you finish all the steps successfully, "
    "and in reasonable time, we redirect you to the main task.\n"
    "Please have a friendly chitchat pretending you live in a "
    "world unaffected by covid and recent controversial events."
)

FINISHED_ONBOARDING = (
    "Good job, you now know how this task works!\n"
    "You can check the task instructions on the left at any time "
    "during the task. Please wait while we pair "
    "you with another participant."
)
