#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from mephisto.operations.registry import register_mephisto_abstraction
from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    ParlAIChatBlueprintArgs,
    ParlAIChatBlueprint,
)
from mephisto.abstractions.blueprint import SharedTaskState
from omegaconf import DictConfig, MISSING

WIZARD_INTERNET_PARLAICHAT_BLUEPRINT = 'wizard_internet_parlaichat_blueprint'


@dataclass
class WizardOfInternetBlueprintArgs(ParlAIChatBlueprintArgs):
    _blueprint_type: str = WIZARD_INTERNET_PARLAICHAT_BLUEPRINT
    _group: str = field(
        default='WizardInternetParlAIChatBlueprint',
        metadata={
            'help': """
                ParlAI chat between two agents with one agent having access to a search API
                that retrieves data from internet (common crawl snapshot). In order to run,
                the search API needs to be up and running.
            """
        },
    )

    role_qualification: str = field(
        default=MISSING,
        metadata={
            'help': """
                Specify the role (wizard or apprentice) that agents are trained on
                during their onboarding.
            """
        },
    )

    min_turns: int = field(
        default=MISSING,
        metadata={
            'help': """
                The minimum number of turns before showing the finish button on chat interface
                and allowing the agents to end the conversations cleanly.
            """
        },
    )

    wizard_time_out: int = field(
        default=180,
        metadata={'help': 'Maximum allowed time (seconds) for Wizard, each round.'},
    )

    apprentice_time_out: int = field(
        default=60,
        metadata={'help': 'Maximum allowed time (seconds) for Apprentice, each round.'},
    )

    personas_file: str = field(
        default=MISSING,
        metadata={'help': 'Path to a text file that keeps a list of curated personas.'},
    )

    persona_counts_file: str = field(
        default=MISSING,
        metadata={
            'help': 'A semicolon seperated list of personas and their count (file)'
        },
    )

    shuffle_persona: bool = field(
        default=True, metadata={'help': 'Whether to shuffle the persona list'}
    )

    use_personas_with_replacement: bool = field(
        default=False,
        metadata={'help': 'Using true does not discard personas after use.'},
    )

    banned_words_file: str = field(
        default=MISSING,
        metadata={
            'help': """
                Path to a text file with a list of banned words to block in the UI.
                Each row in the file is one word/phrase.
                User will receieve an alert and are asked to rephrase, if there is an exact match.
            """
        },
    )

    max_times_persona_use: int = field(
        default=0,
        metadata={
            'help': """
                Maximum number of times to allow a particular persona to be used.
                Default (0) mean no limit.
            """
        },
    )

    locations_file: str = field(
        default=MISSING,
        metadata={
            'help': """
                Path to a text file that keeps a list of locations that will be added
                to some of the curated personas (marked for needing persona).
            """
        },
    )

    search_server: str = field(
        default=MISSING, metadata={'help': 'Address to the search API.'}
    )

    num_passages_retrieved: int = field(
        default=5,
        metadata={'help': 'The number of documents to request from search API.'},
    )

    search_warning_turn: int = field(
        default=2,
        metadata={
            'help': 'The round that wizard may receive warning for using more search.'
        },
    )

    search_warning_threshold: int = field(
        default=2,
        metadata={
            'help': """
            The minimum number of times that wizard needs to use the search bar,
            at the rounds that we check for sending them a warning
            (warning is not send if wizard has used search more than this many times).
            """
        },
    )

    select_warning_turn: int = field(
        default=3,
        metadata={
            'help': 'The round that Wizard may receive warning to select more search results'
        },
    )

    select_warning_threshold: int = field(
        default=2,
        metadata={
            'help': """
            The minimum number of knowledge selections that wizard needs to have,
            at the round that we check for sending them a warning
            (warning is not send if Wizard has selected saerch results
            at least this many times so far).
            """
        },
    )


@register_mephisto_abstraction()
class WizardOfInternetBlueprint(ParlAIChatBlueprint):
    BLUEPRINT_TYPE = WIZARD_INTERNET_PARLAICHAT_BLUEPRINT
    ArgsClass = WizardOfInternetBlueprintArgs

    @classmethod
    def assert_task_args(
        cls, args: 'DictConfig', shared_state: 'SharedTaskState'
    ) -> None:
        """
        Ensure that arguments are properly configured to launch this task.
        """
        ParlAIChatBlueprint.assert_task_args(args=args, shared_state=shared_state)
        blueprint = args.get('blueprint')
        # Check search module is valid
        assert hasattr(blueprint, 'search_server'), 'Provide search API address.'

        assert hasattr(blueprint, 'use_personas_with_replacement')
        assert hasattr(shared_state, 'world_opt')
        assert 'personas' in shared_state.world_opt

        # Number of personas is enough for running without replacement
        if not blueprint.get('use_personas_with_replacement'):
            n_personas = len(shared_state.world_opt['personas'])
            n_conversations = blueprint.get('num_conversations')
            assert (
                n_personas >= n_conversations
            ), f'{n_personas} personas are not enought to use uniquely for {n_conversations} conversations.'

        # Make sure that we first show the warning for using search more often
        # to the wizard, and then the warning for selecting more sentences.
        assert blueprint.get('search_warning_turn') <= blueprint.get(
            'select_warning_turn'
        )
