#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Convenience file that lists all the agents directly.
"""

from parlai.core.tod.impl.agents_and_teachers import (
    ### Base class
    # All datasets should extend this class.
    TodStructuredDataParser as TodStructuredDataParser,
    ### Grounding Goal + Api Schema classes. These should be use together.
    # Empty (Empty API Schema Agent should be used for NO Schema models)
    TodEmptyApiSchemaAgent as TodEmptyApiSchemaAgent,
    TodEmptyGoalAgent as TodEmptyGoalAgent,
    # All API calls (might have multiple)
    TodGoalAgent as TodGoalAgent,
    TodApiSchemaAgent as TodApiSchemaAgent,
    # Single API Calls per episode
    TodSingleGoalAgent as TodSingleGoalAgent,
    TodSingleApiSchemaAgent as TodSingleApiSchemaAgent,
    ### Teachers for training a model
    SystemTeacher as SystemTeacher,
    UserSimulatorTeacher as UserSimulatorTeacher,
    ### Clases for working with API Implementation
    TodStandaloneApiTeacher as TodStandaloneApiTeacher,
    TodStandaloneApiAgent as TodStandaloneApiAgent,
    ### Classes for dumping out gold data,
    TodUserUttAgent as TodUserUttAgent,
    TodApiCallAndSysUttAgent as TodApiCallAndSysUttAgent,
    TodApiResponseAgent as TodApiResponseAgent,
)
