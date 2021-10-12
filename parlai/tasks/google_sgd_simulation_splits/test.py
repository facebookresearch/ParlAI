#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.testing import AutoTeacherTest


class TestInDomainSystemTeacher(AutoTeacherTest):
    task = "google_sgd_simulation_splits:InSystemTeacher"


class TestInDomainUserSimulatorTeacher(AutoTeacherTest):
    task = "google_sgd_simulation_splits:InUserSimulatorTeacher"


class TestOutDomainSystemTeacher(AutoTeacherTest):
    task = "google_sgd_simulation_splits:OutSystemTeacher"


class TestOutDomainUserSimulatorTeacher(AutoTeacherTest):
    task = "google_sgd_simulation_splits:OutUserSimulatorTeacher"
