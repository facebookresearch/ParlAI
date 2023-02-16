#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the allowed worker filtering.
"""

import unittest
import parlai.utils.testing as testing_utils

try:
    from parlai.crowdsourcing.tasks.model_chat.impl import allow_list_filter
    from mephisto.abstractions.providers.mock.mock_worker import MockWorker
    from mephisto.abstractions.providers.mock.provider_type import (
        PROVIDER_TYPE as MOCK_PROVIDER_TYPE,
    )
except ImportError:
    # Tests will be skipped as well because of the @testing_utils.skipUnlessMephisto decorator.
    pass


ALLOW_LIST_QUALIFICATION_NAME = 'test-allow-modelchat-qualification'


class MockQualObject:
    def __init__(self) -> None:
        self.db_id = 0


class MockMephistoDB:
    def __init__(self) -> None:
        self.qualifications = dict()

    def optimized_load(self, *args, **kwargs):
        return kwargs.get("row")

    def cache_result(self, cls, loaded_val):
        pass

    def new_worker(self, worker_name, worker_type):
        self.worker_name = worker_name
        self.worker_id = 1
        return self.worker_id

    def get_worker(self, worker_id: str):
        return {
            'worker_id': worker_id,
            'provider_type': 'mock',
            'worker_name': f'UNITTEST_MOCK_{worker_id}',
        }

    def get_datastore_for_provider(self, provider):
        pass

    def find_qualifications(self, *args):
        return [MockQualObject()]

    def check_granted_qualifications(self, qualification_id, worker_id):
        return self.qualifications.get(worker_id, {}).get(qualification_id, [])

    def grant_qualification(self, qualification_id, worker_id, value):
        self.qualifications[worker_id] = {qualification_id: [value]}


@testing_utils.skipUnlessMephisto
class TestAllowList(unittest.TestCase):
    def test_user_allowed(self):
        db = MockMephistoDB()

        # Setting up the allowed worker
        worker_id = db.new_worker("MOCK_TEST_WORKER", MOCK_PROVIDER_TYPE)
        worker = MockWorker.get(db, worker_id)
        worker.grant_qualification(ALLOW_LIST_QUALIFICATION_NAME, skip_crowd=True)

        # Running the main test
        self.assertTrue(allow_list_filter(ALLOW_LIST_QUALIFICATION_NAME)(worker))

    def test_user_not_allowed(self):
        db = MockMephistoDB()

        # Setting up the allowed worker
        worker_id = db.new_worker("MOCK_TEST_WORKER", MOCK_PROVIDER_TYPE)
        worker = MockWorker.get(db, worker_id)

        # Running the main test
        self.assertFalse(allow_list_filter(ALLOW_LIST_QUALIFICATION_NAME)(worker))
