#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict, List
from parlai.crowdsourcing.utils.analysis import AbstractResultsCompiler
from collections import defaultdict

try:
    # First, trying to test with a class derived from the Mephisto's own Unit object.
    # This MockUnit will cover testing some of the functionalities derived from its parent (Unit).
    from mephisto.data_model.unit import Unit

    class MockUnit(Unit):
        pass

except ModuleNotFoundError:
    # In case Mephisto is not installed we use a simpler mock object.
    class MockUnit:
        def __init__(self, *args, **kwargs) -> None:
            pass


######################################################################
#
#           Mock version of Mephisto abstarctions
#
######################################################################


parser_ = AbstractResultsCompiler.setup_args()
opt_string = '--task-name mock_task_name --output-folder /dummy/tmp'
args_ = parser_.parse_args(opt_string.split())
DUMMY_OPT = vars(args_)
LEN_MOCK_DATA = 5


class MockMephistoDB:
    def optimized_load(self, *args, **kwargs):
        return kwargs.get("row")

    def cache_result(self, cls, loaded_val):
        pass

    def get_worker_name(self, worker_id: str):
        return {'worker_name': f'UNITTEST_MOCK_{worker_id}'}


class MockMephistoBrowser:
    def __init__(self) -> None:
        self._data = dict()
        db = MockMephistoDB()
        for idx in range(LEN_MOCK_DATA):
            unit = MockUnit.get(db, "mock_db", defaultdict(int))
            self._data[unit] = {
                'unit_id': idx * 10,
                'worker_id': idx,
                'worker_name': db.get_worker_name(idx),
                'data': {
                    'dialogue': [
                        f'mock dialogue round {i} for unit {idx}' for i in range(4)
                    ]
                },
            }

    def get_units_for_task_name(self, taske_name: str) -> List[MockUnit]:
        return self._data.keys()

    def get_data_from_unit(self, get_data_from_unit: MockUnit) -> Dict[str, Any]:
        return self._data[get_data_from_unit]


######################################################################
#
#           Unit tests
#
######################################################################


class MockResultsCompiler(AbstractResultsCompiler):
    """
    A mock result compiler used for testing.
    """

    def __init__(self, opt):
        super().__init__(opt)
        # Injecting mock dependancies
        self._mephisto_db = MockMephistoDB()
        self._mephisto_data_browser = MockMephistoBrowser()

    def compile_results(self):
        compiled_results = dict()
        for tdata in self.get_task_data():
            tdata_id = tdata['unit_id']
            compiled_results[tdata_id] = tdata
        return compiled_results


class MockResultsCompilerWithFilter(MockResultsCompiler):
    """
    A mock result compiler with unit filtering, based on worker id.
    """

    def is_unit_acceptable(self, unit_data):
        REJECTED_WORKER_ID = 2
        return unit_data['worker_id'] != REJECTED_WORKER_ID


class TestResultsCompiler(unittest.TestCase):
    """
    Test the base ResultsCompiler class.
    """

    def test_compile_all_results(self):
        data_compiler = MockResultsCompiler(DUMMY_OPT)

        compiled_data = data_compiler.compile_results()
        self.assertIsNotNone(compiled_data)
        self.assertIsInstance(compiled_data, dict)
        self.assertEqual(len(compiled_data), LEN_MOCK_DATA)

    def test_compile_results_with_eliminated_workers(self):
        data_compiler = MockResultsCompilerWithFilter(DUMMY_OPT)

        compiled_data = data_compiler.compile_results()
        self.assertIsNotNone(compiled_data)
        self.assertIsInstance(compiled_data, dict)
        self.assertEqual(len(compiled_data), LEN_MOCK_DATA - 1)


if __name__ == "__main__":
    unittest.main()
