#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Any
from mephisto.abstractions.blueprints.abstract.static_task.static_agent_state import (
    StaticAgentState,
)


class AcuteEvalAgentState(StaticAgentState):
    """
    Agent state for acute eval tasks.

    Equivalent to StaticAgentState but doesn't have file IO.
    """

    def get_parsed_data(self) -> List[Dict[str, Any]]:
        data = self.get_data()
        assert data is not None, "Should only check parsed data for completed tasks"
        response_list = []
        inputs: List[Dict[str, Any]] = data["inputs"]
        outputs = data["outputs"]
        assert inputs is not None
        assert outputs is not None
        for idx in range(len(inputs)):
            entry: Dict[str, Any] = {}
            entry.update(inputs[idx])
            entry.update(outputs["final_data"][idx])
            response_list.append(entry)
        return response_list
