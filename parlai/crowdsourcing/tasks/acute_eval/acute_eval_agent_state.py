#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Any, TYPE_CHECKING
from mephisto.server.blueprints.abstract.static_task.static_agent_state import (
    StaticAgentState,
)
import time

if TYPE_CHECKING:
    from mephisto.data_model.packet import Packet


DATA_FILE = "agent_data.json"


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

    def update_data(self, packet: "Packet") -> None:
        """
        Process the incoming data packet, and handle updating the state.
        """
        assert (
            packet.data.get("MEPHISTO_is_submit") is True
        ), "Static tasks should only have final act"
        self.state["times"]["task_end"] = time.time()
        self.state["outputs"] = packet.data["task_data"]
        self.save_data()
