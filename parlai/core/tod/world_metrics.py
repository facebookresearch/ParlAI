#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Wrapper object holding metrics for TODWorld.

This class is in its own file to prevent circular dependencies + monolithic files.
"""

from parlai.core.message import Message
from parlai.core.metrics import Metrics
from parlai.core.tod.tod_core import (
    TodAgentType,
    TOD_AGENT_TYPE_TO_PREFIX,
    SerializationHelpers,
    STANDARD_GOAL,
)
from typing import Any, Dict
import parlai.core.tod.world_metrics_handlers as world_metrics_handlers

# Change the following to define which Metrics Handlers are used in TodWorld.
# The ones used below are from `world_metrics_handlers.py` only. However, See `parlai/projects/tod_simulator/world_metrics/extended_world_metrics.py` for others.

WORLD_METRIC_HANDLERS = [
    world_metrics_handlers.AllGoalApiCallSuccessMetricsHandler,
    world_metrics_handlers.UserGeneratedDoneMetricHandler,
]


class TodMetrics(Metrics):
    """
    Helper container which encapsulates TOD metrics and does some basic prepocessing to
    handlers to calculate said metrics.

    This class should generally not need to be changed; add new metrics handlers to
    `WORLD_METRIC_HANDLERS` (or otherwise override `self.handlers` of this class) to
    change metrics actively being used.
    """

    def __init__(self, shared: Dict[str, Any] = None) -> None:
        super().__init__(shared=shared)
        self.handlers = [x() for x in WORLD_METRIC_HANDLERS]
        self.convo_started = False
        self.last_episode_metrics = Metrics()

    def handle_message(self, message: Message, agent_type: TodAgentType):
        if "text" not in message:
            return
        if agent_type == TodAgentType.GOAL_GROUNDING_AGENT and len(
            message["text"]
        ) > len(STANDARD_GOAL):
            # Only count a conversation as started if there is a goal.
            self.convo_started = True
        for handler in self.handlers:
            metrics = self._handle_message_impl(message, agent_type, handler)
            if metrics is not None:
                for name, metric in metrics.items():
                    if metric is not None:
                        self.add(name, metric)

    def _handle_message_impl(
        self,
        message: Message,
        agent_type: TodAgentType,
        handler: world_metrics_handlers.TodMetricsHandler,
    ):
        prefix_stripped_text = message["text"].replace(
            TOD_AGENT_TYPE_TO_PREFIX[agent_type], ""
        )
        if agent_type is TodAgentType.API_SCHEMA_GROUNDING_AGENT:
            return handler.handle_api_schemas(
                message, SerializationHelpers.str_to_api_schemas(prefix_stripped_text)
            )
        if agent_type is TodAgentType.GOAL_GROUNDING_AGENT:
            return handler.handle_goals(
                message, SerializationHelpers.str_to_goals(prefix_stripped_text)
            )
        if agent_type is TodAgentType.USER_UTT_AGENT:
            return handler.handle_user_utt(message, prefix_stripped_text)
        if agent_type is TodAgentType.API_CALL_AGENT:
            return handler.handle_api_call(
                message, SerializationHelpers.str_to_api_dict(prefix_stripped_text)
            )
        if agent_type is TodAgentType.API_RESP_AGENT:
            return handler.handle_api_resp(
                message, SerializationHelpers.str_to_api_dict(prefix_stripped_text)
            )
        if agent_type is TodAgentType.SYSTEM_UTT_AGENT:
            return handler.handle_sys_utt(message, prefix_stripped_text)

    def get_last_episode_metrics(self):
        """
        This is a bit of a hack so that we can  report whether or not a convo has
        successfully hit all goals and associate this with each episode for the purposes
        of doing filtering.
        """
        return self.last_episode_metrics

    def episode_reset(self):
        self.last_episode_metrics = None
        if self.convo_started:
            self.last_episode_metrics = Metrics()
            for handler in self.handlers:
                metrics = handler.get_episode_metrics()
                handler.episode_reset()
                if metrics is not None:
                    for name, metric in metrics.items():
                        if metric is not None:
                            self.add(name, metric)
                            self.last_episode_metrics.add(name, metric)
            self.convo_started = False
