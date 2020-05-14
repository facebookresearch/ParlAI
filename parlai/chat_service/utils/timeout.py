#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import List, Optional
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.worlds import World

DEFAULT_TIMEOUT = 120
THREAD_SLEEP = 0.1


class TimeoutUtils(object):
    """
    Provide interface for getting an agent's act with timeout.
    """

    @staticmethod
    def get_timeout_act(
        agent: Agent,
        timeout: int = DEFAULT_TIMEOUT,
        quick_replies: Optional[List[str]] = None,
    ) -> Optional[Message]:
        """
        Return an agent's act, with a specified timeout.

        :param agent:
            Agent who is acting
        :param timeout:
            how long to wait
        :param quick_replies:
            If given, agent's message *MUST* be one of the quick replies

        :return:
            An act dictionary if no timeout; else, None
        """

        def _is_valid(act):
            return act.get("text", "") in quick_replies if quick_replies else True

        act = None
        curr_time = time.time()
        allowed_timeout = timeout
        while act is None and time.time() - curr_time < allowed_timeout:
            act = agent.act()
            if act is not None and not _is_valid(act):
                agent.observe(
                    {
                        "id": "",
                        "text": "Invalid response. Please choose one of the quick replies",
                        "quick_replies": quick_replies,
                    }
                )
            time.sleep(THREAD_SLEEP)  # TODO: use threading.Event() rather than polling
        return act

    @staticmethod
    def _get_response_timeout_loop(
        agent: Agent,
        world: World,
        timeout: int = DEFAULT_TIMEOUT,
        timeout_msg: str = 'You have timed out',
    ) -> Optional[Message]:
        """
        Get a response from the agent.

        :param agent:
            agent who is acting
        :param world:
            world in which agent is acting
        :param timeout:
            timeout in secs
        :param timeout_msg:
            what to say to agent when they timeout

        :return response:
            Response if given, else None
        """
        a = TimeoutUtils.get_timeout_act(agent, timeout)
        if a is None:
            world.episodeDone = True  # type: ignore
            agent.observe({"id": "", "text": timeout_msg})
            return None

        if (a.get("text", "") or "").upper() == "EXIT":
            world.episodeDone = True  # type: ignore
            return None
        return a
