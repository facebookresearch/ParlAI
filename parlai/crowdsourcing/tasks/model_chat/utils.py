#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.message import Message
from parlai.core.metrics import Metric


class Compatibility(object):
    """
    Class to address backward compatibility issues with older ParlAI models.
    """

    @staticmethod
    def backward_compatible_force_set(act, key, value):
        if isinstance(act, Message):
            act.force_set(key, value)
        elif isinstance(act, dict):
            act[key] = value
        else:
            raise Exception(f'Unknown type of act: {type(act)}')
        return act

    @staticmethod
    def maybe_fix_act(incompatible_act):
        if 'id' not in incompatible_act:
            new_act = Compatibility.backward_compatible_force_set(
                incompatible_act, 'id', 'NULL_ID'
            )
            return new_act
        return incompatible_act

    @staticmethod
    def serialize_bot_message(bot_message):
        if 'metrics' in bot_message:
            metric_report = bot_message['metrics']
            bot_message['metrics'] = {
                k: v.value() if isinstance(v, Metric) else v
                for k, v in metric_report.items()
            }
        return bot_message
