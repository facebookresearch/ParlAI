#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Wrappers for ParlAI models in the model zoo.

Available models include:
- blenderbot_90M
- blenderbot_400Mdistill
- blenderbot_1Bdistill
- blenderbot_3B
"""
from abc import ABC, abstractproperty

from parlai.core.agents import create_agent_from_model_file
from projects.safety_bench.utils.wrapper_loading import register_model_wrapper


class ParlAIModelZooWrapper(ABC):
    """
    Base class wrapper for ParlAI models in the ParlAI zoo.
    """

    def __init__(self):
        # Load the model from the model zoo via ParlAI
        overrides = {"skip_generation": False, "interactive_mode": True}
        self.model = create_agent_from_model_file(self.zoo_path, overrides)

    @abstractproperty
    def zoo_path(self):
        # Return the path to the agent in the model zoo
        pass

    def get_response(self, input_text: str) -> str:
        # In ParlAI, we use observe/act syntax to get a response from the model
        # Please see the ParlAI docs for more info
        self.model.observe({"text": input_text, "episode_done": True})
        response = self.model.act()

        return response.get("text")


@register_model_wrapper("blenderbot_90M")
class BlenderBot90MWrapper(ParlAIModelZooWrapper):
    @property
    def zoo_path(self):
        return "zoo:blender/blender_90M/model"


@register_model_wrapper("blenderbot_400Mdistill")
class BlenderBot400MDistillWrapper(ParlAIModelZooWrapper):
    @property
    def zoo_path(self):
        return "zoo:blender/blender_400Mdistill/model"


@register_model_wrapper("blenderbot_1Bdistill")
class BlenderBot1BDistillWrapper(ParlAIModelZooWrapper):
    @property
    def zoo_path(self):
        return "zoo:blender/blender_1Bdistill/model"


@register_model_wrapper("blenderbot_3B")
class BlenderBot3BWrapper(ParlAIModelZooWrapper):
    @property
    def zoo_path(self):
        return "zoo:blender/blender_3B/model"
