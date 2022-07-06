#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Wrappers for GPT models from HF (in ParlAI).

Available models include:
- GPT2 large
- DialoGPT medium
"""
from abc import ABC, abstractproperty
from typing import Dict

from parlai.core.agents import create_agent
from projects.safety_bench.utils.wrapper_loading import register_model_wrapper


class GPTWrapper(ABC):
    """
    Base class wrapper for GPT wrapper.
    """

    def __init__(self):
        # Load the model from the model zoo via ParlAI
        opt = {
            "skip_generation": False,
            "interactive_mode": True,
            "model": f"hugging_face/{self.model_name}",
            "gpt2_size": self.model_size,
            "add_special_tokens": False,
        }
        opt.update(self.additional_opts)
        self.model = create_agent(opt)

    @abstractproperty
    def model_name(self) -> str:
        # Return the path to the agent in the model zoo
        return ""

    @abstractproperty
    def model_size(self) -> str:
        # Return the requested model size
        return ""

    @abstractproperty
    def additional_opts(self) -> Dict:
        # Return any model specific opts
        return {}

    def get_response(self, input_text: str) -> str:
        # In ParlAI, we use observe/act syntax to get a response from the model
        # Please see the ParlAI docs for more info
        self.model.observe({"text": input_text, "episode_done": True})
        response = self.model.act()

        return response.get("text")


@register_model_wrapper("dialogpt_medium")
class DialoGPTMediumWrapper(GPTWrapper):
    @property
    def model_name(self):
        return "dialogpt"

    @property
    def model_size(self):
        return "medium"

    @property
    def additional_opts(self):
        return {
            "beam_context_block_ngram": 3,
            "beam_block_ngram": 3,
            "beam_size": 10,
            "inference": "beam",
            "beam_min_length": 20,
            "beam_block_full_context": False,
        }


@register_model_wrapper("gpt2_large")
class GPT2LargeWrapper(GPTWrapper):
    @property
    def model_name(self):
        return "gpt2"

    @property
    def model_size(self):
        return "large"

    @property
    def additional_opts(self):
        return {
            "beam_context_block_ngram": 3,
            "beam_block_ngram": 3,
            "beam_size": 10,
            "inference": "beam",
            "beam_min_length": 20,
            "beam_block_full_context": False,
        }

    def get_response(self, input_text: str) -> str:
        # For GPT-2, we add punctuation and an extra newline if one does
        # not exist, and then take the first line generated

        if input_text.strip()[-1] not in ['.', '?', '!']:
            input_text += "."

        self.model.observe({"text": input_text + "\n", "episode_done": True})
        response = self.model.act()
        # split on newline
        response_texts = response.get("text").split("\n")
        for response_text in response_texts:
            if response_text:
                # return first non-empty string
                return response_text

        # produced only newlines or empty strings
        return ""
