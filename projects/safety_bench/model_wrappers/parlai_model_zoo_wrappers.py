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


@register_model_wrapper("seeker_dialogue")
class SeekerDialogue3BWrapper(ParlAIModelZooWrapper):
    def __init__(self):
        # Load the model from the model zoo via ParlAI
        overrides = {
            "skip_generation": False,
            "interactive_mode": True,
            "init_opt": "gen/seeker_dialogue",
            "all_model_path": "zoo:seeker/seeker_dialogue_3B/model",
            # seeker_dialogue
            "beam_disregard_knowledge_for_context_blocking": False,
            "drm_beam_block_full_context": True,
            "drm_beam_block_ngram": 3,
            "drm_beam_context_block_ngram": 3,
            "drm_beam_min_length": 20,
            "drm_beam_size": 10,
            "drm_inference": "beam",
            "drm_message_mutators": None,
            "drm_model": "projects.seeker.agents.seeker:ComboFidSearchQueryAgent",
            "exclude_context_in_krm_context_blocking": False,
            "include_knowledge_in_krm_context_blocking": True,
            "inject_query_string": None,
            "knowledge_response_control_token": None,
            "krm_beam_block_ngram": 3,
            "krm_beam_context_block_ngram": 3,
            "krm_beam_min_length": 1,
            "krm_beam_size": 3,
            "krm_doc_chunks_ranker": "woi_chunk_retrieved_docs",
            "krm_inference": "beam",
            "krm_message_mutators": None,
            "krm_model": "projects.seeker.agents.seeker:ComboFidSearchQueryAgent",
            "krm_n_ranked_doc_chunks": 1,
            "krm_rag_retriever_type": "search_engine",
            "krm_search_query_generator_model_file": "''",
            "loglevel": "debug",
            "min_knowledge_length_when_search": 10,
            "model": "projects.seeker.agents.seeker:SeekerAgent",
            "model_file": "zoo:seeker/seeker_dialogue_3B/model",
            "sdm_beam_block_ngram": -1,
            "sdm_beam_min_length": 1,
            "sdm_beam_size": 1,
            "sdm_history_size": 1,
            "sdm_inference": "greedy",
            "sdm_model": "projects.seeker.agents.seeker:ComboFidSearchQueryAgent",
            "search_decision": "always",
            "search_decision_control_token": "__is-search-required__",
            "search_decision_do_search_reply": "__do-search__",
            "search_decision_dont_search_reply": "__do-not-search__",
            "search_query_control_token": "__generate-query__",
            "sqm_beam_block_ngram": -1,
            "sqm_beam_min_length": 2,
            "sqm_beam_size": 1,
            "sqm_inference": "beam",
            "sqm_model": "projects.seeker.agents.seeker:ComboFidSearchQueryAgent",
        }
        self.model = create_agent_from_model_file(self.zoo_path, overrides)

    @property
    def zoo_path(self):
        return "zoo:seeker/seeker_dialogue_3B/model"
