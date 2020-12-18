#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from parlai.crowdsourcing.tasks.model_chat.worlds import ModelChatWorld, get_bot_worker


class ModelImageChatWorld(ModelChatWorld):
    """
    A chat world in which an image is shown to the worker and bot at the beginning.
    """

    def __init__(self, opt, agent, bot, context_info: dict, image_idx: int):
        super().__init__(opt, agent=agent, bot=bot, context_info=context_info)

        self.image_stack = opt['image_stack']
        self.image_idx = image_idx

        # {{{TODO}}}

    def get_final_chat_data(self) -> Dict[str, Any]:
        """
        Add image-specific fields to the final chat data.
        """
        data = super().get_final_chat_data()
        data['image_idx'] = self.image_idx
        return data

    def shutdown(self):

        if not self.chat_done:
            # If the HIT was not completed, remove this worker from the stack
            worker = self.agents[0].mephisto_agent.get_worker().db_id
            self.image_stack.remove_worker_from_stack(
                worker=worker, stack_idx=self.image_idx
            )

        self.agent.shutdown()


def make_world(opt, agents):

    agents[0].agent_id = "Worker"

    # We are showing an image to the worker and bot, so grab the image path and other
    # context info
    image_idx, context_info, model_name, no_more_work = opt[
        'image_stack'
    ].get_next_image(agents[0].mephisto_agent.get_worker().db_id)
    if no_more_work:
        # There are no more HITs for this worker to do, so give them a qualification
        agents[0].mephisto_agent.get_worker().grant_qualification(
            qualification_name=opt['block_qualification'], value=1
        )

    # Get a bot agent
    bot_worker = get_bot_worker(opt=opt, model_name=model_name)

    return ModelImageChatWorld(
        opt=opt,
        agent=agents[0],
        bot=bot_worker,
        context_info=context_info,
        image_idx=image_idx,
    )


def get_world_params():
    return {"agent_count": 1}
