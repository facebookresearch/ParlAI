#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple

from parlai.core.image_featurizers import ImageLoader
from parlai.core.message import Message
from parlai.core.worlds import validate
from parlai.crowdsourcing.tasks.model_chat.utils import Compatibility, get_image_src
from parlai.crowdsourcing.tasks.model_chat.worlds import (
    BaseModelChatWorld,
    get_bot_worker,
)


class ModelImageChatWorld(BaseModelChatWorld):
    """
    A chat world in which an image is shown to the worker and bot at the beginning.
    """

    def __init__(self, opt, agent, bot, image_idx: int, image_act: Message):
        super().__init__(opt, agent=agent, bot=bot)

        self.image_stack = opt['image_stack']
        self.image_idx = image_idx
        self.image_act = image_act

        # Get a stringified version of the image to show the user
        orig_image = self.image_act['image']
        self.image_src = get_image_src(image=orig_image)

        # Get a featurized version of the image to show the bot
        with NamedTemporaryFile(suffix='.jpg') as f:
            orig_image.save(f)
            image_loader = ImageLoader(self.bot.model_agent.opt)
            self.image_act.force_set('image', image_loader.load(f.name))

    def _run_initial_turn(self) -> None:
        """
        Show the image to the human and bot, and show the bot's response to the human.
        """

        system_id = 'SYSTEM'
        system_agent_idx = None

        # Show the image to the human
        image_act_for_human = {
            'episode_done': False,
            'id': system_id,
            'text': f"""Welcome! You'll now have a conversation with your partner.

<-- FIRST, YOUR PARTNER WILL SAY SOMETHING ABOUT THIS IMAGE TO YOUR LEFT.

Be sure to talk about this image a little bit before discussing other things!
""",
            'task_data': {'image_src': self.image_src},
            'agent_idx': system_agent_idx,
        }
        self.agent.observe(validate(image_act_for_human))

        # Show the image to the bot
        image_act = {
            **self.image_act,
            'episode_done': False,
            'id': system_id,
            'agent_idx': system_agent_idx,
        }
        self.bot.observe(validate(image_act))
        del image_act['image']
        # Don't save the image features to disk

        # Have the bot respond
        bot_first_act_raw = self.bot.act()
        bot_first_act_raw = Message(
            Compatibility.maybe_fix_act(bot_first_act_raw)
        ).json_safe_payload()
        bot_first_act_raw['id'] = self.bot.agent_id
        self.agent.observe(validate(bot_first_act_raw))
        bot_first_act = {
            'episode_done': False,
            'id': bot_first_act_raw['id'],
            'text': bot_first_act_raw['text'],
            'agent_idx': 1,
        }

        # Record lines of dialogue
        self.dialog.append(image_act)
        self.dialog.append(bot_first_act)

    def _postprocess_acts(self, acts: List[dict], agent_idx: int):
        """
        Show the bot the image again on every turn.
        """
        if agent_idx == 0:
            # Add the image to every human act, seen by the bot. Also adds in any other
            # image-related fields needed by the model
            for key, value in self.image_act.items():
                if key not in ['episode_done', 'id', 'text', 'agent_idx']:
                    acts[agent_idx][key] = value

    def get_final_chat_data(self) -> Dict[str, Any]:
        """
        Add image-specific fields to the final chat data.
        """
        data = super().get_final_chat_data()
        data['image_idx'] = self.image_idx
        return data

    def _prepare_acceptability_checking(self) -> Tuple[List[str], List[str]]:
        """
        Apply acceptability checking params specific to image-chat conversation.

        The conversation starts with an image, so the human shouldn't be starting their
        first message with "Hi", etc.
        """
        human_messages, violation_types = super()._prepare_acceptability_checking()
        violation_types.append('penalize_greetings')
        return human_messages, violation_types

    def shutdown(self):

        if not self.chat_done:
            # If the HIT was not completed, remove this worker from the stack
            worker = self.agent.mephisto_agent.get_worker().db_id
            self.image_stack.remove_worker_from_stack(
                worker=worker, stack_idx=self.image_idx
            )

        self.agent.shutdown()


def make_world(opt, agents):

    # We are showing an image to the worker and bot, so grab the image path and other
    # context info
    image_idx, model_name, no_more_work = opt['image_stack'].get_next_image(
        agents[0].mephisto_agent.get_worker().db_id
    )
    full_image_context = opt['image_contexts'][image_idx]
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
        image_idx=image_idx,
        image_act=full_image_context['image_act'],
    )


def get_world_params():
    return {"agent_count": 1}
