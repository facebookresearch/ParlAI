#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Talk with a model using a Slack channel.

# Examples

```shell
parlai interactive_slack --model-file "zoo:tutorial_transformer_generator/model"
```
"""
from os import getenv
from pprint import pformat

from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging

try:
    from slack import RTMClient
except ImportError:
    raise ImportError(
        'The slackclient package must be installed to run this script')

SHARED = {}
SLACK_BOT_TOKEN = getenv('SLACK_BOT_TOKEN')


def setup_slack_args(shared):
    """
    Build and parse CLI opts.
    """
    parser = setup_args()
    parser.description = 'Interactive chat with a model in a Slack channel'
    parser.add_argument('--token', default=SLACK_BOT_TOKEN, metavar='SLACK_BOT_TOKEN',
                        help='A legacy Slack bot token to use for RTM messaging')
    return parser


@RTMClient.run_on(event='message')
async def rtm_handler(rtm_client, web_client, data, **kwargs):
    global SHARED
    if 'bot_profile' in data:
        # Dont respond to bot messages (eg the one this app generates)
        return
    logging.info(f'Got new message {pformat(data)}')
    channel = data['channel']
    web_client.reactions_add(
        channel=channel,
        name='eyes',
        timestamp=data['ts']
    )
    reply = {'episode_done': False, 'text': data['text']}
    SHARED['agent'].observe(reply)
    logging.info('Agent observed')
    await rtm_client.typing(channel=channel)
    model_response = SHARED['agent'].act()
    logging.info('Agent acted')
    web_client.chat_postMessage(
        channel=channel, text=model_response['text'])
    logging.info(f'Sent response: {model_response["text"]}')
    web_client.reactions_remove(
        channel=channel,
        name='eyes',
        timestamp=data['ts']
    )


def interactive_slack(opt):
    global SHARED

    if not opt.get('token'):
        raise RuntimeError(
            'A Slack bot token must be specified. Must be a legacy bot app token for RTM messaging')
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()
    agent.opt['verbose'] = True
    SHARED['opt'] = agent.opt
    SHARED['agent'] = agent
    SHARED['world'] = create_task(SHARED.get('opt'), SHARED['agent'])
    SHARED['client'] = client = RTMClient(token=opt['token'])
    logging.info('Slack client is starting')
    client.start()


@register_script('interactive_slack', aliases=['slack'], hidden=True)
class InteractiveSlack(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_slack_args(SHARED)

    def run(self):
        return interactive_slack(self.opt)


if __name__ == '__main__':
    InteractiveSlack.main()
