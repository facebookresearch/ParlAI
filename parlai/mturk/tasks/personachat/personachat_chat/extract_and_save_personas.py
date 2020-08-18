#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import os
from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task


def extract_and_save(opt):
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    teacher = world.agents[0]

    personas_path = opt.get('personas_path')
    if not os.path.exists(personas_path):
        os.makedirs(personas_path)

    new_episode = True
    personas = []
    while not teacher.epoch_done():
        act = teacher.act()
        if new_episode:
            persona_text = act['text'].split('\n')[:-1]
            if opt.get('persona_type') == 'both':
                persona_1 = [p for p in persona_text if 'your persona:' in p]
                persona_2 = [p for p in persona_text if 'partner\'s persona:' in p]
                persona_1 = [p[p.find(':') + 1 :] for p in persona_1]
                persona_2 = [p[p.find(':') + 1 :] for p in persona_2]
                personas += [persona_1, persona_2]
            else:
                persona = [p for p in persona_text if 'persona:' in p]
                persona = [p[p.find(':') + 1 :] for p in persona]
                personas.append(persona)
            new_episode = act.get('episode_done')
        else:
            new_episode = act.get('episode_done')

    for idx, persona in enumerate(personas):
        with open('{}/{}.pkl'.format(personas_path, idx), 'wb') as f:
            pickle.dump(persona, f)
    print('---Finished extracting and saving personas, to {}'.format(personas_path))


def main(opt):
    print('---Extracting and saving personas---')
    teacher_name = 'personachat:{}'.format(opt.get('persona_type'))
    teacher_name += 'Revised' if opt.get('revised') else 'Original'
    opt['task'] = teacher_name
    assert 'personas_path' in opt, 'Must specify personas path'
    opt['datatype'] = 'train:ordered:stream'
    opt['batchsize'] = 1
    extract_and_save(opt)


if __name__ == '__main__':
    parser = ParlaiParser()
    parser.add_argument(
        '--persona-type',
        default='both',
        type=str,
        choices=['both', 'self', 'other'],
        help='Which personas to load from personachat',
    )
    parser.add_argument(
        '--revised', default=False, type='bool', help='Whether to use revised personas'
    )
    opt = parser.parse_args()
