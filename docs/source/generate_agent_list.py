#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.abs

import parlai.agents

import os
import pkgutil


def prepare_agent_readme(agent):
    """
    Load agent readme, add title if necessary.

    :param agent:
        string indicating agent module

    :return:
        agent's readme
    """
    readme_path = f'{os.path.join(parlai.agents.__path__[0], agent)}/README.md'
    if not os.path.exists(readme_path):
        raise RuntimeError(f'Agent {agent} must have README.md')
    with open(readme_path) as f:
        readme = f.readlines()

    if '# ' not in readme[0]:
        readme[0] = f'# {agent}'

    return ''.join(readme)


def write_all_agents():
    """
    Write list of agents to fout.

    :param fout:
        file object to write to
    """
    os.makedirs('agent_refs', exist_ok=True)
    agents = [
        name
        for _, name, _ in pkgutil.iter_modules(
            [os.path.dirname(parlai.agents.__file__)]
        )
    ]
    for agent in agents:
        with open(f'agent_refs/{agent}.md', 'w') as fout:
            fout.write(prepare_agent_readme(agent))


if __name__ == '__main__':
    # Write the agents!
    write_all_agents()
