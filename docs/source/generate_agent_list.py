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

    return readme


def write_agent(fout, agent_readme):
    """
    Write agent README to file.

    Handle a few special cases:
    - Attempt to convert section headings in MD to RST
    - Attempt to convert MD bash blocks to RST bash blocks

    :param fout:
        file object
    :param agent_readme:
        string to write
    """
    i = 0
    while i < len(agent_readme):
        line = agent_readme[i]
        split = line.split()
        if not split:
            fout.write('\n')
            i += 1
            continue
        header = ''
        is_header = split[0].count('#') > 0
        if i == 0 and is_header:
            header = '*'  # title
        elif is_header:
            header = '-'
        if header:  # Format Headings
            title = ' '.join(split[1:])
            fout.write(f'{title}\n')
            fout.write(f'{header * len(title)}\n')
            i += 1
        elif '```bash' in line:  # Format Bash Commands
            line = '\n\n.. code-block:: bash\n'
            fout.write(line)
            agent_readme[i + 1] = f'\n  {agent_readme[i+1]}'
            i += 1
        elif '```' in line:
            fout.write('\n')
            i += 1
            continue
        else:
            fout.write(line)
            i += 1

    fout.write('\n\n')


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
        with open(f'agent_refs/{agent}.rst', 'w') as fout:
            write_agent(fout, prepare_agent_readme(agent))


# Write the agents!
write_all_agents()
