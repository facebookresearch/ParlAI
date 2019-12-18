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
    readme_path = f'../../parlai/agents/{agent}/README.md'
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

    :param fout:
        file object

    :param agent_readme:
        string to write
    """

    # pound_to_heading = {
    #     0: '',
    #     1: '#',
    #     2: '*',
    #     3: '=',
    #     4: '-',
    #     5: '^',
    #     6: '\"'
    # }

    pound_to_heading = {
        0: '',
        1: '*',
        2: '-',
        3: '-',
        4: '-',
        5: '-'
    }

    # pound_to_heading = {
    #     0: '',
    #     1: '*',
    #     2: '=',
    #     3: '-',
    #     4: '^',
    #     5: '\"'
    # }

    i = 0
    while i < len(agent_readme):
        line = agent_readme[i]
        split = line.split()
        if not split:
            fout.write('\n')
            i += 1
            continue
        header_count = split[0].count('#')
        if i != 0 and header_count > 0:
            header_count += 1  # Not a title
        header = pound_to_heading[header_count]
        if header:  # Format Headings
            title = ' '.join(split[1:])
            fout.write(f'{title}\n')
            fout.write(f'{header * len(title)}\n')
            i += 1
        elif '```bash' in line:  # Format Bash Commands
            # if '```\n' in agent_readme[i + 1:]:
            #     last_bash_line = agent_readme.index('```\n', i + 1)
            # elif '```' in agent_readme:
            #     last_bash_line = agent_readme.index('```', i + 1)
            # else:
            #     last_bash_line = i + 1
            # new_lines = agent_readme[i: last_bash_line + 1]
            # new_lines = [l.replace('```', '') for l in new_lines]
            # new_lines = ['.. code-block:: bash'] + [f"  {l.replace('```', '')}" for l in new_lines[1:]]
            # new_line = '\n\n'.join(new_lines)
            #
            # fout.write(f'{new_line}\n')
            # i += len(new_lines)
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


def write_all_agents(fout):
    """
    Write list of agents to fout.

    :param fout:
        file object to write to
    """
    agents = [
        name for _, name, _ in pkgutil.iter_modules(
            [os.path.dirname(parlai.agents.__file__)]
        )
    ]
    for agent in agents:
        write_agent(fout, prepare_agent_readme(agent))


# Write the agents!

with open('agent_refs.inc', 'w') as f:
    write_all_agents(f)
