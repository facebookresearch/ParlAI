#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.abs

import parlai.core.agents
import parlai.agents
from parlai.core.params import ParlaiParser

import os
import pkgutil
import importlib
import inspect
import io
import csv


"""
Extract the readme and CLI args for all of the standard agents.
"""


def _make_argparse_table(class_):
    """
    Build the reStructuredText table containing the args and descriptions.
    """
    readme = []
    parser = ParlaiParser(False, False)
    class_.add_cmdline_args(parser)
    # group by whatever ArgumentGroups there are
    for ag in parser._action_groups:
        actions = []
        # get options defined within only this group
        for action in ag._group_actions:
            if hasattr(action, 'hidden') and action.hidden:
                # some options are marked hidden
                continue
            action_strings = ",  ".join(f'``{a}``' for a in action.option_strings)
            description = []
            # start with the help message
            if action.help:
                h = action.help
                if not h[0].isupper():
                    h = h[0].upper() + h[1:]
                description += [h]
            # list choices if there are any
            if action.choices:
                description += [
                    "Choices: " + ", ".join(f'``{c}``' for c in action.choices) + "."
                ]
            # list default and recommended values.
            default_value = ""
            if action.default is not None:
                default_value += f"Default: ``{action.default}``.  "
            if hasattr(action, 'recommended') and action.recommended:
                default_value += f"Recommended: ``{action.recommended}``. "

            # special escape for a few args which use a literal newline as their default
            if default_value:
                default_value = default_value.replace("\n", "\\n")
                description.append(default_value)

            description = "\n".join(description)
            # escape for the fact that we're inserting this inside a table
            description = description.replace("\n", "\n   \n   ")
            actions.append((action_strings, description))

        if not actions:
            continue

        # render the table
        readme.append(f"```eval_rst\n")
        readme.append(f".. csv-table:: {ag.title}\n")
        readme.append(f'   :widths: 35, 65\n\n')
        cout = io.StringIO()
        csvw = csv.writer(cout, csv.unix_dialect, delimiter=",")
        for row in actions:
            cout.write("   ")
            csvw.writerow(row)
        readme.append(cout.getvalue())
        readme.append("```\n\n")
    return readme


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

    # try to import all of the agents and look for their classes
    root = os.path.join(parlai.agents.__path__[0], agent)
    submodules = pkgutil.iter_modules([root])
    for sm in submodules:
        # look in the main folder
        if not (sm.name == agent or sm.name == 'agents'):
            continue
        module_name = f'parlai.agents.{agent}.{sm.name}'
        module = importlib.import_module(module_name)
        for itemname in dir(module):
            # skip all private items
            if itemname.startswith('_'):
                continue
            item = getattr(module, itemname)
            # avoid catching TorchAgent/TorchRankerAgent/...
            if (
                inspect.isclass(item)
                and issubclass(item, parlai.core.agents.Agent)
                and hasattr(item, 'add_cmdline_args')
                and not inspect.isabstract(item)
            ):
                # gather all the options
                options = _make_argparse_table(item)
                if options:
                    # if there were no options, don't mention it
                    readme.append(f"## {itemname} Options\n\n")
                    readme += options

    return readme


def write_all_agents():
    """
    Write list of agents to fout.

    :param fout:
        file object to write to
    """
    os.makedirs('agent_refs', exist_ok=True)
    agents = [name for _, name, _ in pkgutil.iter_modules(parlai.agents.__path__)]
    for agent in agents:
        with open(f'agent_refs/{agent}.md', 'w') as fout:
            fout.write(''.join(prepare_agent_readme(agent)))


if __name__ == '__main__':
    # Write the agents!
    write_all_agents()
