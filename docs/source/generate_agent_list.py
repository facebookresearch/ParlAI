#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.abs

import parlai.core.agents
import parlai.agents

import os
import pkgutil
import importlib
import inspect

"""
                    parser = ParlaiParser(False, False)
                    item.add_cmdline_args(parser)
                    readme.append(f"## {itemname} Options\n\n")
                    for ag in parser._action_groups:
                        actions = []
                        for action in ag._group_actions:
                            action_strings = ", ".join(action.option_strings)
                            help_ = action.help or ''
                            if action.choices:
                                type_ = (
                                    "{"
                                    + ", ".join(str(c) for c in action.choices)
                                    + "}"
                                )
                            elif type(action.type) is str:
                                type_ = action.type
                            elif action.type is not None:
                                type_ = action.type.__name__
                            else:
                                type_ = "?"
                            actions.append((action_strings, type_, help_))
                        if not actions:
                            continue
                        readme.append(f"### {ag.title} Arguments\n\n")
                        actions.insert(0, ("Options", "Choices", "Help"))
                        widths = [max(len(z) for z in item) for item in zip(*actions)]
                        formatters = ["{:<%ds}" % w for w in widths]
                        for i, action in enumerate(actions):
                            formatted = (
                                f.format(s) for f, s in zip(formatters, action)
                            )
                            readme.append("| " + " | ".join(formatted) + " |\n")
                            if i == 0:
                                readme.append(
                                    "| "
                                    + " | ".join("-" * (w + 0) for w in widths)
                                    + " |\n"
                                )

                        readme.append("\n\n")
"""


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

    root = os.path.join(parlai.agents.__path__[0], agent)
    submodules = pkgutil.iter_modules([root])
    argkeepers = []
    for sm in submodules:
        if not (sm.name == agent or sm.name == 'agents'):
            continue
        print(f"importing 'parlai.agents.{agent}.{sm.name}'")
        module_name = f'parlai.agents.{agent}.{sm.name}'
        module = importlib.import_module(module_name)
        for itemname in dir(module):
            if itemname.startswith('_'):
                continue
            item = getattr(module, itemname)
            if (
                inspect.isclass(item)
                and issubclass(item, parlai.core.agents.Agent)
                and hasattr(item, 'add_cmdline_args')
                and not inspect.isabstract(item)
            ):
                if itemname != 'BertClassifierAgent':
                    continue
                readme.append(f"## {itemname} CLI options\n")
                readme.append("```eval_rst\n.. argparse::\n")
                readme.append(f"   :filename: source/agent_refs/{agent}.py\n")
                readme.append(f"   :func: {itemname}\n")
                readme.append(f"   :prog: foo\n")

                #readme.append(f"   :passparser:\n")
                #readme.append(f"   :markdown:\n")
                readme.append("```\n\n")
                argkeepers.append((module_name, itemname))


    if argkeepers:
        with open(f'agent_refs/{agent}.py', 'w') as fout:
            fout.write("from parlai.core.params import ParlaiParser\n")
            for module_name, classname in argkeepers:
                fout.write(f"import {module_name} as {classname}_\n")
                fout.write(f"def {classname}():\n")
                fout.write(f"    pp = ParlaiParser(False, False)\n")
                fout.write(f"    {classname}_.{classname}.add_cmdline_args(pp)\n")
                fout.write(f"    return pp\n")
    with open(f'agent_refs/{agent}.md', 'w') as fout:
        fout.write(''.join(readme))


def write_all_agents():
    """
    Write list of agents to fout.

    :param fout:
        file object to write to
    """
    os.makedirs('agent_refs', exist_ok=True)
    agents = [name for _, name, _ in pkgutil.iter_modules([parlai.agents.__path__[0]])]
    for agent in agents:
        prepare_agent_readme(agent)


if __name__ == '__main__':
    # Write the agents!
    write_all_agents()
