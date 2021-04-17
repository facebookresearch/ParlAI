#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import textwrap
import argparse
from parlai.core.params import ParlaiParser
from parlai.core.mutators import setup_mutator_registry
from parlai.utils.testing import capture_output
import parlai.utils.logging as logging

from parlai.scripts.display_data import DisplayData

TASK = 'convai2:sample'


def _make_argparse_table(class_):
    """
    Build the reStructuredText table containing the args and descriptions.
    """
    readme = []
    parser = ParlaiParser(False, False)
    class_.add_cmdline_args(parser, partial_opt=None)
    # group by whatever ArgumentGroups there are
    for ag in parser._action_groups:
        actions = []
        # get options defined within only this group
        for action in ag._group_actions:
            if hasattr(action, 'hidden') and action.hidden:
                # some options are marked hidden
                continue
            if action.dest == argparse.SUPPRESS or action.dest == 'help':
                continue
            action_strings = ",  ".join(f'`{a}`' for a in action.option_strings)
            description = []
            if action.help:
                h = action.help
                if not h[0].isupper():
                    h = h[0].upper() + h[1:]
                h = h.replace("%(default)s", str(action.default))
                description += [h]
            # list choices if there are any
            if action.choices:
                description += [
                    "Choices: " + ", ".join(f'`{c}`' for c in action.choices) + "."
                ]
            # list default and recommended values.
            default_value = ""
            if action.default is not None and action.default is not argparse.SUPPRESS:
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

        readme.append(f'__{ag.title.title()}__\n\n')
        readme.append("| Argument | Description |\n")
        readme.append("|----------|----------|\n")
        for row in actions:
            text = "| " + " | ".join(row) + " |"
            text = text.replace("\n", "<br>")
            readme.append(f"{text}\n")
        readme.append("\n\n")
    return readme


logging.disable()

mutators = setup_mutator_registry()


def _display_data(**kwargs):
    with capture_output() as output:
        DisplayData.main(**kwargs)
    return output.getvalue()


with open('mutators_list.inc', 'w') as fout:
    output = _display_data(task=TASK)
    fout.write("## Original output\n\n")
    fout.write("We show the unmutated output of the examples for reference:\n\n")
    fout.write(f"```\n{output}\n```\n")

    for mutator_name in sorted(mutators.keys()):
        mutator = mutators[mutator_name]
        options = _make_argparse_table(mutator)
        if not mutator.__doc__:
            continue
        fout.write('\n------------\n\n')
        fout.write(f'## {mutator_name}\n\n')
        fout.write(textwrap.dedent(mutator.__doc__).strip() + '\n\n')
        fout.write(
            f'**Example usage**:\n\n`parlai display_data -t {TASK} '
            f'--mutators {mutator_name}`\n\n'
        )
        fout.write("**Example output**:\n\n")
        fout.write("```\n")
        output = _display_data(task=TASK, mutators=mutator_name)
        fout.write(output)
        fout.write("\n```\n")
        if options:
            fout.write("".join(options) + '\n\n')
