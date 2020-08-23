#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import parlai.core.script as pcs


def render_script(fout, key, registration):
    script_parser = registration.klass.setup_args()
    description = script_parser.description
    underline = "#" * (5 + len(key))
    fout.write(f"{key}\n{underline}\n\n")
    if description:
        fout.write(f"**Short description:** {description}\n\n")
    if registration.aliases:
        aliases = ", ".join(f'``{a}``' for a in registration.aliases)
        fout.write(f"**Aliases:** {aliases}\n")

    fout.write(f'\n.. automodule:: {registration.klass.__module__}\n\n')

    actions = []
    for action in script_parser._actions:
        if hasattr(action, 'hidden') and action.hidden:
            # some options are marked hidden
            continue
        if action.dest == argparse.SUPPRESS:
            continue
        action_strings = ",  ".join(f'``{a}``' for a in action.option_strings)
        if not action_strings:
            continue
        description = []
        if action.help:
            h = action.help
            if not h[0].isupper():
                h = h[0].upper() + h[1:]
            h = h.replace("%(default)s", f'``{action.default}``')
            description += [h]
        # list choices if there are any
        if action.choices:
            description += [
                "Choices: " + ", ".join(f'``{c}``' for c in action.choices) + "."
            ]
        default_value = ""
        if action.default and action.default != argparse.SUPPRESS:
            default_value += f"Default: ``{action.default}``.  "
        if hasattr(action, 'recommended') and action.recommended:
            default_value += f"Recommended: ``{action.recommended}``. "

        # special escape for a few args which use a literal newline as their default
        if default_value:
            default_value = default_value.replace("\n", "\\n")
            description.append(default_value)

        # escape for the fact that we're inserting this inside a table
        if len(description) > 1:
            description = [' | ' + d for d in description]
        if len(description) == 0:
            description = [""]
        actions.append((action_strings, description))

    if not actions:
        return

    action_width = max(max(len(a) for a, d in actions), len("Argument"))
    desc_width = len("Description")
    for _, desc in actions:
        for line in desc:
            width = len(line)
            if width > desc_width:
                desc_width = width

    fout.write("CLI Arguments\n")
    fout.write("-------------\n")
    fout.write("+" + "-" * action_width + "+" + "-" * desc_width + "+\n")
    fstr = f'|{{:{action_width}s}}|{{:{desc_width}}}|\n'
    fout.write(fstr.format("Argument", "Description"))
    fout.write("+" + "=" * action_width + "+" + "=" * desc_width + "+\n")
    for action, description in actions:
        for i, line in enumerate(description):
            aval = action if i == 0 else ""
            fout.write(fstr.format(aval, line))
        fout.write("+" + "-" * action_width + "+" + "-" * desc_width + "+\n")
    fout.write('\n\n')


def main():
    fout = open('cli_usage.inc', 'w')
    pcs.setup_script_registry()

    first = []
    second = []
    for key, registration in sorted(pcs.SCRIPT_REGISTRY.items()):
        if not registration.hidden:
            first.append((key, registration))
        else:
            second.append((key, registration))

    for key, registration in first:
        render_script(fout, key, registration)

    fout.close()

    fout = open('cli_advanced.inc', 'w')

    for key, registration in second:
        render_script(fout, key, registration)


if __name__ == '__main__':
    main()
