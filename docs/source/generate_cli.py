#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect
import argparse
import parlai.core.script as pcs


def render_script(fout, key, registration):
    script_parser = registration.klass.setup_args()
    description = script_parser.description
    fout.write(f"## {key}\n\n")
    if description:
        fout.write(f"__Short description:__ {description}\n\n")
    if registration.aliases:
        aliases = ", ".join(f'`{a}`' for a in registration.aliases)
        fout.write(f"__Aliases:__ {aliases}\n")

    mod = importlib.import_module(registration.klass.__module__)
    doc = inspect.getdoc(mod)
    if doc:
        doc = doc.replace("## Examples", "### Examples")
        fout.write(doc)
        fout.write("\n")

    actions = []
    for action in script_parser._actions:
        if hasattr(action, 'hidden') and action.hidden:
            # some options are marked hidden
            continue
        if action.dest == argparse.SUPPRESS:
            continue
        if action.dest == 'help' or action.dest == 'helpall':
            continue
        action_strings = ",  ".join(f'`{a}`' for a in action.option_strings)
        if not action_strings:
            continue
        description = []
        if action.help:
            h = action.help
            if not h[0].isupper():
                h = h[0].upper() + h[1:]
            h = h.replace("%(default)s", f'`{action.default}`')
            description += [h]
        # list choices if there are any
        if action.choices:
            description += ["Choices: " + ", ".join(f'`{c}`' for c in action.choices)]
        default_value = ""
        if action.default and action.default != argparse.SUPPRESS:
            default_value += f"Default: `{action.default}`.  "
        if hasattr(action, 'recommended') and action.recommended:
            default_value += f"Recommended: `{action.recommended}`. "

        # special escape for a few args which use a literal newline as their default
        if default_value:
            default_value = default_value.replace("\n", "\\n")
            description.append(default_value)

        # escape for the fact that we're inserting this inside a table
        description = " <BR> ".join(description)
        actions.append((action_strings, description))

    if not actions:
        return

    fout.write("### CLI Arguments\n\n")
    fout.write("| Argument | Description |\n")
    fout.write("| ------- | --------- |\n")
    for action, description in actions:
        fout.write(f"| {action} | {description} |\n")
    fout.write('\n\n')


def main():
    pcs.setup_script_registry()

    first = []
    second = []
    for key, registration in sorted(pcs.SCRIPT_REGISTRY.items()):
        if not registration.hidden:
            first.append((key, registration))
        else:
            second.append((key, registration))

    with open("cli_usage.inc", "w") as fout:
        for i, (key, registration) in enumerate(first):
            if i != 0:
                fout.write("\n----------\n")
            render_script(fout, key, registration)

    with open("cli_advanced.inc", "w") as fout:
        for i, (key, registration) in enumerate(second):
            if i != 0:
                fout.write("\n----------\n")
            render_script(fout, key, registration)


if __name__ == '__main__':
    main()
