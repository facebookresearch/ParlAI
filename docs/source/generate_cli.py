#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import parlai.scripts
import io
import importlib


def indent(rawstr, indentstr='    '):
    lines = rawstr.split('\n')
    indented = [(indentstr + l) for l in lines]
    return '\n'.join(indented).replace('\\', '\\\\').rstrip()


def get_scripts():
    pathname = os.path.dirname(parlai.scripts.__file__)
    for fn in os.listdir(pathname):
        if fn.endswith('.py') and not fn.startswith('__'):
            yield os.path.join(pathname, fn)


def main():
    fout = open('cli_usage.inc', 'w')

    for script_path in get_scripts():
        script_name = os.path.basename(script_path).replace(".py", "")
        try:
            module = importlib.import_module("parlai.scripts." + script_name)
        except ModuleNotFoundError:
            continue
        if not hasattr(module, 'setup_args'):
            continue
        # header
        fout.write(script_name)
        fout.write('\n')
        fout.write('-' * len(script_name))
        fout.write('\n')

        # docs from the module
        fout.write('.. automodule:: parlai.scripts.{}\n'.format(script_name))

        # fout.write('   :members:\n')
        # fout.write('   :exclude-members: setup_args\n')
        fout.write('\n')
        fout.write('CLI help\n')
        fout.write('~~~~~~~~\n\n\n')

        # output the --help
        fout.write('.. code-block:: text\n\n')  # literal block
        capture = io.StringIO()
        parser = module.setup_args()
        parser.prog = 'python -m parlai.scripts.{}'.format(script_name)
        parser.print_help(capture)
        fout.write(indent(capture.getvalue()))
        fout.write('\n\n')


if __name__ == '__main__':
    main()
