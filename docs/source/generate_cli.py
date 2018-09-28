#!/usr/bin/env python3

import os
import parlai.scripts
import subprocess


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
        captured = subprocess.run(
            ['python3', script_path, '--help'],
            stdout=subprocess.PIPE
        )
        fout.write(script_name)
        fout.write('\n')
        fout.write('-' * len(script_name))
        fout.write('\n')
        fout.write('::\n\n')
        fout.write(indent(captured.stdout.decode('utf-8')))
        fout.write('\n\n')


if __name__ == '__main__':
    main()
