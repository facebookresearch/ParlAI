#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

NEEDLE3 = '&mdash; ParlAI  documentation</title>'
REPLACEMENT3 = """
&mdash; ParlAI Documentation</title>
<link rel="shortcut icon" type="image/png" href="/static/img/favicon-32x32.png" sizes="32x32"/>
<link rel="shortcut icon" type="image/png" href="/static/img/favicon-16x16.png" sizes="16x16"/>
<link rel="shortcut icon" type="image/png" href="/static/img/favicon-96x96.png" sizes="96x96"/>
""".strip()  # noqa: E501


if __name__ == '__main__':
    for root, _, files in os.walk("build/docs/"):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                print("Postprocessing ", file_path)
                with open(file_path, 'r') as fin:
                    content = fin.read()
                    content = content.replace(NEEDLE3, REPLACEMENT3)
                with open(file_path, 'w') as fout:
                    fout.write(content)
