#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import pkg_resources

from parlai.opt_presets.docs import PRESET_DESCRIPTIONS

fout = open('opt_presets_list.inc', 'w')
fout.write('Preset name | Description | Expansion\n')
fout.write('----------- | ----------- | ---------\n')
for alias in sorted(PRESET_DESCRIPTIONS.keys()):
    description = PRESET_DESCRIPTIONS[alias]
    with pkg_resources.resource_stream("parlai", f"opt_presets/{alias}.opt") as f:
        expansion = json.load(f)
        assert isinstance(expansion, dict)
        expansion_str = []
        for key, value in expansion.items():
            key = '--' + key.replace('_', '-')
            value = str(value).replace("\n", r"\n")
            expansion_str.append(f'`{key} {value}`')
        expansion_str = " ".join(expansion_str)
        fout.write(f'`{alias}` | {description} | {expansion_str}\n')
fout.close()
