#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json
import re
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/salesforce/decanlp/d594b2bf127e13d0e61151b6a2af3bf63612f380/local_data/schema.txt',
        'schema.txt',
        '31da9bee05796bbe0f6c957f54d1eb82eb5c644a8ee59f2ff1fa890eff3885dd',
        zipped=False,
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'MWSC')
    version = 'None'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        pattern = '\\[.*\\]'

        def get_both_schema(context):
            variations = [x[1:-1].split('/') for x in re.findall(pattern, context)]
            splits = re.split(pattern, context)
            results = []
            for which_schema in range(2):
                vs = [v[which_schema] for v in variations]
                context = ''
                for idx in range(len(splits)):
                    context += splits[idx]
                    if idx < len(vs):
                        context += vs[idx]
                results.append(context)
            return results

        schemas = []
        with open(os.path.join(dpath, RESOURCES[0].file_name)) as schema_file:
            schema = []
            for line in schema_file:
                if len(line.split()) == 0:
                    schemas.append(schema)
                    schema = []
                    continue
                else:
                    schema.append(line.strip())

        examples = []
        for schema in schemas:
            context, question, answer = schema
            contexts = get_both_schema(context)
            questions = get_both_schema(question)
            answers = answer.split('/')
            for idx in range(2):
                answer = answers[idx]
                question = questions[idx] + ' {} or {}?'.format(answers[0], answers[1])
                examples.append(
                    {'context': contexts[idx], 'question': question, 'answer': answer}
                )

        traindev = examples[:-100]
        test = examples[-100:]
        train = traindev[:80]
        dev = traindev[80:]

        splits = ['train', 'validation', 'test']
        for split, examples in zip(splits, [train, dev, test]):
            split_fname = '{}.json'.format(split)
            with open(os.path.join(dpath, split_fname), 'a') as split_file:
                for ex in examples:
                    split_file.write(json.dumps(ex) + '\n')

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
