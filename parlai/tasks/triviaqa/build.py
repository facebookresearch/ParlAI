#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import json
import os
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager

RESOURCES = [
    DownloadableFile(
        'http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz',
        'triviaqa-rc.tar.gz',
        'ef94fac6db0541e5bb5b27020d067a8b13b1c1ffc52717e836832e02aaed87b9',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'TriviaQA')
    version = "4"  # build changes, not upstream changes

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Make *-union-noevidence-*.json
        for section in ["verified-{}-dev.json", "{}-train.json", "{}-dev.json"]:
            section = os.path.join(dpath, "qa", section)
            q2as = {}
            with PathManager.open(section.format("web")) as data_file:
                for datapoint in json.load(data_file)['Data']:
                    question = datapoint['Question']
                    answers = datapoint['Answer']['Aliases']
                    prime_answer = datapoint['Answer']['Value']
                    assert question not in q2as
                    q2as[question] = (prime_answer, answers)
            with PathManager.open(section.format("wikipedia")) as data_file:
                for datapoint in json.load(data_file)['Data']:
                    question = datapoint['Question']
                    answers = datapoint['Answer']['Aliases']
                    prime_answer = datapoint['Answer']['Value']
                    if question not in q2as:
                        q2as[question] = (prime_answer, answers)
                    else:
                        old_prime_answer, old_answers = q2as[question]
                        if old_prime_answer != prime_answer:
                            assert (old_prime_answer, prime_answer) == (
                                'Tony (Manero).',
                                'Tony',
                            )
                        q2as[question] = (old_prime_answer, old_answers + answers)
            with PathManager.open(
                section.format("noevidence-union"), "wt"
            ) as data_file:
                json.dump(
                    {
                        "Data": [
                            {
                                "Question": question,
                                "Answer": {
                                    "Value": prime_answer,
                                    "Aliases": sorted(
                                        list(
                                            set(
                                                a.replace(" (disambiguation)", "")
                                                for a in answers
                                            )
                                        )
                                    ),
                                },
                            }
                            for question, (prime_answer, answers) in q2as.items()
                        ]
                    },
                    data_file,
                )

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
