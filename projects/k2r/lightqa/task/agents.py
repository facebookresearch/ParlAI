#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from parlai.core.message import Message
from parlai.core.metrics import F1Metric, normalize_answer, AverageMetric

from .build import build


class SummaryQATeacher(DialogTeacher):
    """
    Teacher for the SummaryQA dataset.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype'].split(':')[0]
        build(opt)
        opt['datafile'] = os.path.join(
            opt['datapath'], f'lightqa/lightqa-wild-summaryqa2-{self.datatype}.json'
        )
        self.id = 'summaryqa'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            self.episodes = json.load(data_file)
        for ex in self.episodes:
            episode_done = ex.pop('episode_done')
            yield ex, episode_done

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        if 'text' in model_response and model_response['text']:
            normalized_response = normalize_answer(model_response['text'])

            if labels:
                normalized_labels = [normalize_answer(a) for a in labels]
                self.metrics.add(
                    'norm_f1',
                    F1Metric.compute(normalized_response, normalized_labels),
                )
                self.metrics.add(
                    'norm_em',
                    AverageMetric(int(normalized_response in normalized_labels)),
                )
                self.metrics.add(
                    'kaa',
                    AverageMetric(
                        int(any([l in normalized_response for l in normalized_labels]))
                    ),
                )

                if 'knowledge_response' in model_response:
                    # Is the predicted knowledge response in the dialogue response?
                    self.metrics.add(
                        'pkaa',
                        AverageMetric(
                            int(
                                normalize_answer(model_response['knowledge_response'])
                                in normalized_response
                            )
                        ),
                    )


class DefaultTeacher(SummaryQATeacher):
    pass
