#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This file provides interface to log any metrics in tensorboard, could be
extended to any other tool like visdom
Tensorboard:
    If you use tensorboard logging, all event folders will be stored in
        PARLAI_DATA/tensorboard folder. In order to
    Open it with TB, launch tensorboard as:
        tensorboard --logdir <PARLAI_DATA/tensorboard> --port 8888.
"""

import os


class TensorboardLogger(object):
    _shared_state = {}

    @staticmethod
    def add_cmdline_args(argparser):
        logger = argparser.add_argument_group('Tensorboard Arguments')
        logger.add_argument(
            '-tblog', '--tensorboard-log', type='bool', default=False,
            help="Tensorboard logging of metrics, default is %(default)s",
            hidden=True
        )
        logger.add_argument(
            '-tbtag', '--tensorboard-tag', type=str, default=None,
            help='Specify all opt keys which you want to be presented in in TB name',
            hidden=True
        )
        logger.add_argument(
            '-tbmetrics', '--tensorboard-metrics', type=str, default=None,
            help='Specify metrics which you want to track, it will be extracted '
                 'from report dict.',
            hidden=True
        )
        logger.add_argument(
            '-tbcomment', '--tensorboard-comment', type=str, default='',
            hidden=True,
            help='Add any line here to distinguish your TB event file, optional'
        )

    def __init__(self, opt):
        self.__dict__ = self._shared_state
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please `pip install tensorboardX` for logs with TB.')

        if opt['tensorboard_tag'] is None:
            tensorboard_tag = opt['starttime']
        else:
            tensorboard_tag = opt['starttime'] + '__'.join([
                i + '-' + str(opt[i])
                for i in opt['tensorboard_tag'].split(',')
            ])
        if opt['tensorboard_comment']:
            tensorboard_tag += '__' + opt['tensorboard_comment']

        tbpath = os.path.join(os.path.dirname(opt['model_file']), 'tensorboard')
        print('[ Saving tensorboard logs here: {} ]'.format(tbpath))
        if not os.path.exists(tbpath):
            os.makedirs(tbpath)
        self.writer = SummaryWriter(
            log_dir='{}/{}'.format(tbpath, tensorboard_tag))
        if opt['tensorboard_metrics'] is None:
            self.tbmetrics = ['ppl', 'loss']
        else:
            self.tbmetrics = opt['tensorboard_metrics'].split(',')

    def add_metrics(self, setting, step, report):
        """
        Adds all metrics from tensorboard_metrics opt key

        :param setting: whatever setting is used, train valid or test, it will
            be just the title of the graph
        :param step: num of parleys (x axis in graph), in train - parleys, in
            valid - wall time
        :param report: from TrainingLoop
        :return:
        """
        for met in self.tbmetrics:
            if met in report.keys():
                self.writer.add_scalar(
                    "{}/{}".format(setting, met),
                    report[met],
                    global_step=step
                )

    def add_scalar(self, name, y, step=None):
        """
        :param name: the title of the graph, use / to group like "train/loss/ce" or so
        :param y: value
        :param step: x axis step
        :return:
        """
        self.writer.add_scalar(name, y, step)

    def add_histogram(self, name, vector, step=None):
        """
        :param name:
        :param vector:
        :return:
        """
        self.writer.add_histogram(name, vector, step)

    def add_text(self, name, text, step=None):
        self.writer.add_text(name, text, step)
