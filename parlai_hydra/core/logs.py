#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Log metrics to tensorboard.

This file provides interface to log any metrics in tensorboard, could be
extended to any other tool like visdom.

.. code-block: none

   tensorboard --logdir <PARLAI_DATA/tensorboard> --port 8888.
"""

import logging
import numbers
import os

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

log = logging.getLogger(__file__)


class TensorboardLogger(object):
    """Log objects to tensorboard."""

    def __init__(self, cfg):
        if not cfg.enabled:
            return

        if SummaryWriter is None:
            raise ImportError('Please run `pip install tensorboard tensorboardX`.')

        path = cfg.tensorboard.file_path
        log.info("Saving tensorboard logs to: {}".format(cfg.tensorboard.file_path))
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = SummaryWriter(path, comment=cfg.pretty(resolve=True))

    def log_metrics(self, setting, step, report):
        """
        Add all metrics from tensorboard_metrics opt key.

        :param setting:
            One of train/valid/test. Will be used as the title for the graph.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        for k, v in report.items():
            if isinstance(v, numbers.Number):
                self.writer.add_scalar(f'{setting}/{k}', v, global_step=step)
            else:
                log.info(f'k {k} v {v} is not a number')
