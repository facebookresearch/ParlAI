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

import os
from typing import Optional
from parlai.core.params import ParlaiParser
import json
import numbers
import datetime
from parlai.core.opt import Opt
from parlai.core.metrics import Metric, dict_report
from parlai.utils.io import PathManager
import parlai.utils.logging as logging


class TensorboardLogger(object):
    """
    Log objects to tensorboard.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add tensorboard CLI args.
        """
        logger = parser.add_argument_group('Tensorboard Arguments')
        logger.add_argument(
            '-tblog',
            '--tensorboard-log',
            type='bool',
            default=False,
            help="Tensorboard logging of metrics, default is %(default)s",
            hidden=False,
        )
        logger.add_argument(
            '-tblogdir',
            '--tensorboard-logdir',
            type=str,
            default=None,
            help="Tensorboard logging directory, defaults to model_file.tensorboard",
            hidden=False,
        )
        return parser

    def __init__(self, opt: Opt):
        try:
            # tensorboard is a very expensive thing to import. Wait until the
            # last second to import it.
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please run `pip install tensorboard tensorboardX`.')

        if opt['tensorboard_logdir'] is not None:
            tbpath = opt['tensorboard_logdir']
        else:
            tbpath = opt['model_file'] + '.tensorboard'

        logging.debug(f'Saving tensorboard logs to: {tbpath}')
        if not PathManager.exists(tbpath):
            PathManager.mkdirs(tbpath)
        self.writer = SummaryWriter(tbpath, comment=json.dumps(opt))

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
                self.writer.add_scalar(f'{k}/{setting}', v, global_step=step)
            elif isinstance(v, Metric):
                self.writer.add_scalar(f'{k}/{setting}', v.value(), global_step=step)
            else:
                logging.error(f'k {k} v {v} is not a number')

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    """
    Log objects to Weights and Biases.
    """

    @staticmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add w&b CLI args.
        """
        logger = argparser.add_argument_group('Tensorboard Arguments')
        logger.add_argument(
            '-wblog',
            '--wandb-log',
            type='bool',
            default=False,
            help="Enable W&B logging of metrics, default is %(default)s",
            hidden=False,
        )
        logger.add_argument(
            '--wandb-name', type=str, default=None, help='W&B run name', hidden=True
        )

        logger.add_argument(
            '--wandb-project',
            type=str,
            default=None,
            help='W&B project name. Defaults to timestamp.',
            hidden=False,
        )

    def __init__(self, opt: Opt, model=None):
        try:
            # tensorboard is a very expensive thing to import. Wait until the
            # last second to import it.
            import wandb

        except ImportError:
            raise ImportError('Please run `pip install wandb`.')

        name = opt.get('wandb_name')
        project = opt.get('wandb_project') or datetime.datetime.now().strftime(
            '%Y-%m-%d-%H-%M'
        )

        self.run = wandb.init(
            name=name,
            project=project,
            dir=os.path.dirname(opt['model_file']),
            notes=f"{opt['model_file']}",
        )
        # suppress wandb's output
        logging.getLogger("wandb").setLevel(logging.ERROR)
        for key, value in opt.items():
            if value is None or isinstance(value, (str, numbers.Number, tuple)):
                setattr(self.run.config, key, value)
        if model is not None:
            self.run.watch(model)

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
        report = dict_report(report)
        report = {
            f'{k}/{setting}': v
            for k, v in report.items()
            if isinstance(v, numbers.Number)
        }
        report['custom_step'] = step
        self.run.log(report)

    def log_final(self, setting, report):
        report = dict_report(report)
        report = {
            f'{k}/{setting}': v
            for k, v in report.items()
            if isinstance(v, numbers.Number)
        }
        for key, value in report.items():
            self.run.summary[key] = value

    def finish(self):
        self.run.finish()

    def flush(self):
        pass
