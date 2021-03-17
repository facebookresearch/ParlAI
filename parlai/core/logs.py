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

from collections import namedtuple
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


MetricDisplayData = namedtuple('MetricDisplayData', ('title', 'description'))

METRICS_DISPLAY_DATA = {
    "accuracy": MetricDisplayData("Accuracy", "Exact match text accuracy"),
    "bleu-4": MetricDisplayData(
        "BLEU-4",
        "BLEU-4 of the generation, under a standardized (model-independent) tokenizer",
    ),
    "clen": MetricDisplayData(
        "Context Length", "Average length of context in number of tokens"
    ),
    "clip": MetricDisplayData(
        "Clipped Gradients", "Fraction of batches with clipped gradients"
    ),
    "ctpb": MetricDisplayData("Context Tokens Per Batch", "Context tokens per batch"),
    "ctps": MetricDisplayData("Context Tokens Per Second", "Context tokens per second"),
    "ctrunc": MetricDisplayData(
        "Context Truncation", "Fraction of samples with some context truncation"
    ),
    "ctrunclen": MetricDisplayData(
        "Context Truncation Length", "Average length of context tokens truncated"
    ),
    "exps": MetricDisplayData("Examples Per Second", "Examples per second"),
    "exs": MetricDisplayData(
        "Examples", "Number of examples processed since last print"
    ),
    "f1": MetricDisplayData(
        "F1", "Unigram F1 overlap, under a standardized (model-independent) tokenizer"
    ),
    "gnorm": MetricDisplayData("Gradient Norm", "Gradient norm"),
    "gpu_mem": MetricDisplayData(
        "GPU Memory",
        "Fraction of GPU memory used. May slightly underestimate true value.",
    ),
    "hits@1": MetricDisplayData(
        "Hits@1", "Fraction of correct choices in 1 guess. (Similar to recall@K)"
    ),
    "hits@5": MetricDisplayData(
        "Hits@5", "Fraction of correct choices in 5 guesses. (Similar to recall@K)"
    ),
    "interdistinct-1": MetricDisplayData(
        "Interdistinct-1", "Fraction of n-grams unique across _all_ generations"
    ),
    "interdistinct-2": MetricDisplayData(
        "Interdistinct-1", "Fraction of n-grams unique across _all_ generations"
    ),
    "intradistinct-1": MetricDisplayData(
        "Intradictinct-1", "Fraction of n-grams unique _within_ each utterance"
    ),
    "intradictinct-2": MetricDisplayData(
        "Intradictinct-2", "Fraction of n-grams unique _within_ each utterance"
    ),
    "jga": MetricDisplayData("Joint Goal Accuracy", "Joint Goal Accuracy"),
    "llen": MetricDisplayData(
        "Label Length", "Average length of label in number of tokens"
    ),
    "loss": MetricDisplayData("Loss", "Loss"),
    "lr": MetricDisplayData("Learning Rate", "The most recent learning rate applied"),
    "ltpb": MetricDisplayData("Label Tokens Per Batch", "Label tokens per batch"),
    "ltps": MetricDisplayData("Label Tokens Per Second", "Label tokens per second"),
    "ltrunc": MetricDisplayData(
        "Label Truncation", "Fraction of samples with some label truncation"
    ),
    "ltrunclen": MetricDisplayData(
        "Label Truncation Length", "Average length of label tokens truncated"
    ),
    "rouge-1": MetricDisplayData("ROUGE-1", "ROUGE metrics"),
    "rouge-2": MetricDisplayData("ROUGE-2", "ROUGE metrics"),
    "rouge-L": MetricDisplayData("ROUGE-L", "ROUGE metrics"),
    "token_acc": MetricDisplayData(
        "Token Accuracy", "Token-wise accuracy (generative only)"
    ),
    "token_em": MetricDisplayData(
        "Token Exact Match",
        "Utterance-level token accuracy. Roughly corresponds to perfection under greedy search (generative only)",
    ),
    "total_train_updates": MetricDisplayData(
        "Total Train Updates", "Number of SGD steps taken across all batches"
    ),
    "tpb": MetricDisplayData(
        "Tokens Per Batch", "Total tokens (context + label) per batch"
    ),
    "tps": MetricDisplayData(
        "Tokens Per Second", "Total tokens (context + label) per second"
    ),
    "ups": MetricDisplayData("Updates Per Second", "Updates per second (approximate)"),
}


def _get_display_data(metric: str) -> MetricDisplayData:
    return METRICS_DISPLAY_DATA.get(
        metric, MetricDisplayData(title=metric, description="No description provided.")
    )


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
            help="Tensorboard logging of metrics",
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
        Log all metrics to tensorboard.

        :param setting:
            One of train/valid/test. Will be used as the title for the graph.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        for k, v in report.items():
            v = v.value() if isinstance(v, Metric) else v
            if not isinstance(v, numbers.Number):
                logging.error(f'k {k} v {v} is not a number')
                continue
            display = _get_display_data(metric=k)
            self.writer.add_scalar(
                f'{k}/{setting}',
                v,
                global_step=step,
                display_name=f"{display.title} ({k})",
                summary_description=display.description,
            )

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    """
    Log objects to Weights and Biases.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add WandB CLI args.
        """
        logger = parser.add_argument_group('WandB Arguments')
        logger.add_argument(
            '-wblog',
            '--wandb-log',
            type='bool',
            default=False,
            help="Enable W&B logging of metrics",
        )
        logger.add_argument(
            '--wandb-name',
            type=str,
            default=None,
            help='W&B run name. If not set, WandB will randomly generate a name.',
            hidden=True,
        )

        logger.add_argument(
            '--wandb-project',
            type=str,
            default=None,
            help='W&B project name. Defaults to timestamp. Usually the name of the sweep.',
            hidden=False,
        )
        return logger

    def __init__(self, opt: Opt, model=None):
        try:
            # wand is a very expensive thing to import. Wait until the
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
            reinit=True,  # in case of preemption
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
        Log all metrics to W&B.

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
