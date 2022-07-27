#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for ParlAI.

The standard way to train a model. After training, also computes
validation and test error.

The user must provide a model (with `--model`) and a task (with
`--task`).

## Examples

```shell
parlai train_model --model ir_baseline --task dialog_babi:Task:1 --model-file /tmp/model
parlai train_model --model seq2seq --task babi:Task10k:1 --model-file '/tmp/model' --batchsize 32 --learningrate 0.5
```
"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.
import copy
import random
import torch
import json
import os
import numpy as np
import signal
from typing import Tuple

from parlai.core.metrics import Metric
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.exceptions import StopTrainException
from parlai.core.logs import TensorboardLogger, WandbLogger
from parlai.core.metrics import (
    aggregate_named_reports,
    aggregate_unnamed_reports,
    dict_report,
)
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.worlds import create_task, World
from parlai.scripts.build_dict import build_dict, setup_args as setup_dict_args
from parlai.scripts.eval_model import get_task_world_logs
from parlai.utils.distributed import (
    sync_object,
    is_primary_worker,
    all_gather_list,
    is_distributed,
    get_rank,
    num_workers,
)
from parlai.utils.misc import Timer, nice_report
from parlai.utils.world_logging import WorldLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
from parlai.utils.io import PathManager


def _num_else_inf(opt: Opt, key: str, distributed_warn=False):
    if opt[key] > 0:
        if distributed_warn and is_distributed():
            nicekey = '--' + key.replace('_', '-')
            logging.warning(
                f'Using {nicekey} in distributed mode can lead to slowdowns. '
                'See https://github.com/facebookresearch/ParlAI/pull/3379 for more info.'
            )
        value = opt[key]
    else:
        value = float('inf')
    return value


def setup_args(parser=None) -> ParlaiParser:
    """
    Build the ParlAI parser, adding command line args if necessary.

    :param ParlaiParser parser:
        Preexisting parser to append options to. Will be created if needed.

    :returns:
        the ParlaiParser with CLI options added.
    """
    if parser is None:
        parser = ParlaiParser(True, True, 'Train a model')
    train = parser.add_argument_group('Training Loop Arguments')
    train.add_argument(
        '-et',
        '--evaltask',
        help='task to use for valid/test (defaults to the one used for training)',
    )
    train.add_argument(
        '--final-extra-opt',
        type=str,
        default='',
        help="A '.opt' file that is used for final eval. Useful for setting skip-generation to false. 'datatype' must be included as part of the opt.",
    )
    train.add_argument(
        '--eval-batchsize',
        type=int,
        hidden=True,
        help='Eval time batch size (defaults to same as -bs)',
    )
    train.add_argument(
        '--eval-dynamic-batching',  # FIXME: see https://github.com/facebookresearch/ParlAI/issues/3367
        default=None,
        type='nonestr',
        choices={None, 'off', 'full', 'batchsort'},
        help=(
            'Set dynamic batching at evaluation time. Set to off for '
            'train-only dynamic batching. Set to none (default) to use same '
            'setting as --dynamic-batching.'
        ),
    )
    train.add_argument(
        '--num-workers',
        default=0,
        type=int,
        help='Number of background workers (training only)',
    )
    train.add_argument('--display-examples', type='bool', default=False, hidden=True)
    train.add_argument('-eps', '--num-epochs', type=float, default=-1)
    train.add_argument('-ttim', '--max-train-time', type=float, default=-1)
    train.add_argument(
        '-tstep',
        '--max-train-steps',
        '--max-lr-steps',
        type=int,
        default=-1,
        help='End training after n model updates',
    )
    train.add_argument('-ltim', '--log-every-n-secs', type=float, default=-1)
    train.add_argument(
        '-lstep',
        '--log-every-n-steps',
        type=int,
        default=50,
        help='Log every n training steps',
    )
    train.add_argument(
        '-vtim',
        '--validation-every-n-secs',
        type=float,
        default=-1,
        help='Validate every n seconds. Saves model to model_file '
        '(if set) whenever best val metric is found',
    )
    train.add_argument(
        '-vstep',
        '--validation-every-n-steps',
        type=int,
        default=-1,
        help='Validate every n training steps. Saves model to model_file '
        '(if set) whenever best val metric is found',
    )
    train.add_argument(
        '-stim',
        '--save-every-n-secs',
        type=float,
        default=-1,
        help='Saves the model to model_file.checkpoint after '
        'every n seconds (default -1, never).',
    )
    train.add_argument(
        '-sval',
        '--save-after-valid',
        type='bool',
        default=False,
        help='Saves the model to model_file.checkpoint after '
        'every validation (default %(default)s).',
    )
    train.add_argument(
        '-veps',
        '--validation-every-n-epochs',
        type=float,
        default=-1,
        help='Validate every n epochs. Saves model to model_file '
        '(if set) whenever best val metric is found',
    )
    train.add_argument(
        '-vme',
        '--validation-max-exs',
        type=int,
        default=-1,
        hidden=True,
        help='max examples to use during validation (default -1 uses all)',
    )
    train.add_argument(
        '--short-final-eval',
        default=False,
        hidden=True,
        type='bool',
        help='If true, obeys --validation-max-exs in the final '
        'validation and test evaluations.',
    )
    train.add_argument(
        '-vp',
        '--validation-patience',
        type=int,
        default=10,
        help=(
            'number of iterations of validation where result'
            ' does not improve before we stop training'
        ),
    )
    train.add_argument(
        '-vmt',
        '--validation-metric',
        default='accuracy',
        help='key into report table for selecting best validation',
    )
    train.add_argument(
        '-vmm',
        '--validation-metric-mode',
        type=str,
        choices=['max', 'min'],
        help='the direction in which to optimize the validation metric, i.e. maximize or minimize',
    )
    train.add_argument(
        '-vcut',
        '--validation-cutoff',
        type=float,
        default=1.0,
        hidden=True,
        help='value at which training will stop if exceeded by metric',
    )
    train.add_argument(
        '-lfc',
        '--load-from-checkpoint',
        type='bool',
        default=True,
        hidden=True,
        help='load model from checkpoint if available',
    )
    train.add_argument(
        '-vshare',
        '--validation-share-agent',
        default=False,
        hidden=True,
        help='use a shared copy of the agent for validation. '
        'this will eventually default to True, but '
        'currently defaults to False.',
    )
    train.add_argument(
        '-mcs',
        '--metrics',
        type=str,
        default='default',
        help='list of metrics to show/compute, e.g. all, default,'
        'or give a list split by , like '
        'ppl,f1,accuracy,hits@1,rouge,bleu'
        'the rouge metrics will be computed as rouge-1, rouge-2 and rouge-l',
    )
    train.add_argument(
        '-micro',
        '--aggregate-micro',
        type='bool',
        default=False,
        help='Report micro-averaged metrics instead of macro averaged metrics.',
        recommended=False,
    )
    train.add_argument(
        '--world-logs',
        type=str,
        default='',
        help='Saves a jsonl file of the world logs.'
        'Set to the empty string to not save at all.',
    )
    train.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
    )
    train.add_argument('--seed', type=int, default=None)
    WorldLogger.add_cmdline_args(parser, partial_opt=None)
    TensorboardLogger.add_cmdline_args(parser, partial_opt=None)
    WandbLogger.add_cmdline_args(parser, partial_opt=None)
    parser = setup_dict_args(parser)
    return parser


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_eval_worlds(agent, opt, datatype):
    """
    Create a new eval world for the agent and the given opt.

    Overrides the datatype options for doing this.  Handles some magic
    overrides of other special options for the training script.

    :param Agent agent:
        The model being trained.

    :param Opt opt:
        The global CLI opts.

    :param string datatype:
        The new datatype.
    """

    if 'stream' in opt['datatype']:
        datatype += ':stream'
    opt = opt.copy()
    opt['datatype'] = datatype
    if opt.get('evaltask'):
        # if a different eval task is specified, use it.
        opt['task'] = opt['evaltask']
    if opt.get('eval_batchsize'):
        # override eval time batchsize
        opt['batchsize'] = opt['eval_batchsize']
    if opt.get('eval_dynamic_batching'):
        # FIXME: see issue tracked in https://github.com/facebookresearch/ParlAI/issues/3367
        # override eval time dynamic batching settings
        eval_dyn_batch = (
            None
            if opt['eval_dynamic_batching'] == 'off'
            else opt['eval_dynamic_batching']
        )
        opt['dynamic_batching'] = eval_dyn_batch

    tasks = opt['task'].split(',')
    worlds = []
    # possibly load agent
    if opt.get('validation_share_agent', False):
        valid_agent = create_agent_from_shared(agent.share())
    else:
        valid_agent = agent
    # create worlds
    for task in tasks:
        task_opt = opt.copy()  # copy opt since we edit the task
        task_opt['task'] = task
        valid_world = create_task(task_opt, valid_agent)
        worlds.append(valid_world)

    return worlds


class TrainLoop:
    """
    TrainLoop contains the core training loop logic.
    """

    def __init__(self, opt):
        # if python is called from a non-interactive shell, like a bash script,
        # it will by-default ignore SIGINTs, and KeyboardInterrupt exceptions are
        # not produced. This line brings them back
        signal.signal(signal.SIGINT, signal.default_int_handler)
        # Possibly load from checkpoint
        trainstats_suffix = '.trainstats'  # we might load training statistics from here
        if (
            opt['load_from_checkpoint']
            and opt.get('model_file')
            and PathManager.exists(opt['model_file'] + '.checkpoint')
        ):
            opt['init_model'] = opt['model_file'] + '.checkpoint'
            trainstats_suffix = '.checkpoint.trainstats'
        # Possibly build a dictionary (not all models do this).
        if not (opt.get('dict_file') or opt.get('model_file')):
            raise RuntimeError(
                'WARNING: For train_model, please specify either a '
                'model_file or dict_file.'
            )
        if 'dict_file' in opt:
            if opt['dict_file'] is None and opt.get('model_file'):
                opt['dict_file'] = opt['model_file'] + '.dict'
            logging.info("building dictionary first...")
            build_dict(opt, skip_if_built=True)

        # Create model and assign it to the specified task
        self.agent = create_agent(opt)
        self.agent.opt.log()
        self.world = create_task(opt, self.agent)
        # set up timers
        self.train_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()
        self.save_time = Timer()

        self.parleys = 0
        self._train_steps = 0
        self._last_log_steps = 0
        self.update_freq = opt.get('update_freq', 1)

        self.max_num_epochs = _num_else_inf(opt, 'num_epochs', distributed_warn=True)
        self.max_train_time = _num_else_inf(
            opt, 'max_train_time', distributed_warn=True
        )
        self.max_train_steps = _num_else_inf(opt, 'max_train_steps')
        self.log_every_n_secs = _num_else_inf(
            opt, 'log_every_n_secs', distributed_warn=True
        )
        self.log_every_n_steps = _num_else_inf(opt, 'log_every_n_steps')
        self.val_every_n_secs = _num_else_inf(
            opt, 'validation_every_n_secs', distributed_warn=True
        )
        self.val_every_n_epochs = _num_else_inf(
            opt, 'validation_every_n_epochs', distributed_warn=True
        )
        self.val_every_n_steps = _num_else_inf(opt, 'validation_every_n_steps')
        self.save_every_n_secs = _num_else_inf(
            opt, 'save_every_n_secs', distributed_warn=True
        )

        # smart defaults for --validation-metric-mode
        if opt['validation_metric'] in {'loss', 'ppl', 'mean_rank'}:
            opt['validation_metric_mode'] = 'min'
        elif opt['validation_metric'] in {'accuracy', 'hits@1', 'hits@5', 'f1', 'bleu'}:
            opt['validation_metric_mode'] = 'max'
        if opt.get('validation_metric_mode') is None:
            opt['validation_metric_mode'] = 'max'

        self.last_valid_epoch = 0
        self._last_valid_steps = 0
        self.valid_optim = 1 if opt['validation_metric_mode'] == 'max' else -1
        self.train_reports = []
        self.valid_reports = []
        self.final_valid_report = {}
        self.final_test_report = {}
        self.final_extra_valid_report = {}
        self.best_valid = None

        self.impatience = 0
        self.saved = False
        self.valid_worlds = None
        self.opt = opt

        # we may have been preempted, make sure we note that amount
        self._preempted_epochs = 0.0
        if opt.get('model_file') and PathManager.exists(
            opt['model_file'] + trainstats_suffix
        ):
            # looks like we were preempted. make sure we load up our total
            # training stats, etc
            with PathManager.open(opt['model_file'] + trainstats_suffix) as ts:
                obj = json.load(ts)
                self.parleys = obj.get('parleys', 0)
                self._preempted_epochs = obj.get('total_epochs', 0)
                self.train_time.total = obj.get('train_time', 0)
                self._train_steps = obj.get('train_steps', 0)
                self.impatience = obj.get('impatience', 0)
                self.valid_reports = obj.get('valid_reports', [])
                if self.valid_reports:
                    self.last_valid_epoch = self.valid_reports[-1].get(
                        'total_epochs', 0.0
                    )
                self.train_reports = obj.get('train_reports', [])
                if 'best_valid' in obj:
                    self.best_valid = obj['best_valid']
                else:
                    # old method
                    if opt.get('model_file') and PathManager.exists(
                        opt['model_file'] + '.best_valid'
                    ):
                        with PathManager.open(
                            opt['model_file'] + ".best_valid", 'r'
                        ) as f:
                            x = f.readline()
                            self.best_valid = float(x)
                            f.close()

        if opt['tensorboard_log'] and is_primary_worker():
            self.tb_logger = TensorboardLogger(opt)
        if opt['wandb_log'] and is_primary_worker():
            model = self.agent.model if hasattr(self.agent, 'model') else None
            self.wb_logger = WandbLogger(opt, model)

    def save_model(self, suffix=None):
        """
        Save the model to disk, possibly with a suffix.
        """
        if not self.opt.get('model_file'):
            # nothing to save to, just exit
            return

        fn = self.opt['model_file']
        if suffix:
            fn += suffix

        if not is_primary_worker():
            # never do IO as a non-primary worker
            if hasattr(self.agent, 'save_nonprimary'):
                self.agent.save_nonprimary(fn)
            return

        while True:
            # don't ever let a ctrl-c interrupt saving
            try:
                self.agent.save(fn)
                self._save_train_stats(suffix)
                break
            except KeyboardInterrupt:
                pass

    def _save_train_stats(self, suffix=None):
        if not is_primary_worker():
            # never do IO as a non-primary worker
            return
        fn = self.opt.get('model_file', None)
        if not fn:
            return
        if suffix:
            fn += suffix
        fn += '.trainstats'
        with PathManager.open(fn, 'w') as f:
            json.dump(
                {
                    'parleys': self.parleys,
                    'train_time': self.train_time.time(),
                    'train_steps': self._train_steps,
                    'total_epochs': self._total_epochs,
                    'train_reports': self.train_reports,
                    'valid_reports': self.valid_reports,
                    'best_valid': self.best_valid,
                    'impatience': self.impatience,
                    'final_valid_report': dict_report(self.final_valid_report),
                    'final_test_report': dict_report(self.final_test_report),
                    'final_extra_valid_report': dict_report(
                        self.final_extra_valid_report
                    ),
                },
                f,
                indent=4,
            )

    def validate(self):
        """
        Perform a validation run, checking whether we should stop training.

        :return: boolean indicating whether training should stop
        :rtype: bool
        """
        opt = self.opt

        if self.valid_worlds is None:
            # we need to load the world now
            self.valid_worlds = load_eval_worlds(self.agent, opt, 'valid')

        # run evaluation on valid set
        valid_report = self._run_eval(
            self.valid_worlds, opt, 'valid', opt['validation_max_exs']
        )
        v = dict_report(valid_report)
        v['train_time'] = self.train_time.time()
        v['parleys'] = self.parleys
        v['train_steps'] = self._train_steps
        v['total_exs'] = self._total_exs
        v['total_epochs'] = self._total_epochs
        self.valid_reports.append(v)
        # logging
        if opt['tensorboard_log'] and is_primary_worker():
            valid_report['total_exs'] = self._total_exs
            self.tb_logger.log_metrics('valid', self.parleys, valid_report)
            # flush on a validation
            self.tb_logger.flush()
        if opt['wandb_log'] and is_primary_worker():
            valid_report['total_exs'] = self._total_exs
            self.wb_logger.log_metrics('valid', self.parleys, valid_report)

        # send valid metrics to agent if the agent wants them
        if hasattr(self.agent, 'receive_metrics'):
            self.agent.receive_metrics(valid_report)

        # check which metric to look at
        new_valid = valid_report[opt['validation_metric']]

        if isinstance(new_valid, Metric):
            new_valid = new_valid.value()

        # check if this is the best validation so far
        if (
            self.best_valid is None
            or self.valid_optim * new_valid > self.valid_optim * self.best_valid
        ):
            logging.success(
                'new best {}: {:.4g}{}'.format(
                    opt['validation_metric'],
                    new_valid,
                    ' (previous best was {:.4g})'.format(self.best_valid)
                    if self.best_valid is not None
                    else '',
                )
            )
            self.best_valid = new_valid
            self.impatience = 0
            if opt.get('model_file'):
                logging.info(f"saving best valid model: {opt['model_file']}")
                self.save_model()
                self.saved = True
            if (
                opt['validation_metric_mode'] == 'max'
                and self.best_valid >= opt['validation_cutoff']
            ) or (
                opt['validation_metric_mode'] == 'min'
                and self.best_valid <= opt['validation_cutoff']
            ):
                logging.info('task solved! stopping.')
                return True
        else:
            self.impatience += 1
            logging.report(
                'did not beat best {}: {} impatience: {}'.format(
                    opt['validation_metric'], round(self.best_valid, 4), self.impatience
                )
            )
        self.validate_time.reset()

        # saving
        if opt.get('model_file') and opt.get('save_after_valid'):
            logging.info(f"saving model checkpoint: {opt['model_file']}.checkpoint")
            self.save_model('.checkpoint')

        # check if we are out of patience
        if (
            opt['validation_patience'] > 0
            and self.impatience >= opt['validation_patience']
        ):
            logging.info('ran out of patience! stopping training.')
            return True
        return False

    def _run_single_eval(self, opt, valid_world, max_exs, datatype, is_multitask, task):

        # run evaluation on a single world
        valid_world.reset()

        world_logger = None
        task_opt = opt.copy()
        # set up world logger for the "test" fold
        if opt['world_logs'] and datatype == 'test':
            task_opt['world_logs'] = get_task_world_logs(
                task, opt['world_logs'], is_multitask
            )
            world_logger = WorldLogger(task_opt)

        cnt = 0
        max_cnt = max_exs if max_exs > 0 else float('inf')
        while not valid_world.epoch_done() and cnt < max_cnt:
            valid_world.parley()
            if world_logger is not None:
                world_logger.log(valid_world)
            if cnt == 0 and opt['display_examples']:
                print(valid_world.display() + '\n~~')
                print(valid_world.report())
            cnt = valid_world.report().get('exs') or 0

        if world_logger is not None:
            # dump world acts to file
            world_logger.reset()  # add final acts to logs
            if is_distributed():
                rank = get_rank()
                base_outfile, extension = os.path.splitext(task_opt['world_logs'])
                outfile = base_outfile + f'_{rank}' + extension
            else:
                outfile = task_opt['world_logs']
            world_logger.write(outfile, valid_world, file_format=opt['save_format'])

        valid_report = valid_world.report()
        if opt.get('validation_share_agent', False):
            valid_world.reset()  # make sure world doesn't remember valid data

        return valid_report

    def _run_eval(
        self,
        valid_worlds,
        opt,
        datatype,
        max_exs=-1,
        write_log=False,
        extra_log_suffix="",
    ):
        """
        Eval on validation/test data.

        :param valid_world:
            list of the pre-created validation worlds.
        :param opt:
            the options that specific the task, eval_task, etc
        :param datatype:
            the datatype to use, such as "valid" or "test"
        :param bool write_log:
            specifies to write metrics to file if the model_file is set
        :param int max_exs:
            limits the number of examples if max_exs > 0
        """

        logging.info(f'running eval: {datatype}')
        timer = Timer()
        reports = []

        max_exs_per_worker = max_exs / (len(valid_worlds) * num_workers())
        is_multitask = len(valid_worlds) > 1
        for index, v_world in enumerate(valid_worlds):
            if opt.get('evaltask'):
                task = opt['evaltask'].split(',')[index]
            else:
                task = opt['task'].split(',')[index]
            task_report = self._run_single_eval(
                opt, v_world, max_exs_per_worker, datatype, is_multitask, task
            )
            reports.append(task_report)

        tasks = [world.getID() for world in valid_worlds]
        named_reports = dict(zip(tasks, reports))
        report = aggregate_named_reports(
            named_reports, micro_average=self.opt.get('aggregate_micro', False)
        )
        # get the results from all workers
        report = self._sync_metrics(report)

        metrics = f'{datatype}:\n{nice_report(report)}\n'
        logging.info(f'eval completed in {timer.time():.2f}s')
        logging.report(metrics)

        # write to file
        if write_log and opt.get('model_file') and is_primary_worker():
            # Write out metrics
            with PathManager.open(
                opt['model_file'] + extra_log_suffix + '.' + datatype, 'a'
            ) as f:
                f.write(f'{metrics}\n')

        return report

    def _run_final_extra_eval(self, opt):
        final_valid_opt = copy.deepcopy(opt)
        final_valid_opt_raw = Opt.load_init(opt['final_extra_opt'])
        final_datatype = final_valid_opt_raw["datatype"]
        for k, v in final_valid_opt_raw.items():
            final_valid_opt[k] = v
        final_max_exs = (
            final_valid_opt['validation_max_exs']
            if final_valid_opt.get('short_final_eval')
            else -1
        )
        final_valid_world = load_eval_worlds(
            self.agent, final_valid_opt, final_datatype
        )
        final_valid_report = self._run_eval(
            final_valid_world,
            final_valid_opt,
            final_datatype,
            final_max_exs,
            write_log=True,
            extra_log_suffix="_extra",
        )
        if opt['wandb_log'] and is_primary_worker():
            self.wb_logger.log_final(final_datatype, final_valid_report)

        return final_valid_report

    def _sync_metrics(self, metrics):
        """
        Sync training metrics across workers.

        A handful of special cases are handled as exceptions, and the remaining metrics
        are simply averaged across workers.
        """
        if not is_distributed():
            # nothing special needed
            return metrics
        all_versions = all_gather_list(metrics)
        return aggregate_unnamed_reports(all_versions)

    def _compute_eta(
        self, epochs_completed: float, time_elapsed: float, steps_taken: int
    ):
        """
        Compute the estimated seconds remaining in training.

        :param float epochs_completed: number of epochs already completed.
        :param float time_elapsed: total time spent already, in seconds.
        :return: ETA in seconds, or None if not computable
        """
        # start off with no estimate
        eta = None

        # Determine time_left and num_epochs
        max_epochs = self.opt.get('num_epochs', 0)
        if max_epochs > 0 and epochs_completed > 0:
            epoch_progress = epochs_completed / max_epochs
            eta = (1 - epoch_progress) * time_elapsed / epoch_progress

        max_training_time = self.opt.get('max_training_time', -1)
        if max_training_time > 0:
            time_left = max_training_time - time_elapsed
            if eta is None or time_left < eta:
                eta = time_left

        max_train_steps = self.opt.get('max_train_steps', -1)
        if max_train_steps > 0 and steps_taken > 0:
            steps_progress = steps_taken / max_train_steps
            eta = (1 - steps_progress) * time_elapsed / steps_progress

        return eta

    def _get_time(self, world: World) -> Tuple[float, float, float]:
        """
        Return train, log, and validate timing.

        If relying on the time for validation/logging/max train time purposes,
        we sync and return primary worker's time.

        Otherwise, it's not super relevant what we do here.

        **SIDE EFFECT**: Update _total_epochs trained.

        :param world:
            current running world

        :return (train, log, valid):
            return time for each of train, log, and validation
        """
        if (
            self.max_train_time < float('inf')
            or self.log_every_n_secs < float('inf')
            or self.val_every_n_secs < float('inf')
            or self.val_every_n_epochs < float('inf')
            or self.max_num_epochs < float('inf')
        ):
            self._total_epochs = self._preempted_epochs + sum(
                all_gather_list(world.get_total_epochs())
            )
            train_time, log_time, validate_time, save_time = sync_object(
                (
                    self.train_time.time(),
                    self.log_time.time(),
                    self.validate_time.time(),
                    self.save_time.time(),
                )
            )
        else:
            train_time, log_time, validate_time, save_time = (
                self.train_time.time(),
                self.log_time.time(),
                self.validate_time.time(),
                self.save_time.time(),
            )
            self._total_epochs = self._preempted_epochs + (
                num_workers() * world.get_total_epochs()
            )

        return train_time, log_time, validate_time, save_time

    def log(self):
        """
        Output a training log entry.
        """
        opt = self.opt
        if opt['display_examples']:
            print(self.world.display() + '\n~~')
        logs = []
        # get report
        train_report = self.world.report()
        train_report = self._sync_metrics(train_report)
        self.world.reset_metrics()

        train_report_trainstats = dict_report(train_report)
        train_report_trainstats['total_epochs'] = self._total_epochs
        train_report_trainstats['total_exs'] = self._total_exs
        train_report_trainstats['parleys'] = self.parleys
        train_report_trainstats['train_steps'] = self._train_steps
        train_report_trainstats['train_time'] = self.train_time.time()
        self.train_reports.append(train_report_trainstats)

        # time elapsed
        logs.append(f'time:{self.train_time.time():.0f}s')
        logs.append(f'total_exs:{self._total_exs}')
        logs.append(f'total_steps:{self._train_steps}')

        if self._total_epochs >= 0:
            # only if it's unbounded
            logs.append(f'epochs:{self._total_epochs:.2f}')

        time_left = self._compute_eta(
            self._total_epochs, self.train_time.time(), self._train_steps
        )
        if time_left is not None:
            logs.append(f'time_left:{max(0,time_left):.0f}s')

        log = '{}\n{}\n'.format(' '.join(logs), nice_report(train_report))
        logging.info(log)
        self.log_time.reset()
        self._last_log_steps = 0

        if opt['tensorboard_log'] and is_primary_worker():
            self.tb_logger.log_metrics('train', self.parleys, train_report)
        if opt['wandb_log'] and is_primary_worker():
            self.wb_logger.log_metrics('train', self.parleys, train_report)

        return train_report

    def train_steps(self):
        """
        Core training loop.

        Yields a metrics dict with each log.
        """
        logging.info('training...')
        opt = self.opt
        world = self.world
        with world:
            while True:
                # do one example / batch of examples
                try:
                    world.parley()
                except StopTrainException as e:
                    logging.info(f"Stopping from {e}")
                    break

                self.parleys += 1
                self._train_steps = self.parleys // self.update_freq
                self._last_log_steps += 1 / self.update_freq

                # the following additionally updates self._total_epochs
                train_time, log_time, validate_time, save_time = self._get_time(world)
                # get the total training examples done, compute epochs
                exs_per_epoch = world.num_examples()
                self._total_exs = int(np.round(self._total_epochs * exs_per_epoch))

                # check counters and timers
                if self._total_epochs >= self.max_num_epochs:
                    yield self.log()
                    logging.info(
                        f'num_epochs completed:{self.max_num_epochs} time elapsed:{train_time}s'
                    )
                    break
                if train_time > self.max_train_time:
                    logging.info(f'max_train_time elapsed:{train_time}s')
                    break
                if self._train_steps >= self.max_train_steps:
                    logging.info(
                        f'max_train_steps elapsed:{self._train_steps} '
                        f'time elapsed:{train_time}s'
                    )
                    break
                if (
                    log_time > self.log_every_n_secs
                    or self._last_log_steps >= self.log_every_n_steps
                ):
                    yield self.log()
                if (
                    validate_time > self.val_every_n_secs
                    or self._total_epochs - self.last_valid_epoch
                    >= self.val_every_n_epochs
                    or self._train_steps - self._last_valid_steps
                    >= self.val_every_n_steps
                ):
                    try:
                        # log before we validate
                        if self._last_log_steps:
                            yield self.log()
                        world.reset_metrics()
                        stop_training = self.validate()
                    except StopTrainException:
                        break
                    # reset the log time because we logged right before validating
                    self.log_time.reset()
                    self.last_valid_epoch = self._total_epochs
                    self._last_valid_steps = self._train_steps
                    if stop_training:
                        break
                    # make sure metrics are clean before we log
                    world.reset_metrics()
                if save_time > self.save_every_n_secs and opt.get('model_file'):
                    logging.info(
                        f"saving model checkpoint: {opt['model_file']}.checkpoint"
                    )
                    if opt['tensorboard_log'] and is_primary_worker():
                        self.tb_logger.flush()
                    self.save_model('.checkpoint')
                    self.save_time.reset()

        if not sync_object(self.saved):
            # save agent
            self.save_model()

        # there's a rare edge case where the we never saved the model, and we try
        # # to reload it. This sync_object ensures all workers wait for the primary
        # worker to finish flushing before loading from disk.
        sync_object(None)
        if opt.get('model_file'):
            # clean up all our memory, just to make sure we don't OOM on GPU when
            # reloading the world
            del world
            del self.world
            del self.agent
            del self.valid_worlds
            # reload best validation model
            self.agent = create_agent(opt)

    def train(self):
        """
        Perform a training run.

        :return: tuple of reports (validation_report, test_report)
        """
        opt = self.opt
        for _train_log in self.train_steps():
            # we've already done what we need in these
            pass

        # perform final validation/testing
        valid_worlds = load_eval_worlds(self.agent, opt, 'valid')
        max_exs = opt['validation_max_exs'] if opt.get('short_final_eval') else -1
        self.final_valid_report = self._run_eval(
            valid_worlds, opt, 'valid', max_exs, write_log=True
        )
        test_worlds = load_eval_worlds(self.agent, opt, 'test')
        self.final_test_report = self._run_eval(
            test_worlds, opt, 'test', max_exs, write_log=True
        )

        if opt['wandb_log'] and is_primary_worker():
            self.wb_logger.log_final('valid', self.final_valid_report)
            self.wb_logger.log_final('test', self.final_test_report)
            self.wb_logger.finish()

        if valid_worlds:
            for valid_world in valid_worlds:
                valid_world.shutdown()
        if test_worlds:
            for test_world in test_worlds:
                test_world.shutdown()

        print_announcements(opt)

        if opt['final_extra_opt'] != '':
            self.final_extra_valid_report = self._run_final_extra_eval(opt)

        if opt['wandb_log'] and is_primary_worker():
            self.wb_logger.finish()

        self._save_train_stats()

        return self.final_valid_report, self.final_test_report


@register_script('train_model', aliases=['tm', 'train'])
class TrainModel(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        self.train_loop = TrainLoop(self.opt)
        if self.opt['seed'] is not None:
            set_seed(self.opt['seed'])
        return self.train_loop.train()


if __name__ == '__main__':
    TrainModel.main()
