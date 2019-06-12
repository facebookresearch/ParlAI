#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for ParlAI.

The standard way to train a model. After training, also computes validation
and test error.

The user must provide a model (with ``--model``) and a task (with ``--task`` or
``--pytorch-teacher-task``).

Examples
--------
.. code-block:: shell

  python -m parlai.scripts.train -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
  python -m parlai.scripts.train -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128
  python -m parlai.scripts.train -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
import os
import signal
import json

from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.metrics import aggregate_task_reports
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.utils import Timer, round_sigfigs, warn_once
from parlai.core.logs import TensorboardLogger
from parlai.scripts.build_dict import build_dict, setup_args as setup_dict_args
from parlai.core.distributed_utils import (
    sync_object,
    is_primary_worker,
    all_gather_list,
    is_distributed,
    num_workers,
)
from parlai.scripts.build_pytorch_data import get_pyt_dict_file


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
    parser.add_pytorch_datateacher_args()
    train = parser.add_argument_group('Training Loop Arguments')
    train.add_argument(
        '-et',
        '--evaltask',
        help=(
            'task to use for valid/test (defaults to the '
            'one used for training if not set)'
        ),
    )
    train.add_argument(
        '--eval-batchsize',
        type=int,
        hidden=True,
        help='Eval time batch size (defaults to same as -bs)',
    )
    train.add_argument('--display-examples', type='bool', default=False, hidden=True)
    train.add_argument('-eps', '--num-epochs', type=float, default=-1)
    train.add_argument('-ttim', '--max-train-time', type=float, default=-1)
    train.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    train.add_argument(
        '-vtim',
        '--validation-every-n-secs',
        type=float,
        default=-1,
        help='Validate every n seconds. Saves model to model_file '
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
        help='max examples to use during validation (default ' '-1 uses all)',
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
        help='key into report table for selecting best ' 'validation',
    )
    train.add_argument(
        '-vmm',
        '--validation-metric-mode',
        type=str,
        choices=['max', 'min'],
        help='how to optimize validation metric (max or min)',
    )
    train.add_argument(
        '-vcut',
        '--validation-cutoff',
        type=float,
        default=1.0,
        hidden=True,
        help='value at which training will stop if exceeded by ' 'training metric',
    )
    train.add_argument(
        '-dbf',
        '--dict-build-first',
        hidden=True,
        type='bool',
        default=True,
        help='build dictionary first before training agent',
    )
    train.add_argument(
        '-lfc',
        '--load-from-checkpoint',
        type='bool',
        default=False,
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
        '-micro',
        '--aggregate-micro',
        type='bool',
        default=True,
        help='If multitasking, average metrics over the number of examples. '
             'If false, averages over the number of tasks.'
     )
    TensorboardLogger.add_cmdline_args(parser)
    parser = setup_dict_args(parser)
    return parser


def _maybe_load_eval_worlds(agent, opt, datatype):
    if not is_primary_worker():
        # only need the validation on the main worker
        return None
    return load_eval_worlds(agent, opt, datatype)


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
    if opt.get('pytorch_teacher_task'):
        # never use pytorch teachers for evaluation
        # but don't forget what we were normally using
        opt['task'] = opt['pytorch_teacher_task']
        del opt['pytorch_teacher_task']
    if opt.get('evaltask'):
        # if a different eval task is specified, use it.
        opt['task'] = opt['evaltask']
    if opt.get('eval_batchsize'):
        # override eval time batchsize
        opt['batchsize'] = opt['eval_batchsize']

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


def _run_single_eval(opt, valid_world, max_exs):
    # run evaluation on a single world
    valid_world.reset()

    cnt = 0
    max_cnt = max_exs if max_exs > 0 else float('inf')
    while not valid_world.epoch_done() and cnt < max_cnt:
        valid_world.parley()
        if cnt == 0 and opt['display_examples']:
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        cnt += valid_world.opt['batchsize']

    valid_report = valid_world.report()
    valid_world.reset()  # make sure world doesn't remember valid data

    return valid_report


def run_eval(valid_worlds, opt, datatype, max_exs=-1, write_log=False):
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
    if valid_worlds is None:
        # This isn't the primary worker, so we can just skip evaluation
        return None

    print('[ running eval: ' + datatype + ' ]')
    reports = []
    for v_world in valid_worlds:
        task_report = _run_single_eval(opt, v_world,
                                       max_exs / len(valid_worlds))
        reports.append(task_report)

    tasks = [world.opt['task'] for world in valid_worlds]
    report = aggregate_task_reports(reports, tasks,
                                    micro=opt.get('aggregate_micro', True))

    metrics = '{}:{}'.format(datatype, report)
    print(metrics)

    # write to file
    if write_log and opt.get('model_file'):
        # Write out metrics
        f = open(opt['model_file'] + '.' + datatype, 'a+')
        f.write(metrics + '\n')
        f.close()

    return report


def _save_best_valid(model_file, best_valid):
    """Save the best validation score to disk."""
    f = open(model_file + '.best_valid', 'w')
    f.write(str(best_valid))
    f.close()


class TrainLoop():
    """TrainLoop contains the core training loop logic."""

    def __init__(self, opt):
        # if python is called from a non-interactive shell, like a bash script,
        # it will by-default ignore SIGINTs, and KeyboardInterrupt exceptions are
        # not produced. This line brings them back
        signal.signal(signal.SIGINT, signal.default_int_handler)

        if isinstance(opt, ParlaiParser):
            print('[ Deprecated Warning: TrainLoop should be passed opt not Parser ]')
            opt = opt.parse_args()
        # Possibly load from checkpoint
        trainstats_suffix = '.trainstats'  # we might load training statistics from here
        if (
            opt['load_from_checkpoint']
            and opt.get('model_file')
            and os.path.isfile(opt['model_file'] + '.checkpoint')
        ):
            opt['init_model'] = opt['model_file'] + '.checkpoint'
            trainstats_suffix = '.checkpoint.trainstats'
        # Possibly build a dictionary (not all models do this).
        if opt['dict_build_first'] and not (
            opt.get('dict_file') or opt.get('model_file')
        ):
            raise RuntimeError(
                'WARNING: For train_model, please specify either a '
                'model_file or dict_file.'
            )
        if opt['dict_build_first'] and 'dict_file' in opt:
            # If data built via pytorch data teacher, we need to load prebuilt dict
            if opt.get('pytorch_teacher_task'):
                opt['dict_file'] = get_pyt_dict_file(opt)
            elif opt['dict_file'] is None and opt.get('model_file'):
                opt['dict_file'] = opt['model_file'] + '.dict'
            print("[ building dictionary first... ]")
            build_dict(opt, skip_if_built=True)
        # Create model and assign it to the specified task
        self.agent = create_agent(opt)
        self.world = create_task(opt, self.agent)
        # set up timers
        self.train_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()
        self.save_time = Timer()
        print('[ training... ]')
        self.parleys = 0
        self.max_num_epochs = (
            opt['num_epochs'] if opt['num_epochs'] > 0 else float('inf')
        )
        self.max_train_time = (
            opt['max_train_time'] if opt['max_train_time'] > 0 else float('inf')
        )
        self.log_every_n_secs = (
            opt['log_every_n_secs'] if opt['log_every_n_secs'] > 0 else float('inf')
        )
        self.val_every_n_secs = (
            opt['validation_every_n_secs']
            if opt['validation_every_n_secs'] > 0
            else float('inf')
        )
        self.save_every_n_secs = (
            opt['save_every_n_secs'] if opt['save_every_n_secs'] > 0 else float('inf')
        )
        self.val_every_n_epochs = (
            opt['validation_every_n_epochs']
            if opt['validation_every_n_epochs'] > 0
            else float('inf')
        )

        # smart defaults for --validation-metric-mode
        if opt['validation_metric'] in {'loss', 'ppl', 'mean_rank'}:
            opt['validation_metric_mode'] = 'min'
        elif opt['validation_metric'] in {'accuracy', 'hits@1', 'hits@5', 'f1', 'bleu'}:
            opt['validation_metric_mode'] = 'max'
        if opt.get('validation_metric_mode') is None:
            opt['validation_metric_mode'] = 'max'

        self.last_valid_epoch = 0
        self.valid_optim = 1 if opt['validation_metric_mode'] == 'max' else -1
        self.valid_reports = []
        self.best_valid = None
        if opt.get('model_file') and os.path.isfile(opt['model_file'] + '.best_valid'):
            with open(opt['model_file'] + ".best_valid", 'r') as f:
                x = f.readline()
                self.best_valid = float(x)
                f.close()
        self.impatience = 0
        self.saved = False
        self.valid_worlds = None
        self.opt = opt

        # we may have been preempted, make sure we note that amount
        self._preempted_epochs = 0.0
        if opt.get('model_file') and os.path.isfile(
            opt['model_file'] + trainstats_suffix
        ):
            # looks like we were preempted. make sure we load up our total
            # training stats, etc
            with open(opt['model_file'] + trainstats_suffix) as ts:
                obj = json.load(ts)
                self._preempted_epochs = obj.get('total_epochs', 0)
                self.train_time.total = obj.get('train_time', 0)
                self.impatience = obj.get('impatience', 0)
                self.valid_reports = obj.get('valid_reports', [])

        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)

    def save_model(self, suffix=None):
        """Save the model to disk, possibly with a suffix."""
        if not is_primary_worker():
            # never do IO as a non-primary worker
            return
        if not self.opt.get('model_file'):
            # nothing to save to, just exit
            return

        fn = self.opt['model_file']
        if suffix:
            fn += suffix
        while True:
            # don't ever let a ctrl-c interrupt saving
            try:
                self.agent.save(fn)
                self._save_train_stats(suffix)
                break
            except KeyboardInterrupt:
                pass

    def _save_train_stats(self, suffix=None):
        fn = self.opt['model_file']
        if suffix:
            fn += suffix
        fn += '.trainstats'
        with open(fn, 'w') as f:
            json.dump(
                {
                    'train_time': self.train_time.time(),
                    'total_epochs': (
                        self._preempted_epochs
                        + num_workers() * self.world.get_total_epochs()
                    ),
                    'impatience': self.impatience,
                    'valid_reports': self.valid_reports,
                },
                f,
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
            self.valid_worlds = _maybe_load_eval_worlds(self.agent, opt, 'valid')

        # run evaluation on valid set
        valid_report = sync_object(
            run_eval(self.valid_worlds, opt, 'valid', opt['validation_max_exs'])
        )
        v = valid_report.copy()
        v['train_time'] = self.train_time.time()
        self.valid_reports.append(v)
        # logging
        if opt['tensorboard_log'] is True and is_primary_worker():
            self.writer.add_metrics('valid', int(self.train_time.time()), valid_report)
        # saving
        if (
            opt.get('model_file')
            and opt.get('save_after_valid')
            and is_primary_worker()
        ):
            print("[ saving model checkpoint: " + opt['model_file'] + ".checkpoint ]")
            self.save_model('.checkpoint')

        # send valid metrics to agent if the agent wants them
        if hasattr(self.agent, 'receive_metrics'):
            self.agent.receive_metrics(valid_report)

        # check which metric to look at
        if 'tasks' in valid_report and '/' in opt['validation_metric']:
            # if you are multitasking and want your validation metric to be
            # a metric specific to a subtask, specify your validation metric
            # as -vmt subtask/metric
            subtask = opt['validation_metric'].split('/')[0]
            validation_metric = opt['validation_metric'].split('/')[1]
            new_valid = valid_report['tasks'][subtask][validation_metric]
        else:
            new_valid = valid_report[opt['validation_metric']]

        # check if this is the best validation so far
        if (
            self.best_valid is None
            or self.valid_optim * new_valid > self.valid_optim * self.best_valid
        ):
            print(
                '[ new best {}: {}{} ]'.format(
                    opt['validation_metric'],
                    new_valid,
                    ' (previous best was {})'.format(self.best_valid)
                    if self.best_valid is not None
                    else '',
                )
            )
            self.best_valid = new_valid
            self.impatience = 0
            if opt.get('model_file') and is_primary_worker():
                print("[ saving best valid model: " + opt['model_file'] + " ]")
                self.save_model()
                print(
                    "[ saving best valid metric: " + opt['model_file'] + ".best_valid ]"
                )
                _save_best_valid(opt['model_file'], self.best_valid)
                self.saved = True
            if (
                opt['validation_metric'] == 'accuracy'
                and self.best_valid >= opt['validation_cutoff']
            ):
                print('[ task solved! stopping. ]')
                return True
        else:
            self.impatience += 1
            print(
                '[ did not beat best {}: {} impatience: {} ]'.format(
                    opt['validation_metric'], round(self.best_valid, 4), self.impatience
                )
            )
        self.validate_time.reset()

        # check if we are out of patience
        if (
            opt['validation_patience'] > 0
            and self.impatience >= opt['validation_patience']
        ):
            print('[ ran out of patience! stopping training. ]')
            return True
        return False

    def _average_dicts(self, all_versions):
        # instead of a list-of-dicts with like keys, make a dict-of-lists with
        # keys to reduce
        to_reduce = {}
        for d in all_versions:
            for k, v in d.items():
                to_reduce.setdefault(k, []).append(v)
        # now perform the reduction
        finalized = {}
        for k, values in to_reduce.items():
            if k == 'exs' or k == 'total_skipped_batches':
                # sum across workers
                finalized[k] = np.sum(values)
            elif isinstance(values[0], dict):
                # do the same procedure recursively
                finalized[k] = self._average_dicts(values)
            else:
                # all other cases, take the mean across the workers
                finalized[k] = np.mean(values)
        return finalized

    def _cleanup_inaccurate_metrics(self, metrics):
        """
        Remove inaccurate multiworld metrics.

        When training in multitask mode, agent-level metrics may be shown,
        but are actually averages
        not distinguished across the worlds. This method adds a warning.

        Issue: https://github.com/facebookresearch/ParlAI/issues/1750
        """
        # TODO: fix the root issue
        if 'tasks' in metrics:
            metrics[
                'warning'
            ] = 'agent level metrics (e.g. loss, mean_loss, ppl) are averaged over tasks'

    def _sync_training_metrics(self, metrics):
        """
        Sync training metrics across workers.

        A handful of special cases are handled as exceptions, and the remaining
        metrics are simply averaged across workers.
        """
        if not is_distributed():
            # nothing special needed
            return metrics
        all_versions = all_gather_list(metrics)
        return self._average_dicts(all_versions)

    def _nice_format(self, dictionary):
        rounded = {}
        for k, v in dictionary.items():
            if isinstance(v, dict):
                rounded[k] = self._nice_format(v)
            elif isinstance(v, float):
                rounded[k] = round_sigfigs(v, 4)
            else:
                rounded[k] = v
        return rounded

    def _compute_eta(self, epochs_completed, time_elapsed):
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

        return eta

    def log(self):
        """Output a training log entry."""
        opt = self.opt
        if opt['display_examples']:
            print(self.world.display() + '\n~~')
        logs = []
        # get report
        train_report = self.world.report()
        self._cleanup_inaccurate_metrics(train_report)
        train_report = self._sync_training_metrics(train_report)
        self.world.reset_metrics()

        # time elapsed
        logs.append('time:{}s'.format(np.floor(self.train_time.time())))
        logs.append('total_exs:{}'.format(self._total_exs))

        if self._total_epochs >= 0:
            # only if it's unbounded
            logs.append('epochs:{}'.format(round(self._total_epochs, 2)))

        time_left = self._compute_eta(self._total_epochs, self.train_time.time())
        if time_left is not None:
            logs.append('time_left:{}s'.format(max(0, np.ceil(time_left))))

        log = '[ {} ] {}'.format(' '.join(logs), self._nice_format(train_report))
        print(log)
        self.log_time.reset()

        if opt['tensorboard_log'] is True and is_primary_worker():
            self.writer.add_metrics('train', self._total_exs, train_report)

    def train(self):
        """
        Perform a training run.

        :return: tuple of reports (validation_report, test_report)
        """
        if is_distributed():
            warn_once(
                "Distributed training outputs average-per-worker metrics during "
                "training, and may be slightly distorted. Validation/test are "
                "unadulterated."
            )
        opt = self.opt
        world = self.world
        with world:
            while True:
                # do one example / batch of examples
                world.parley()
                self.parleys += 1

                # get the total training examples done, compute epochs
                self._total_epochs = (
                    self._preempted_epochs
                    + num_workers() * self.world.get_total_epochs()
                )
                exs_per_epoch = self.world.num_examples()
                self._total_exs = int(np.round(self._total_epochs * exs_per_epoch))

                # and use the primary worker's timings for everything
                train_time, log_time, validate_time = sync_object(
                    (
                        self.train_time.time(),
                        self.log_time.time(),
                        self.validate_time.time(),
                    )
                )

                # check counters and timers
                if self._total_epochs >= self.max_num_epochs:
                    self.log()
                    print(
                        '[ num_epochs completed:{} time elapsed:{}s ]'.format(
                            self.max_num_epochs, train_time
                        )
                    )
                    break
                if train_time > self.max_train_time:
                    print('[ max_train_time elapsed:{}s ]'.format(train_time))
                    break
                if log_time > self.log_every_n_secs:
                    self.log()
                if (
                    validate_time > self.val_every_n_secs
                    or self._total_epochs - self.last_valid_epoch
                    >= self.val_every_n_epochs
                ):
                    stop_training = self.validate()
                    self.last_valid_epoch = self._total_epochs
                    if stop_training:
                        break
                if (
                    self.save_time.time() > self.save_every_n_secs
                    and opt.get('model_file')
                    and is_primary_worker()
                ):
                    print(
                        "[ saving model checkpoint: {}.checkpoint".format(
                            opt['model_file']
                        )
                    )
                    self.save_model('.checkpoint')
                    self.save_time.reset()

        if not self.saved and is_primary_worker():
            # save agent
            self.save_model()
        elif opt.get('model_file'):
            # reload best validation model
            self.agent = create_agent(opt)

        valid_worlds = _maybe_load_eval_worlds(self.agent, opt, 'valid')
        max_exs = opt['validation_max_exs'] if opt.get('short_final_eval') else -1
        v_report = run_eval(valid_worlds, opt, 'valid', max_exs, write_log=True)
        test_worlds = _maybe_load_eval_worlds(self.agent, opt, 'test')
        t_report = run_eval(test_worlds, opt, 'test', max_exs, write_log=True)
        if valid_worlds:
            for valid_world in valid_worlds:
                valid_world.shutdown()
        if test_worlds:
            for test_world in test_worlds:
                test_world.shutdown()

        print_announcements(opt)

        return v_report, t_report


if __name__ == '__main__':
    TrainLoop(setup_args().parse_args()).train()
    print()
