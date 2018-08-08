# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train a model.

After training, computes validation and test error.

Run with, e.g.:

python examples/train_model.py -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model

..or..

python examples/train_model.py -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128

..or..

python examples/train_model.py -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

TODO List:
- More logging (e.g. to files), make things prettier.
"""

from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
from parlai.core.logs import TensorboardLogger
from parlai.scripts.build_dict import build_dict, setup_args as setup_dict_args
import math
import os

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    train = parser.add_argument_group('Training Loop Arguments')
    train.add_argument('-et', '--evaltask',
                       help=('task to use for valid/test (defaults to the '
                             'one used for training if not set)'))
    train.add_argument('--display-examples', type='bool', default=False)
    train.add_argument('-eps', '--num-epochs', type=float, default=-1)
    train.add_argument('-ttim', '--max-train-time',
                       type=float, default=-1)
    train.add_argument('-ltim', '--log-every-n-secs',
                       type=float, default=2)
    train.add_argument('-vtim', '--validation-every-n-secs',
                       type=float, default=-1,
                       help='Validate every n seconds. Whenever the the best '
                            'validation metric is found, saves the model to '
                            'the model_file path if set.')
    train.add_argument('-stim', '--save-every-n-secs',
                       type=float, default=-1,
                       help='Saves the model to model_file.checkpoint after '
                            'every n seconds (default -1, never).')
    train.add_argument('-sval', '--save-after-valid', type='bool',
                       default=False,
                       help='Saves the model to model_file.checkpoint after '
                            'every validation (default %(default)s).')
    train.add_argument('-veps', '--validation-every-n-epochs',
                       type=float, default=-1,
                       help='Validate every n epochs. Whenever the the best '
                            'validation metric is found, saves the model to '
                            'the model_file path if set.')
    train.add_argument('-vme', '--validation-max-exs',
                       type=int, default=-1,
                       help='max examples to use during validation (default '
                            '-1 uses all)')
    train.add_argument('-vp', '--validation-patience',
                       type=int, default=10,
                       help=('number of iterations of validation where result'
                             ' does not improve before we stop training'))
    train.add_argument('-vmt', '--validation-metric', default='accuracy',
                       help='key into report table for selecting best '
                            'validation')
    train.add_argument('-vmm', '--validation-metric-mode', default='max',
                       type=str, choices=['max', 'min'],
                       help='how to optimize validation metric (max or min)')
    train.add_argument('-vcut', '--validation-cutoff',
                       type=float, default=1.0,
                       help='value at which training will stop if exceeded by '
                            'training metric')
    train.add_argument('-dbf', '--dict-build-first',
                       type='bool', default=True,
                       help='build dictionary first before training agent')
    train.add_argument('-lfc', '--load-from-checkpoint',
                       type='bool', default=False,
                       help='load model from checkpoint if available')
    train.add_argument('-vshare', '--validation-share-agent', default=False,
                       help='use a shared copy of the agent for validation. '
                            'this will eventually default to True, but '
                            'currently defaults to False.')
    TensorboardLogger.add_cmdline_args(parser)
    parser = setup_dict_args(parser)
    return parser

def run_eval(agent, opt, datatype, max_exs=-1, write_log=False, valid_world=None):
    """Eval on validation/test data.
    - Agent is the agent to use for the evaluation.
    - opt is the options that specific the task, eval_task, etc
    - datatype is the datatype to use, such as "valid" or "test"
    - write_log specifies to write metrics to file if the model_file is set
    - max_exs limits the number of examples if max_exs > 0
    - valid_world can be an existing world which will be reset instead of reinitialized
    """
    print('[ running eval: ' + datatype + ' ]')
    if 'stream' in opt['datatype']:
        datatype += ':stream'

    if valid_world is None:
        opt = opt.copy()
        opt['datatype'] = datatype
        if opt.get('evaltask'):
            opt['task'] = opt['evaltask']
        if opt.get('validation_share_agent', False):
            valid_agent = create_agent_from_shared(agent.share())
        else:
            valid_agent = agent
        valid_world = create_task(opt, valid_agent)
    valid_world.reset()
    cnt = 0
    while not valid_world.epoch_done():
        valid_world.parley()
        if cnt == 0 and opt['display_examples']:
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        cnt += opt['batchsize']
        if max_exs > 0 and cnt > max_exs + opt.get('numthreads', 1):
            # note this max_exs is approximate--some batches won't always be
            # full depending on the structure of the data
            break
    valid_report = valid_world.report()
    valid_world.reset()  # this makes sure agent doesn't remember valid data

    metrics = datatype + ':' + str(valid_report)
    print(metrics)
    if write_log and opt.get('model_file'):
        # Write out metrics
        f = open(opt['model_file'] + '.' + datatype, 'a+')
        f.write(metrics + '\n')
        f.close()

    return valid_report, valid_world

def save_best_valid(model_file, best_valid):
    f = open(model_file + '.best_valid', 'w')
    f.write(str(best_valid))
    f.close()


class TrainLoop():
    def __init__(self, opt):
        if isinstance(opt, ParlaiParser):
            print('[ Deprecated Warning: TrainLoop should be passed opt not Parser ]')
            opt = opt.parse_args()
        # Possibly load from checkpoint
        if opt['load_from_checkpoint'] and opt.get('model_file') and os.path.isfile(opt['model_file'] + '.checkpoint'):
            opt['init_model'] = opt['model_file'] + '.checkpoint'
        # Possibly build a dictionary (not all models do this).
        if opt['dict_build_first'] and 'dict_file' in opt:
            if opt['dict_file'] is None and opt.get('model_file'):
                opt['dict_file'] = opt['model_file'] + '.dict'
            print("[ building dictionary first... ]")
            build_dict(opt, skip_if_built=True)
        # Create model and assign it to the specified task
        self.agent = create_agent(opt)
        self.world = create_task(opt, self.agent)
        self.train_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()
        self.save_time = Timer()
        print('[ training... ]')
        self.parleys = 0
        self.max_num_epochs = opt['num_epochs'] if opt['num_epochs'] > 0 else float('inf')
        self.max_train_time = opt['max_train_time'] if opt['max_train_time'] > 0 else float('inf')
        self.log_every_n_secs = opt['log_every_n_secs'] if opt['log_every_n_secs'] > 0 else float('inf')
        self.val_every_n_secs = opt['validation_every_n_secs'] if opt['validation_every_n_secs'] > 0 else float('inf')
        self.save_every_n_secs = opt['save_every_n_secs'] if opt['save_every_n_secs'] > 0 else float('inf')
        self.val_every_n_epochs = opt['validation_every_n_epochs'] if opt['validation_every_n_epochs'] > 0 else float('inf')
        self.last_valid_epoch = 0
        self.valid_optim = 1 if opt['validation_metric_mode'] == 'max' else -1
        self.best_valid = None
        if opt.get('model_file') and os.path.isfile(opt['model_file'] + '.best_valid'):
            with open(opt['model_file'] + ".best_valid", 'r') as f:
                x = f.readline()
                self.best_valid = float(x)
                f.close()
        self.impatience = 0
        self.saved = False
        self.valid_world = None
        self.opt = opt
        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)

    def validate(self):
        opt = self.opt
        # run evaluation on valid set
        valid_report, self.valid_world = run_eval(
            self.agent, opt, 'valid', opt['validation_max_exs'],
            valid_world=self.valid_world)

        # logging
        if opt['tensorboard_log'] is True:
            self.writer.add_metrics('valid', int(math.floor(self.train_time.time())), valid_report)
        # saving
        if opt.get('model_file') and opt.get('save_after_valid'):
            print("[ saving model checkpoint: " + opt['model_file'] + ".checkpoint ]")
            self.agent.save(opt['model_file'] + '.checkpoint')

        # send valid metrics to agent if the agent wants them
        if hasattr(self.agent, 'receive_metrics'):
            self.agent.receive_metrics(valid_report)

        # check which metric to look at
        if '/' in opt['validation_metric']:
            # if you are multitasking and want your validation metric to be
            # a metric specific to a subtask, specify your validation metric
            # as -vmt subtask/metric
            subtask = opt['validation_metric'].split('/')[0]
            validation_metric = opt['validation_metric'].split('/')[1]
            new_valid = valid_report['tasks'][subtask][validation_metric]
        else:
            new_valid = valid_report[opt['validation_metric']]

        # check if this is the best validation so far
        if self.best_valid is None or self.valid_optim * new_valid > self.valid_optim * self.best_valid:
            print('[ new best {}: {}{} ]'.format(
                opt['validation_metric'], new_valid,
                ' (previous best was {})'.format(self.best_valid)
                    if self.best_valid is not None else ''))
            self.best_valid = new_valid
            self.impatience = 0
            if opt.get('model_file'):
                print("[ saving best valid model: " + opt['model_file'] + " ]")
                self.agent.save(opt['model_file'])
                print("[ saving best valid metric: " + opt['model_file'] + ".best_valid ]")
                save_best_valid(opt['model_file'], self.best_valid)
                self.saved = True
            if opt['validation_metric'] == 'accuracy' and self.best_valid >= opt['validation_cutoff']:
                print('[ task solved! stopping. ]')
                return True
        else:
            self.impatience += 1
            print('[ did not beat best {}: {} impatience: {} ]'.format(
                    opt['validation_metric'], round(self.best_valid, 4),
                    self.impatience))
        self.validate_time.reset()

        # check if we are out of patience
        if opt['validation_patience'] > 0 and self.impatience >= opt['validation_patience']:
            print('[ ran out of patience! stopping training. ]')
            return True
        return False

    def log(self):
        opt = self.opt
        if opt['display_examples']:
            print(self.world.display() + '\n~~')
        logs = []
        # get report
        train_report = self.world.report(compute_time=True)
        self.world.reset_metrics()

        # time elapsed
        logs.append('time:{}s'.format(math.floor(self.train_time.time())))
        total_exs = self.world.get_total_exs()
        logs.append('total_exs:{}'.format(total_exs))

        exs_per_ep = self.world.num_examples()
        if exs_per_ep:
            logs.append('epochs:{}'.format(
                round(total_exs / exs_per_ep, 2)))

        if 'time_left' in train_report:
            logs.append('time_left:{}s'.format(
                         math.floor(train_report.pop('time_left', ""))))

        log = '[ {} ] {}'.format(' '.join(logs), train_report)
        print(log)
        self.log_time.reset()

        if opt['tensorboard_log'] is True:
            self.writer.add_metrics('train', int(logs[1].split(":")[1]), train_report)

    def train(self):
        opt = self.opt
        world = self.world
        with world:
            while True:
                # do one example / batch of examples
                world.parley()
                self.parleys += 1

                # check counters and timers
                if world.get_total_epochs() >= self.max_num_epochs:
                    self.log()
                    print('[ num_epochs completed:{} time elapsed:{}s ]'.format(
                        self.max_num_epochs, self.train_time.time()))
                    break
                if self.train_time.time() > self.max_train_time:
                    print('[ max_train_time elapsed:{}s ]'.format(self.train_time.time()))
                    break
                if self.log_time.time() > self.log_every_n_secs:
                    self.log()
                if self.validate_time.time() > self.val_every_n_secs:
                    stop_training = self.validate()
                    if stop_training:
                        break
                if world.get_total_epochs() - self.last_valid_epoch >= self.val_every_n_epochs:
                    stop_training = self.validate()
                    self.last_valid_epoch = world.get_total_epochs()
                    if stop_training:
                        break
                if self.save_time.time() > self.save_every_n_secs and opt.get('model_file'):
                    print("[ saving model checkpoint: " + opt['model_file'] + ".checkpoint ]")
                    self.agent.save(opt['model_file'] + '.checkpoint')
                    self.save_time.reset()

        if not self.saved:
            # save agent
            self.agent.save(opt['model_file'])
        elif opt.get('model_file'):
            # reload best validation model
            self.agent = create_agent(opt)

        v_report, v_world = run_eval(self.agent, opt, 'valid', write_log=True)
        t_report, t_world = run_eval(self.agent, opt, 'test', write_log=True)
        v_world.shutdown()
        t_world.shutdown()
        return v_report, t_report


if __name__ == '__main__':
    TrainLoop(setup_args().parse_args()).train()
    print()
