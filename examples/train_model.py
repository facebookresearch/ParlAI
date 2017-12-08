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

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
import build_dict
import math

def setup_args():
    parser = ParlaiParser(True, True)
    train = parser.add_argument_group('Training Loop Arguments')
    train.add_argument('-et', '--evaltask',
                       help=('task to use for valid/test (defaults to the '
                             'one used for training if not set)'))
    train.add_argument('-d', '--display-examples',
                       type='bool', default=False)
    train.add_argument('-e', '--num-epochs', type=float, default=-1)
    train.add_argument('-ttim', '--max-train-time',
                       type=float, default=-1)
    train.add_argument('-ltim', '--log-every-n-secs',
                       type=float, default=2)
    train.add_argument('-vtim', '--validation-every-n-secs',
                       type=float, default=-1)
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
    train.add_argument('-vcut', '--validation-cutoff',
                       type=float, default=1.0,
                       help='value at which training will stop if exceeded by '
                            'training metric')
    train.add_argument('-dbf', '--dict-build-first',
                       type='bool', default=True,
                       help='build dictionary first before training agent')
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
    opt['datatype'] = datatype
    if opt.get('evaltask'):
        opt['task'] = opt['evaltask']

    if valid_world is None:
        valid_world = create_task(opt, agent)
    else:
        valid_world.reset()
    cnt = 0
    while not valid_world.epoch_done():
        valid_world.parley()
        if cnt == 0 and opt['display_examples']:
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        cnt += opt['batchsize']
        if max_exs > 0 and cnt >= max_exs:
            # note this max_exs is approximate--some batches won't always be
            # full depending on the structure of the data
            break
    valid_report = valid_world.report()

    metrics = datatype + ':' + str(valid_report)
    print(metrics)
    if write_log and opt['model_file']:
        # Write out metrics
        f = open(opt['model_file'] + '.' + datatype, 'a+')
        f.write(metrics + '\n')
        f.close()

    return valid_report, valid_world


class TrainLoop():
    def __init__(self, parser):
        opt = parser.parse_args()
        # Possibly build a dictionary (not all models do this).
        if opt['dict_build_first'] and 'dict_file' in opt:
            if opt['dict_file'] is None and opt.get('model_file'):
                opt['dict_file'] = opt['model_file'] + '.dict'
            print("[ building dictionary first... ]")
            build_dict.build_dict(opt)
        # Create model and assign it to the specified task
        self.agent = create_agent(opt)
        self.world = create_task(opt, self.agent)
        self.train_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()
        print('[ training... ]')
        self.parleys = 0
        self.total_exs = 0
        self.total_episodes = 0
        self.total_epochs = 0
        self.max_exs = None
        self.max_parleys = None
        self.world_num_exs = self.world.num_examples()
        if self.world_num_exs is not None:
            self.max_exs = opt['num_epochs'] * self.world_num_exs
            self.max_parleys = math.ceil(self.max_exs / opt['batchsize'])
        self.best_valid = 0
        self.impatience = 0
        self.saved = False
        self.valid_world = None
        self.opt = opt

    def validate(self):
        opt = self.opt
        valid_report, self.valid_world = run_eval(
            self.agent, opt, 'valid', opt['validation_max_exs'],
            valid_world=self.valid_world)
        if valid_report[opt['validation_metric']] > self.best_valid:
            self.best_valid = valid_report[opt['validation_metric']]
            self.impatience = 0
            print('[ new best {}: {} ]'.format(
                opt['validation_metric'], self.best_valid))
            self.world.save_agents()
            self.saved = True
            if opt['validation_metric'] == 'accuracy' and self.best_valid > opt['validation_cutoff']:
                print('[ task solved! stopping. ]')
                return True
        else:
            self.impatience += 1
            print('[ did not beat best {}: {} impatience: {} ]'.format(
                    opt['validation_metric'], round(self.best_valid, 4),
                    self.impatience))
        self.validate_time.reset()
        if opt['validation_patience'] > 0 and self.impatience >= opt['validation_patience']:
            print('[ ran out of patience! stopping training. ]')
            return True
        return False

    def log(self):
        opt = self.opt
        if opt['display_examples']:
            print(self.world.display() + '\n~~')
        logs = []
        # time elapsed
        logs.append('time:{}s'.format(math.floor(self.train_time.time())))
        logs.append('parleys:{}'.format(self.parleys))
        # get report and update total examples seen so far
        if hasattr(self.agent, 'report'):
            train_report = self.agent.report()
            self.agent.reset_metrics()
        else:
            train_report = self.world.report()
            self.world.reset_metrics()
        if hasattr(train_report, 'get') and train_report.get('total'):
            self.total_exs += train_report['total']
            logs.append('total_exs:{}'.format(self.total_exs))
        # check if we should log amount of time remaining
        time_left = None
        if (opt['num_epochs'] > 0 and self.total_exs > 0 and
                (self.max_exs is not None and self.max_exs > 0)):
            exs_per_sec = self.train_time.time() / self.total_exs
            time_left = (self.max_exs - self.total_exs) * exs_per_sec
        if opt['max_train_time'] > 0:
            other_time_left = opt['max_train_time'] - self.train_time.time()
            if time_left is not None:
                time_left = min(time_left, other_time_left)
            else:
                time_left = other_time_left
        if time_left is not None:
            logs.append('time_left:{}s'.format(math.floor(time_left)))
        if opt['num_epochs'] > 0:
            if (self.total_exs > 0 and
                    (self.world_num_exs is not None and self.world_num_exs > 0)):
                display_epochs = int(self.total_exs / self.world_num_exs)
            else:
                display_epochs = self.total_epochs
                logs.append('num_epochs:{}'.format(display_epochs))
        # join log string and add full metrics report to end of log
        log = '[ {} ] {}'.format(' '.join(logs), train_report)
        print(log)
        self.log_time.reset()


    def train(self):
        opt = self.opt
        world = self.world
        with world:
            while True:
                world.parley()
                self.parleys += 1
                if world.epoch_done():
                    self.total_epochs += 1

                if opt['num_epochs'] > 0 and self.max_parleys is not None and (
                    (self.max_parleys > 0 and self.parleys >= self.max_parleys)
                    or self.total_epochs >= opt['num_epochs']):
                    self.log()
                    print('[ num_epochs completed:{} time elapsed:{}s ]'.format(
                        opt['num_epochs'], self.train_time.time()))
                    break
                if opt['max_train_time'] > 0 and self.train_time.time() > opt['max_train_time']:
                    print('[ max_train_time elapsed:{}s ]'.format(self.train_time.time()))
                    break
                if opt['log_every_n_secs'] > 0 and self.log_time.time() > opt['log_every_n_secs']:
                    self.log()
                if (opt['validation_every_n_secs'] > 0 and
                        self.validate_time.time() > opt['validation_every_n_secs']):
                    stop_training = self.validate()
                    if stop_training:
                        break

        if not self.saved:
            # save agent
            world.save_agents()
        elif opt.get('model_file'):
            # reload best validation model
            self.agent = create_agent(opt)

        run_eval(self.agent, opt, 'valid', write_log=True)
        run_eval(self.agent, opt, 'test', write_log=True)


if __name__ == '__main__':
    TrainLoop(setup_args()).train()
