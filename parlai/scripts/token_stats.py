#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from parlai.core.script import ParlaiScript, register_script
from parlai.core.agents import create_agent
from parlai.core.torch_agent import TorchAgent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.utils.misc import TimeLogger, nice_report
import parlai.utils.logging as logging


@register_script("token_stats", hidden=True)
class TokenStats(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, True, description='Compute tokenized stats.')
        parser.add_argument('--num-examples', '-n', type=int, default=-1)
        parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=10)
        parser.add_argument('--field', default='text')
        parser.add_argument('--final-only', type='bool', default=False)
        parser.set_defaults(
            no_cuda=True, model='test_agents/null', datatype='train:stream:ordered'
        )
        return parser

    def _compute_stats(self, lengths):
        lengths = np.array(lengths)
        return {
            "exs": len(lengths),
            "min": np.min(lengths),
            "max": np.max(lengths),
            "mean": np.median(lengths),
            "p01": np.quantile(lengths, 0.05),
            "p05": np.quantile(lengths, 0.05),
            "p10": np.quantile(lengths, 0.10),
            "p25": np.quantile(lengths, 0.25),
            "p50": np.quantile(lengths, 0.50),
            "p75": np.quantile(lengths, 0.75),
            "p90": np.quantile(lengths, 0.90),
            "p95": np.quantile(lengths, 0.95),
            "p99": np.quantile(lengths, 0.99),
            "p@128": np.mean(lengths <= 128),
        }

    def run(self):
        self.opt['no_cuda'] = True
        if 'ordered' not in self.opt['datatype'] and 'train' in self.opt['datatype']:
            self.opt['datatype'] = self.opt['datatype'] + ':ordered'
        agent = create_agent(self.opt)
        agent.opt.log()
        num_examples = self.opt['num_examples']
        field = self.opt['field'] + '_vec'
        if num_examples < 0:
            num_examples = float('inf')
        assert self.opt['batchsize'] == 1
        assert isinstance(agent, TorchAgent)

        world = create_task(self.opt, agent)
        teacher = world.get_task_agent()

        # set up logging
        log_every_n_secs = self.opt.get('log_every_n_secs', -1)
        if log_every_n_secs <= 0:
            log_every_n_secs = float('inf')
        log_time = TimeLogger()

        lengths = []

        cnt = 0
        total = min(teacher.num_examples(), num_examples)
        while not teacher.epoch_done() and cnt < num_examples:
            act = teacher.act()
            processed = agent.observe(act)
            text_vec = processed[field]
            if text_vec is not None and (
                not self.opt['final_only'] or act.get('episode_done')
            ):
                cnt += 1
                lengths.append(float(len(text_vec)))
            agent.self_observe({})

            if log_time.time() > log_every_n_secs:
                report = self._compute_stats(lengths)
                text, report = log_time.log(report['exs'], total, report)
                logging.info(text)

        report = self._compute_stats(lengths)
        print(nice_report(report))
        return report


if __name__ == '__main__':
    TokenStats.main()
