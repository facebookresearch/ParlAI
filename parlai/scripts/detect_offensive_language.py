#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which iterates through the tasks specified and checks them for offensive
language.

## Examples

```shell
parlai detect_offensive_language -t "convai_chitchat" --display-examples True
```
"""  # noqa: E501
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
from parlai.utils.misc import TimeLogger
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Check task for offensive language')
    # Get command line arguments
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--safety',
        type=str,
        default='all',
        choices={'string_matcher', 'classifier', 'all'},
        help='Type of safety detector to apply to messages',
    )
    parser.set_defaults(datatype='train:ordered')
    parser.set_defaults(model='repeat_query')
    return parser


def detect(opt):
    """
    Checks a task for offensive language.
    """
    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)
    agent.opt.log()
    if opt['safety'] == 'string_matcher' or opt['safety'] == 'all':
        offensive_string_matcher = OffensiveStringMatcher()
    if opt['safety'] == 'classifier' or opt['safety'] == 'all':
        offensive_classifier = OffensiveLanguageClassifier()

    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    stats = {
        'bad_words': [],
        'bad_words_cnt': 0,
        'string_offensive': 0,
        'classifier_offensive': 0,
        'total_offensive': 0,
        'total': 0,
    }

    def report(world, stats):
        report = world.report()
        log = {
            'word_offenses': stats['bad_words_cnt'],
            'classifier_offenses%': 100
            * (stats['classifier_offensive'] / stats['total']),
            'string_offenses%': 100 * (stats['string_offensive'] / stats['total']),
            'total_offenses%': 100 * (stats['total_offensive'] / stats['total']),
        }
        text, log = log_time.log(report['exs'], world.num_examples(), log)
        logging.info(text)
        return log

    def classify(text, stats):
        offensive = False
        stats['total'] += 1
        if opt['safety'] == 'string_matcher' or opt['safety'] == 'all':
            bad_words = offensive_string_matcher.contains_offensive_language(text)
            if bad_words:
                stats['string_offensive'] += 1
                offensive = True
                stats['bad_words'].append(bad_words)
        if opt['safety'] == 'classifier' or opt['safety'] == 'all':
            if text in offensive_classifier:
                stats['classifier_offensive'] += 1
                offensive = True
        if offensive:
            stats['total_offensive'] += 1

    while not world.epoch_done():
        world.parley()
        stats['bad_words'] = []
        for a in world.acts:
            text = a.get('text', '')
            classify(text, stats)
            labels = a.get('labels', a.get('eval_labels', ''))
            for l in labels:
                classify(l, stats)
        if len(stats['bad_words']) > 0 and opt['display_examples']:
            logging.info(world.display())
            logging.info(
                "Offensive words detected: {}".format(', '.join(stats['bad_words']))
            )
        stats['bad_words_cnt'] += len(stats['bad_words'])
        if log_time.time() > log_every_n_secs:
            report(world, stats)

    if world.epoch_done():
        logging.info("epoch done")
    return report(world, stats)


@register_script('detect_offensive', hidden=True)
class DetectOffensive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return detect(self.opt)


if __name__ == '__main__':
    DetectOffensive.main()
