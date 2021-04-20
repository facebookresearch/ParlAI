#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Verify data doesn't have basic mistakes, like empty text fields or empty label
candidates.

## Examples

```shell
parlai verify_data --task convai2 --datatype valid
```
"""
from collections import Counter
import copy
from typing import Iterable, List, Tuple

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.message import History, Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.misc import TimeLogger, warn_once
from parlai.core.worlds import World, create_task
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Check tasks for common errors')
    # Get command line arguments
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--verify-task',
        type=bool,
        default=False,
        help=(
            'In addition to verifying the specified --datatype, also verifies the '
            'task holistically (looks for issues across train, valid, and test).'
        ),
    )
    parser.add_argument(
        '--datatypes',
        type=str,
        default='train,valid,test',
        help=(
            'Comma separated list of datatype splits to consider in holistic verification. '
            'Only applicable if --verify-task True.'
        ),
    )
    parser.set_defaults(datatype='train:stream:ordered')
    return parser


def report(world, counts, log_time):
    report = world.report()
    log = {
        'missing_text': counts['missing_text'],
        'missing_labels': counts['missing_labels'],
        'missing_label_candidates': counts['missing_label_candidates'],
        'empty_string_label_candidates': counts['empty_string_label_candidates'],
        'label_candidates_with_missing_label': counts[
            'label_candidates_with_missing_label'
        ],
        'did_not_return_message': counts['did_not_return_message'],
    }
    text, log = log_time.log(report['exs'], world.num_examples(), log)
    return text, log


def warn(txt, act, opt):
    if opt.get('display_examples'):
        print(txt + ":\n" + str(act))
    else:
        warn_once(txt)


def verify_split(opt):
    if opt['datatype'] == 'train':
        logging.warning("changing datatype from train to train:ordered")
        opt['datatype'] = 'train:ordered'
    opt.log()
    world = task_world(opt)

    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    counts = {}
    counts['missing_text'] = 0
    counts['missing_labels'] = 0
    counts['missing_label_candidates'] = 0
    counts['empty_string_label_candidates'] = 0
    counts['label_candidates_with_missing_label'] = 0
    counts['did_not_return_message'] = 0

    for act, *_ in task_world_iterator(world):
        if not isinstance(act, Message):
            counts['did_not_return_message'] += 1

        if 'text' not in act and 'image' not in act:
            warn("warning: missing text field:\n", act, opt)
            counts['missing_text'] += 1

        if 'labels' not in act and 'eval_labels' not in act:
            warn("warning: missing labels/eval_labels field:\n", act, opt)
            counts['missing_labels'] += 1
        else:
            if 'label_candidates' not in act:
                counts['missing_label_candidates'] += 1
            else:
                labels = act.get('labels', act.get('eval_labels'))
                is_label_cand = {}
                for l in labels:
                    is_label_cand[l] = False
                for c in act['label_candidates']:
                    if c == '':
                        warn("warning: empty string label_candidate:\n", act, opt)
                        counts['empty_string_label_candidates'] += 1
                    if c in is_label_cand:
                        if is_label_cand[c] is True:
                            warn(
                                "warning: label mentioned twice in candidate_labels:\n",
                                act,
                                opt,
                            )
                        is_label_cand[c] = True
                for _, has in is_label_cand.items():
                    if has is False:
                        warn("warning: label missing in candidate_labels:\n", act, opt)
                        counts['label_candidates_with_missing_label'] += 1

        if log_time.time() > log_every_n_secs:
            text, log = report(world, counts, log_time)
            print(text)

    counts['exs'] = int(world.report()['exs'])
    return counts


def task_world(opt: Opt) -> World:
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    try:
        # print dataset size if available
        logging.info(
            f'Loaded {world.num_episodes()} episodes with a '
            f'total of {world.num_examples()} examples.'
        )
    except AttributeError:
        pass
    return world


def task_world_iterator(world: World) -> Iterable[Tuple[Message, Message, bool]]:
    while not world.epoch_done():
        world.parley()
        acts = world.get_acts()
        assert len(acts) == 2
        yield acts[0], acts[1], world.episode_done()


def _count_conversations(opt: Opt) -> Counter:
    convo_counter = Counter()
    history = History(opt)
    world = task_world(opt)
    for *acts, episode_done in task_world_iterator(world):
        for act in acts:
            history.update_history(act)
        if episode_done:
            convo_counter[str(history)] += 1
            history.reset()
    return convo_counter


def verify_task(opt: Opt) -> None:
    opt = copy.deepcopy(opt)

    def _dupes(counter: Counter) -> List[str]:
        dupes = []
        for convo, cnt in counter.most_common():
            if cnt > 1:
                dupes.append(convo)
            else:
                break
        return dupes

    convo_counts = Counter()
    for dt in opt['datatypes'].split(','):
        if dt == 'train':
            logging.info('changing datatype from train to train:ordered')
            dt = 'train:ordered'
        opt['datatype'] = dt
        split_convo_counts = _count_conversations(opt)
        split_convo_dupes = _dupes(split_convo_counts)
        log_func = logging.warn if len(split_convo_dupes) > 0 else logging.info
        log_func(f"{len(split_convo_dupes)} duplicate conversations found in {dt}.")
        convo_counts += split_convo_counts

    convo_dupes = _dupes(convo_counts)
    log_func = logging.warn if len(convo_dupes) > 0 else logging.info
    log_func(
        f"Found {len(convo_dupes)} duplicate conversations across train/valid/test splits."
    )


def verify_data(opt):
    counts = verify_split(opt)
    print(counts)
    if opt['verify_task']:
        verify_task(opt)
    return counts


@register_script('verify_data', hidden=True)
class VerifyData(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return verify_data(self.opt)


if __name__ == '__main__':
    VerifyData.main()
