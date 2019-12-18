#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Creates supplementary training examples using a trained model and unused train file.

Input:
    model: a trained agent
    deploy: an unused train file (labels available)
    flags: (settings)
Returns:
    supp: a file in parlai_meta format with new training examples


Examples
--------

.. code-block:: shell

  python create_synth_supp.py
  --model-file /path/to/model-file.model
  --deploy-file /path/to/deploy-file.txt
  --supp-file /path/to/supp-file.txt
  --conversion-rate 0.9
  --conversion-acc 0.9
"""
import json
import os
import random

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

from parlai.projects.self_feeding.utils import Parley

PARLAIHOME = os.environ['PARLAIHOME']
NUM_INLINE_CANDS = 20


def setup_args(parser=None):
    # Get command line arguments
    if parser is None:
        parser = ParlaiParser(True, True, 'Display data from a task')
    parser.add_argument(
        '-mf',
        '--model-file',
        type=str,
        default='/private/home/bhancock/metadialog/models/unknown.mdl',
    )
    parser.add_argument('-dfile', '--deploy-file', type=str, default='train_c')
    parser.add_argument('-sfile', '--supp-file', type=str, default='supp_c')
    parser.add_argument(
        '-cr',
        '--conversion-rate',
        type=float,
        default=1.0,
        help="The fraction of misses converted into new training data",
    )
    parser.add_argument(
        '-ca',
        '--conversion-acc',
        type=float,
        default=1.0,
        help="The fraction of converted data that have a correct label",
    )
    parser.set_defaults(bs=1)
    parser.set_defaults(ecands='inline')
    parser.set_defaults(datatype='valid')
    return parser


def create_supp(opt):
    """
    Evaluates a model.

    :param opt: tells the evaluation function how to run
    :param bool print_parser: if provided, prints the options that are set within the
        model after loading the model
    :return: the final result of calling report()
    """
    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    # Extract supp examples from misses on deploy set
    num_seen = 0
    num_misses = 0
    num_supp = 0
    num_supp_correct = 0
    examples = []
    while not world.epoch_done():
        world.parley()
        # Examples are considered one at a time
        num_seen += 1
        if num_seen % 1000 == 0:
            print(f"{num_seen}/{world.num_examples()}")
        report = world.report()
        if report['accuracy'] < 1.0:
            # Example is a miss (i.e., model got it wrong)
            num_misses += 1
            if random.random() < opt['conversion_rate']:
                # Example will be converted (e.g., bot recognized mistake and asked)
                num_supp += 1
                texts = world.acts[0]['text'].split('\n')
                context = texts[-1]
                memories = texts[:-1]
                candidates = world.acts[0]['label_candidates']
                # Reward of 1 indicates positive, -1 indicates negative (for training)
                # For now, we only train with positives, and the reward field is unused
                reward = 1

                if random.random() < opt['conversion_acc']:
                    # Example will be converted correctly (e.g., good user response)
                    num_supp_correct += 1
                    response = world.acts[0]['eval_labels'][0]
                else:
                    # Example will be converted incorrectly (e.g., bad user response)
                    response = random.choice(
                        world.acts[0]['label_candidates'][: NUM_INLINE_CANDS - 1]
                    )

                example = Parley(context, response, reward, candidates, memories)
                examples.append(example)
        world.reset_metrics()

    print("EPOCH DONE")
    print(f"Model file: {opt['model_file']}")
    print(f"Deploy file: {opt['task']}")
    print(f"Supp file: {opt['outfile']}")
    print(f"Deploy size (# examples seen): {num_seen}")
    print(f"Supp size (# examples converted): {num_supp}")

    acc = 1 - (num_misses / num_seen)
    print(f"Accuracy (% of deploy): {acc * 100:.1f}% ({num_misses} misses)")
    print(
        f"Conversion rate (% of misses): {num_supp/num_misses * 100:.2f}% "
        f"({num_supp}/{num_misses})"
    )
    print(
        f"Conversion acc (% of converted): {num_supp_correct/num_supp * 100:.2f}% "
        f"({num_supp_correct}/{num_supp})"
    )

    with open(opt['outfile'], 'w') as outfile:
        for ex in examples:
            outfile.write(json.dumps(ex.to_dict()) + '\n')


if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    opt = parser.parse_args()
    opt['dict_file'] = os.path.splitext(opt['model_file'])[0] + '.dict'
    opt['task'] = 'self_feeding:convai2:' + opt['deploy_file']
    opt['outfile'] = PARLAIHOME + '/data/convai2meta/' + opt['supp_file'] + '.txt'
    create_supp(opt)
