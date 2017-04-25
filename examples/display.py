# Copyright 2004-present Facebook. All Rights Reserved.
"""Basic example which iterates through the tasks specified and prints them out.
Used for verification of data loading and iteration.

For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:
`python examples/display_data.py -t babi:task1k:1`
"""

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

import random

def main():
    random.seed(42)

    # Get command line arguments
    opt = ParlaiParser(True, True).parse_args()
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs.
    with world:
        # show at most 100 exs
        for k in range(100):
            world.parley()
            print(world.display() + "\n~~")
            if k > 10 and world.episode_done():
                # break out at the end of an episode if at least 10 exs shown
                break

if __name__ == '__main__':
    main()
