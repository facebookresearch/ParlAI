from parlai.scripts.interactive import setup_args, interactive

import random

if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()

    parser.set_params(
        batchsize=1,
        beam_size=20,
        beam_min_n_best=10,
    )

    interactive(parser.parse_args(print_args=False), print_parser=parser)
