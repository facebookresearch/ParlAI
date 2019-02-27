#!/usr/bin/env python3

"""Basic example which trains model_smoothing on the tasks 
specified with the given model on them.
For more documentation, see parlai.scripts.eval_model.
"""
from parlai.scripts.train_lambda import setup_args, train_lambda


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args(print_args=False)
    train_lambda(opt, print_parser=parser)