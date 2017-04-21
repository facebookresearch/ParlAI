# Copyright 2004-present Facebook. All Rights Reserved.
"""Generates a dictionary file from the training ."""

from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import DialogPartnerWorld
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

def main():
    # Get command line arguments
    argparser = ParlaiParser()
    DictionaryAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    dictionary = DictionaryAgent(opt)

    for datatype in ['train:ordered', 'valid']:
        # we use train and valid sets to build dictionary
        opt['datatype'] = datatype
        world = create_task(opt, dictionary)

        # pass examples to dictionary
        for _ in range(len(world)):
            world.parley()

    if 'dict_savepath' in opt:
        dictionary.save(opt['dict_savepath'])

if __name__ == '__main__':
    main()
