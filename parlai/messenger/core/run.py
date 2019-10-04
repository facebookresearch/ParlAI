#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.messenger.core.messenger_manager import MessengerManager
import importlib
import shared_utils as utils

def setup_args():
    parser = ParlaiParser(False, False)
    parser.add_parlai_data_path()
    parser.add_messenger_args()
    return parser.parse_args()


def run(opt):
    manager = MessengerManager(opt)
    manager.setup_server()
    manager.init_new_state()
    manager.setup_socket()
    manager.start_new_run()
    try:
        manager.start_task()
    except BaseException:
        manager.shutdown()
        raise
    except:
        manager.shutdown()
    finally:
        print('finally')
        manager.shutdown()



if __name__ == '__main__':
    opt = setup_args()
    config_path = opt.get('config_path')
    config = utils.parse_configuration_file(config_path)
    opt.update(config['world_opt'])
    opt['config'] = config
    run(opt)
