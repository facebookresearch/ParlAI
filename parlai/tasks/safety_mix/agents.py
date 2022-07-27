import json
import random
from typing import Optional
from parlai.core import message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.dialogue_safety.agents import StandardTeacher, NOT_OK_CLASS, OK_CLASS
from parlai.utils.io import PathManager
from .build import build, FILE_TYPE_EXTENSIONS
import os


def _path(opt):
    # build the data if it does not exist
    build(opt['datapath'])
    # set up path to data (specific to each dataset)
    data_path = os.path.join(opt['datapath'], 'safety_mix')
    return data_path


class SafetyMixTeacher(FixedDialogTeacher):
    DATA_PATH_PER_TYPE = {
        'helper': 'helper/helper',
        'troll': 'troll/troll',
        'master_troll': 'master_troll/master_troll',
        'safe_troll': 'safe_troll/safe_troll',
        'unsafe_troll': 'unsafe_troll/unsafe_troll',
        'lazy_troll': 'lazy_troll/lazy_troll',
        'gaslight_troll': 'gaslight_troll/gaslight_troll',
    }

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = StandardTeacher.add_cmdline_args(parser, partial_opt=partial_opt)
        parser = parser.add_argument_group('SafetyMix arguments')
        parser.add_argument(
            '--mix-user-type',
            type=str,
            default='troll',
            help='The troll user type you want in the safety mix.',
        )
        return parser

    def __init__(self, opt, shared=None):
        self.opt = opt
        dpath = _path(opt)
        self.data_path = os.path.join(
            dpath, self.DATA_PATH_PER_TYPE[opt['mix_user_type']]
        )

        self.fixed_random = random.Random(42)
        self.label_candidates = [NOT_OK_CLASS, OK_CLASS]

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self._setup_data(opt['datatype'])
        super().__init__(opt, shared)
        self.reset()

    def _load_data_dump(self, datatype):
        d_type = datatype.split(':')[0]
        loaded_data = []
        __import__('ipdb').set_trace()  # FIXME
        with PathManager.open(self.self.FILE_TYPE_EXTENSION[d_type], 'rb') as f:
            dump = list(f)
        for json_str in dump:
            loaded_data.append(json.loads(json_str))
        return loaded_data

    def _setup_data(self, datatype):
        # load data
        self.data_dump = self._load_data_dump(datatype)
        self.data = self.data_dump
        self.fixed_random.shuffle(self.data)

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def get(self, episode_idx, entry_idx):
        return message(self.data[episode_idx])

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class PosSafetyMixTeacher(SafetyMixTeacher):
    # DATA_PATH = '/checkpoint/daju/catch_the_trolls_data/user_based/5050_mix/balanced_generated_data/'
    DATA_PATH = '/checkpoint/daju/catch_the_trolls_data/user_based/5050_mix_200/balanced_generated_data/'
    DATA_PATH_PER_TYPE = {
        'helper': 'helper/pos_helper',
        'troll': 'troll/pos_troll',
        'master_troll': 'master_troll/pos_master_troll',
        'safe_troll': 'safe_troll/pos_safe_troll',
        'unsafe_troll': 'unsafe_troll/pos_unsafe_troll',
        'lazy_troll': 'lazy_troll/pos_lazy_troll',
        'gaslight_troll': 'gaslight_troll/pos_gaslight_troll',
    }


class DefaultTeacher(SafetyMixTeacher):
    pass
