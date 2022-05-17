from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from typing import Optional

from .build import build
import random
import json
import os


class SPOLINDialogueTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.id = "SPOLIN"
        self.datatype = opt['datatype']
        build(opt)
        suffix = 'train' if opt['datatype'].startswith('train') else 'valid'
        if opt.get('use_acl_version') and suffix == 'train':
            suffix += '-acl'

        opt['datafile'] = os.path.join(
            opt['datapath'], 'spolin', f"spolin-{suffix}.json"
        )
        super().__init__(opt, shared)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('SPOLIN Dialogue Arguments')
        agent.add_argument(
            '-acl',
            '--use_acl_version',
            action="store_true",
            help='Use the version in the ACL paper that does not include further augmentation from SubTle Corpus',
        )

        agent.add_argument(
            '--include_nonyesands',
            action="store_true",
            help='Include non-yesands',
        )

        return parser

    def setup_data(self, path):

        print(f"Loading: {path}")
        with open(path, "r") as f:
            # don't set self.data
            self.data_ = json.load(f)

        processed_data = []
        yesands_dict = self.data_['yesands']
        for _source, yas in yesands_dict.items():
            processed_data += yas

        if self.opt.get("include_nonyesands"):
            non_yesands_dict = self.data['non-yesands']
            for _source, nyas in non_yesands_dict.items():
                processed_data += nyas

        self.processed_data = processed_data
        if 'train' in self.datatype:
            random.shuffle(self.processed_data)

        for ya_pair in self.processed_data:
            new_episode = True
            yield {
                "text": ya_pair['p'],
                "labels": ya_pair['r'],
            }, new_episode


class DefaultTeacher(SPOLINDialogueTeacher):
    pass
