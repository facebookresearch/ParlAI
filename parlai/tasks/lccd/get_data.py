from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager
import parlai.core.build_data as build_data
import codecs
import os
import json

RESOURCES = [
    DownloadableFile(
        'https://cloud.tsinghua.edu.cn/f/8424e7b9454c4e628c24/?dl=1',
        'LCCC-large.zip',
        '2886ec16d892f9155be9470e83007f0630b7b7dd559e1604c4ee65b1bbb2ef56',
    ),
    DownloadableFile(
        'https://cloud.tsinghua.edu.cn/f/f131a4d259184566a29c/?dl=1',
        'LCCC-base-split.zip',
        'f5203511cd8d6a608008af0aa290aa516d983abc16aa510471e3c4ee6bca7886',
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'LCCD')
    version = None

    if not build_data.built(dpath, version_string=version):
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        RESOURCES[1].download_file(dpath)
        #for downloadable_file in RESOURCES:
        #    downloadable_file.download_file(dpath)
        # Format it for use with ParlAIDialogTeacher
        _create_parlai_format(dpath)
        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)


def _create_parlai_format(dpath: str):
    """
    Copy data into the format read by ParlAIDialogTeacher.

    'text' will be from the free Turker, who speaks first, and 'label' will be from the
    guided Turker.
    """

    datatypes = ['train', 'valid', 'test']
    for datatype in datatypes:
        datatype_full = 'LCCC-base_' + datatype
        load_path = os.path.join(dpath, f'{datatype_full}.json')
        save_path = os.path.join(dpath, f'{datatype}.json')

        print(f'Loading {load_path}.')
        with PathManager.open(load_path, 'r', encoding='utf8') as f_read:
            data = json.load(f_read)

        print(f'Saving to {save_path}')
        with PathManager.open(save_path, 'w', encoding='utf8') as f_write:
            for episode in data:
                new_episode = []
                pid = 0
                for text in episode:
                    new_episode.append({
                        'id': 'partner{}'.format(pid + 1),
                        'text': text.replace(' ', '')
                    })
                    pid = (pid + 1) % 2
                print(json.dumps({'dialog': [new_episode]}, ensure_ascii=False), file=f_write)
