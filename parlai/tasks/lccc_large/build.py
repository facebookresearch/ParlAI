from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager
import parlai.core.build_data as build_data
import os
import json

RESOURCES = [
    DownloadableFile(
        'https://cloud.tsinghua.edu.cn/f/8424e7b9454c4e628c24/?dl=1',
        'LCCC-large.zip',
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'LCCC_large')
    version = None

    if not build_data.built(dpath, version_string=version):
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        RESOURCES[0].download_file(dpath)
        # Format it for use with ParlAIDialogTeacher
        _create_parlai_format(dpath)
        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)


def _create_parlai_format(dpath: str):
    """
    Copy data into the format read by ConversationTeacher.
    """
    load_path = os.path.join(dpath, f'LCCD.json')
    save_path = os.path.join(dpath, f'LCCC_large.json')
    with PathManager.open(load_path, 'r', encoding='utf8') as f_read:
        data = json.load(f_read)
    with PathManager.open(save_path, 'w', encoding='utf8') as f_write:
        for episode in data:
            new_episode = []
            pid = 0
            for text in episode:
                new_episode.append(
                    {'id': 'partner{}'.format(pid + 1), 'text': text.replace(' ', '')}
                )
                pid = (pid + 1) % 2
            print(
                json.dumps({'dialog': [new_episode]}, ensure_ascii=False), file=f_write
            )
    os.remove(load_path)
