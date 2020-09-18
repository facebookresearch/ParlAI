import os

from parlai.core import build_data
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        '$HOME/datasets/ParlAI/WritingPrompts/writing_prompts_train.txt',
        'writing_prompts_train.txt',
        '2d46c87db2e5861f8c671c21f247bf944796ab1ecdf7566743ab42dadcc1faeb',
        zipped=True,
    ),
    DownloadableFile(
        '$HOME/datasets/ParlAI/WritingPrompts/writing_prompts_valid.txt',
        '09b0ca93d00c4bcc010eeea07c7b6d3ad5082c1432b07acb655ac1eda3f96876',
        zipped=True,
    ),
    DownloadableFile(
        '$HOME/datasets/ParlAI/WritingPrompts/writing_prompts_test.txt',
        'writing_prompts_test.txt',
        '9cb8a467eaf12d6977994340281d4aaef0467a7dc444462775f144c2a2b04426',
        zipped=True,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'writing_prompts')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES[:2]:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)