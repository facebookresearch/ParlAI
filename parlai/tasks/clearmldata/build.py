import parlai.core.build_data as build_data
import os
from clearml import Dataset


def build(opt):
    dpath = os.path.join(opt['datapath'], 'ClearMLData')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data from ClearML
        clearml_dataset = Dataset.get(
            dataset_name='clearmldata', dataset_project="ParlAI"
        )
        clearml_dataset.get_mutable_local_copy(target_folder=dpath, overwrite=True)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
