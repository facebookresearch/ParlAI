import parlai.core.build_data as build_data
import os

def build(opt):
    dpath = os.path.join(opt['datapath'], 'inference_guided_dialogue')
    version = '1'

    if not build_data.built(dpath, version_string=version):
        # assume alreay put in place 

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

    return dpath