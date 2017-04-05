# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data


def _process(fname, fout):
    with open(fname) as f:
        lines = [line.strip('\n') for line in f]
    # main article
    s = '1 ' + lines[2]
    # add question
    s = s + lines[4]
    # add answer
    s = s + '\t' + lines[6]
    # add candidates (and strip them of the real names)
    for i in range(8, len(lines)):
        lines[i] = lines[i].split(':')[0]
    s = s + '\t\t' + '|'.join(lines[8:-1])
    fout.write(s + '\n\n')


def create_fb_format(outpath, dtype, inpath):
    print("building fbformat:" + dtype)
    import os
    fout = open(outpath + dtype + '.txt', 'w')
    for file in os.listdir(inpath):
        if file.endswith(".question"):
            fname = os.path.join(inpath, file)
            _process(fname, fout)
    fout.close()


def build(opt):
    dpath = opt['datapath'] + "/QADailyMail/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = "qadailymail.tar.gz"
        gd_id = "0BwmD_VLjROrfN0xhTDVteGQ3eG8"
        build_data.download_file_from_google_drive(gd_id, dpath + fname)
        build_data.untar(dpath, fname)

        ext = 'dailymail/questions/'
        create_fb_format(dpath, 'train', dpath + ext + 'training/')
        create_fb_format(dpath, 'valid', dpath + ext + 'validation/')
        create_fb_format(dpath, 'test', dpath + ext + 'test/')

        # Mark the data as built.
        build_data.mark_done(dpath)
