# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import codecs
import gzip
import parlai.core.build_data as build_data


def create_fb_format(inpath, outpath):
    print("[building fbformat]")
    ftrain = open(outpath + 'train.txt', 'w')
    fvalid = open(outpath + 'valid.txt', 'w')
    ftest = open(outpath + 'test.txt', 'w')

    # find all the files.
    import subprocess
    result = subprocess.run(['find', inpath],
                            stdout=subprocess.PIPE)
    list = str(result.stdout).split('\\n')
    
    conv_id = 0
    for f in list:
        if f[-3:] == '.gz':
            dialog = ''
            conv_id = conv_id + 1
            with gzip.open(f, 'r') as f1:
                print(str(conv_id) + ": " + f)
                words = ''
                line_id = 1
                turn_id = 1
                for line in f1:
                    line = str(line)
                    if line.find('<s id="') != -1: 
                        # new sentence
                        if len(words) > 0:
                            if (turn_id % 2) == 0:
                                dialog = dialog + (str(line_id) + ' ' + words)
                            else:
                                dialog = dialog + ('\t' + words + '\n')
                                line_id = line_id + 1
                        turn_id = turn_id + 1
                        words = ''
                    else:
                        i1 = line.find('<w id="')
                        if i1 >= 0:
                            line = line[i1:]
                            word = line[line.find('>')+1:line.find('</w')]
                            words = words + ' ' + word.replace('\t', ' ')
            handle = ftrain
            if (conv_id % 10) == 0:
                handle = ftest
            if (conv_id % 10) == 1:
                handle = fvalid
            handle.write(dialog + '\n')

    ftrain.close()
    fvalid.close()
    ftest.close()


def build(opt):
    dpath = opt['datapath'] + "/OpenSubtitles/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = "download.php?f=OpenSubtitles/en.tar.gz"
        url = ("http://opus.lingfil.uu.se/" + fname)
        build_data.download(dpath, url)
        build_data.untar(dpath, 'download.php?f=OpenSubtitles%2Fen.tar.gz')

        create_fb_format(dpath + '/OpenSubtitles/en/', dpath)

        # Mark the data as built.
        build_data.mark_done(dpath)
