# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import gzip
import os

import parlai.core.build_data as build_data


def read_gz(filename):
    f = gzip.open(filename, 'rb')
    return [x.decode('ascii') for x in f.readlines()]


def wids2sent(wids, d_vocab):
    return " ".join([d_vocab[w] for w in wids])


def read_vocab(vocab_path):
    d_vocab = {}
    with open(vocab_path, "r") as f:
        for line in f:
            line = line.rstrip('\n')
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError("vocab file (%s) corrupted. Line (%s)" % (repr(line), vocab_path))
            else:
                wid, word = fields
                d_vocab[wid] = word
    return d_vocab


def read_label2answer(label2answer_path_gz, d_vocab):
    lines = read_gz(label2answer_path_gz)
    d_label_answer = {}
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError("label2answer file (%s) corrupted. Line (%s)" % (repr(line), label2answer_path_gz))
        else:
            aid, s_wids = fields
            sent = wids2sent(s_wids.split(), d_vocab)
            d_label_answer[aid] = sent
    return d_label_answer


def create_fb_format(outpath, dtype, inpath, d_vocab, d_label_answer):
    print('building fbformat:' + dtype)
    fout = open(os.path.join(outpath, dtype + '.txt'), 'w')
    lines = read_gz(inpath)

    for line in lines:
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError("data file (%s) corrupted. Line (%s)" % (repr(line), inpath))
        else:
            _, s_q_wids, s_good_aids, s_bad_aids = fields
            q = wids2sent(s_q_wids.split(), d_vocab)
            good_ans = [d_label_answer[aid_] for aid_ in s_good_aids.split()]
            bad_ans = [d_label_answer[aid_] for aid_ in s_bad_aids.split()]
            # save
            s = '1 ' + q + '\t' + "|".join(good_ans) + '\t\t' + "|".join(good_ans + bad_ans)
            fout.write(s + '\n')
    fout.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'InsuranceQA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data from github.
        fname = 'insuranceqa.zip'
        url = 'https://github.com/shuzi/insuranceQA/archive/master.zip'
        build_data.download(url, dpath, fname, redownload=False)
        build_data.untar(dpath, fname)

        # According to the author, V2 holds the latest data
        dpext = os.path.join(dpath, 'insuranceQA-master/V2')

        # read vocab file
        vocab_path = os.path.join(dpext, "vocabulary")
        d_vocab = read_vocab(vocab_path)

        # read label2answer file
        label2answer_path_gz = os.path.join(dpext, "InsuranceQA.label2answer.token.encoded.gz")
        d_label_answer = read_label2answer(label2answer_path_gz, d_vocab)

        # TODO: right now it uses 100 by default, but 500, 1000, 1500 (# of label candidates) should also be available
        train_path_gz = os.path.join(dpext, "InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded.gz")
        valid_path_gz = os.path.join(dpext, "InsuranceQA.question.anslabel.token.100.pool.solr.valid.encoded.gz")
        test_path_gz = os.path.join(dpext, "InsuranceQA.question.anslabel.token.100.pool.solr.test.encoded.gz")

        create_fb_format(dpath, 'train', train_path_gz, d_vocab, d_label_answer)
        create_fb_format(dpath, 'valid', valid_path_gz, d_vocab, d_label_answer)
        create_fb_format(dpath, 'test', test_path_gz, d_vocab, d_label_answer)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
