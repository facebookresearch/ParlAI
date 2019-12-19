#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

from typing import Optional

import gzip
import os

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data


class ParseInsuranceQA(object):
    version: Optional[str] = None
    label2answer_fname: Optional[str] = None

    @classmethod
    def read_gz(cls, filename):
        f = gzip.open(filename, 'rb')
        return [x.decode('utf-8') for x in f.readlines()]

    @classmethod
    def readlines(cls, path):
        if path.endswith(".gz"):
            lines = cls.read_gz(path)
        else:
            lines = open(path).readlines()
        return lines

    @classmethod
    def wids2sent(cls, wids, d_vocab):
        return " ".join([d_vocab[w] for w in wids])

    @classmethod
    def read_vocab(cls, vocab_path):
        d_vocab = {}
        with open(vocab_path, "r") as f:
            for line in f:
                fields = line.rstrip('\n').split("\t")
                if len(fields) != 2:
                    raise ValueError(
                        "vocab file (%s) corrupted. Line (%s)"
                        % (repr(line), vocab_path)
                    )
                else:
                    wid, word = fields
                    d_vocab[wid] = word
        return d_vocab

    @classmethod
    def read_label2answer(cls, label2answer_path_gz, d_vocab):
        lines = cls.readlines(label2answer_path_gz)

        d_label_answer = {}
        for line in lines:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(
                    "label2answer file (%s) corrupted. Line (%s)"
                    % (repr(line), label2answer_path_gz)
                )
            else:
                aid, s_wids = fields
                sent = cls.wids2sent(s_wids.split(), d_vocab)
                d_label_answer[aid] = sent
        return d_label_answer

    @classmethod
    def create_fb_format(cls, out_path, dtype, inpath, d_vocab, d_label_answer):
        pass

    @classmethod
    def write_data_files(cls, dpext, out_path, d_vocab, d_label_answer):
        pass

    @classmethod
    def build(cls, dpath):
        print("building version: %s" % cls.version)

        # the root of dataset
        dpext = os.path.join(dpath, 'insuranceQA-master/%s' % cls.version)

        # read vocab file
        vocab_path = os.path.join(dpext, "vocabulary")
        d_vocab = cls.read_vocab(vocab_path)

        # read label2answer file
        label2answer_path_gz = os.path.join(dpext, cls.label2answer_fname)
        d_label_answer = cls.read_label2answer(label2answer_path_gz, d_vocab)

        # Create out path
        out_path = os.path.join(dpath, cls.version)
        build_data.make_dir(out_path)

        # Parse and write data files
        cls.write_data_files(dpext, out_path, d_vocab, d_label_answer)


class ParseInsuranceQAV1(ParseInsuranceQA):
    version = "V1"
    label2answer_fname = "answers.label.token_idx"

    @classmethod
    def write_data_files(cls, dpext, out_path, d_vocab, d_label_answer):
        data_fnames = [
            ("train", "question.train.token_idx.label"),
            ("valid", "question.dev.label.token_idx.pool"),
            ("test", "question.test1.label.token_idx.pool"),
            # ("test2", "question.test2.label.token_idx.pool")
        ]
        for dtype, data_fname in data_fnames:
            data_path = os.path.join(dpext, data_fname)
            cls.create_fb_format(out_path, dtype, data_path, d_vocab, d_label_answer)

    @classmethod
    def create_fb_format(cls, out_path, dtype, inpath, d_vocab, d_label_answer):
        print('building fbformat:' + dtype)
        fout = open(os.path.join(out_path, dtype + '.txt'), 'w')
        lines = open(inpath).readlines()

        for line in lines:
            fields = line.rstrip("\n").split("\t")
            if dtype == "train":
                assert len(fields) == 2, "data file (%s) corrupted." % inpath
                s_q_wids, s_good_aids = fields

                q = cls.wids2sent(s_q_wids.split(), d_vocab)
                good_ans = [d_label_answer[aid_] for aid_ in s_good_aids.split()]
                # save good answers (train only)
                s = '1 ' + q + '\t' + "|".join(good_ans)
                fout.write(s + '\n')
            else:
                assert len(fields) == 3, "data file (%s) corrupted." % inpath
                s_good_aids, s_q_wids, s_bad_aids = fields

                q = cls.wids2sent(s_q_wids.split(), d_vocab)
                good_ans = [d_label_answer[aid_] for aid_ in s_good_aids.split()]
                bad_ans = [d_label_answer[aid_] for aid_ in s_bad_aids.split()]
                # save good answers and candidates
                s = (
                    '1 '
                    + q
                    + '\t'
                    + "|".join(good_ans)
                    + '\t\t'
                    + "|".join(good_ans + bad_ans)
                )
                fout.write(s + '\n')
        fout.close()


class ParseInsuranceQAV2(ParseInsuranceQA):
    version = "V2"
    label2answer_fname = "InsuranceQA.label2answer.token.encoded.gz"

    @classmethod
    def write_data_files(cls, dpext, out_path, d_vocab, d_label_answer):
        data_fnames_tmpl = [
            (
                "train.%s",
                "InsuranceQA.question.anslabel.token.%s.pool.solr.train.encoded.gz",
            ),  # noqa: E501
            (
                "valid.%s",
                "InsuranceQA.question.anslabel.token.%s.pool.solr.valid.encoded.gz",
            ),  # noqa: E501
            (
                "test.%s",
                "InsuranceQA.question.anslabel.token.%s.pool.solr.test.encoded.gz",
            ),  # noqa: E501
        ]
        for n_cands in [100, 500, 1000, 1500]:
            for dtype_tmp, data_fname_tmp in data_fnames_tmpl:
                dtype = dtype_tmp % n_cands
                data_fname = data_fname_tmp % n_cands
                data_path = os.path.join(dpext, data_fname)
                cls.create_fb_format(
                    out_path, dtype, data_path, d_vocab, d_label_answer
                )

    @classmethod
    def create_fb_format(cls, out_path, dtype, inpath, d_vocab, d_label_answer):
        print('building fbformat:' + dtype)
        fout = open(os.path.join(out_path, dtype + '.txt'), 'w')
        lines = cls.readlines(inpath)

        for line in lines:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(
                    "data file (%s) corrupted. Line (%s)" % (repr(line), inpath)
                )
            else:
                _, s_q_wids, s_good_aids, s_bad_aids = fields
                q = cls.wids2sent(s_q_wids.split(), d_vocab)
                good_ans = [d_label_answer[aid_] for aid_ in s_good_aids.split()]
                bad_ans = [d_label_answer[aid_] for aid_ in s_bad_aids.split()]
                # save
                s = (
                    '1 '
                    + q
                    + '\t'
                    + "|".join(good_ans)
                    + '\t\t'
                    + "|".join(good_ans + bad_ans)
                )
                fout.write(s + '\n')
        fout.close()


RESOURCES = [
    DownloadableFile(
        'https://github.com/shuzi/insuranceQA/archive/master.zip',
        'insuranceqa.zip',
        '53e1c4a68734c6a0955dcba50d5a2a9926004d4cd4cda2e988cc7b990a250fbf',
    )
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'InsuranceQA')
    version = '1'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        ParseInsuranceQAV1.build(dpath)
        ParseInsuranceQAV2.build(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
