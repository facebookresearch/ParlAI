# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import DialogTeacher
from .build import build

import ujson as json
import os


class AllTeacher(DialogTeacher):
    """Reads Wikipedia pages one at a time
    """
    def __init__(self, opt, shared=None):
        opt['task'] = 'wikipedia:all'
        success = build(opt)
        self.opt = opt
        if not success:
            '''Need to extract the rest of the data'''
            raise RuntimeError(self.get_instructions())
        opt['datafile'] = os.path.join(opt['datapath'], 'wiki_full_extracted')
        self.id = 'wikipedia'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        if not os.path.isfile(path):
            '''Need to extract the rest of the data'''
            raise RuntimeError(self.get_instructions())
        for subdir in os.listdir(path):
            for wiki_file in os.listdir(subdir):
                with open(wiki_file) as wf:
                    for article_json in wf:
                        article = json.load(article_json)
                        title = article['title']
                        text = article['text']
                        yield (title + '\n' + text), True

    def get_instructions(self):
        dpath = os.path.join(self.opt['datapath'], 'wikipedia', 'full')
        fname = 'enwiki-latest-pages-articles.xml.bz2'
        instructions = """
        To complete the data extraction, please do the following:
        \n
            1. (wherever you would like) git clone https://github.com/attardi/wikiextractor
            2. cd wikiextractor
            3. python WikiExtractor.py {} --filter_disambig_pages -o {} --json
        """.format(dpath + '/' + fname, dpath + '/' + 'wiki_extracted')

        return instructions

class SummaryTeacher(DialogTeacher):
    """Reads Wikipedia pages one at a time, only uses summaries
    """
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'wiki_summary')
        self.id = 'wikipedia'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as wf:
            for article_json in wf:
                article = json.load(article_json)
                title = article['title']
                text = article['text']
                yield (title + '\n' + text), True


class DefaultTeacher(AllTeacher):
    pass
