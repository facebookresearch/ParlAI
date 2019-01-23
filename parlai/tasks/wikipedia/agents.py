#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
    Provides a dump of Wikipedia articles from 2/3/18.

    One can either load full articles, using 'wikipedia:full',
    or simply load the first paragraphs of the articles,
    using 'wikipedia:summary'

    To put the article in the labels and the title in the text, specify
    ':key-value' at the end (for a title/content key-value association)

'''
from parlai.core.teachers import DialogTeacher
from .build import build

import json
import os


class FullTeacher(DialogTeacher):
    """Reads Wikipedia pages one at a time
    """
    def __init__(self, opt, shared=None):
        self.key_value = ':key-value' in opt['task']
        opt['task'] = 'wikipedia:all'
        build(opt)
        self.opt = opt
        opt['datafile'] = os.path.join(
            opt['datapath'],
            'wikipedia/full/wiki_full_extracted')
        self.id = 'wikipedia'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        for subdir in os.listdir(path):
            if subdir == 'README.md':
                continue
            subdir_path = os.path.join(path, subdir)
            for wiki_file in os.listdir(subdir_path):
                wiki_file_path = os.path.join(subdir_path, wiki_file)
                with open(wiki_file_path) as wf:
                    for article_json in wf:
                        article = json.loads(article_json)
                        title = article['title']
                        text = article['text']
                        if self.key_value:
                            yield (title, [text]), True
                        else:
                            yield (text, ['']), True

    def get_extraction_instructions(self):
        '''If one wants to run extraction themselves on a raw wikipedia dump'''
        dpath = os.path.join(self.opt['datapath'], 'wikipedia', 'full')
        fname = 'enwiki-latest-pages-articles.xml.bz2'
        instructions = (
            "To complete the data extraction, please run the following:\n"
            "mkdir -p {download} && "
            "git clone https://github.com/attardi/wikiextractor "
            "{download}/wikiextract && cd {download}/wikiextract && "
            "python WikiExtractor.py {wikifile} --filter_disambig_pages "
            "-o {output} --json"
        ).format(
            download=self.opt['download_path'],
            wikifile=dpath + '/' + fname,
            output=dpath + '/' + 'wiki_extracted'
        )

        return instructions


class SummaryTeacher(DialogTeacher):
    """Reads Wikipedia pages one at a time, only uses summaries
    """
    def __init__(self, opt, shared=None):
        self.key_value = ':key-value' in opt['task']
        opt['task'] = 'wikipedia:summary'
        build(opt)
        opt['datafile'] = os.path.join(
            opt['datapath'],
            'wikipedia/summary/summaries.json')
        self.id = 'wikipedia'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as wf:
            for article_json in wf:
                article = json.loads(article_json)
                title = article['title']
                text = article['text']
                if self.key_value:
                    yield (title, [text]), True
                else:
                    yield (title + '\n' + text, ['']), True


class DefaultTeacher(SummaryTeacher):
    pass
