#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Provides a dump of Wikipedia articles from 2/3/18.

One can either load full articles, using 'wikipedia:full',
or simply load the first paragraphs of the articles,
using 'wikipedia:summary'

To put the article in the labels and the title in the text, specify
':key-value' at the end (for a title/content key-value association)
"""
from parlai.core.teachers import DialogTeacher, ChunkTeacher
from parlai.core.message import Message
from .build import build

import json
import os
from typing import List, Tuple


class FullTeacher(DialogTeacher):
    """
    Reads Wikipedia pages one at a time.
    """

    def __init__(self, opt, shared=None):
        self.key_value = ':key-value' in opt['task']
        opt['task'] = 'wikipedia:all'
        build(opt)
        self.opt = opt
        opt['datafile'] = os.path.join(
            opt['datapath'], 'wikipedia/full/wiki_full_extracted'
        )
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
        """
        If one wants to run extraction themselves on a raw wikipedia dump.
        """
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
            output=dpath + '/' + 'wiki_extracted',
        )

        return instructions


class FullSplitTeacher(ChunkTeacher):
    """
    Full Wikipedia teacher that splits the chunks into train/valid/test.
    """

    def __init__(self, opt, shared=None):
        self.TRAINSIZE = 5437097
        self.VALIDSIZE = 71052
        self.TESTSIZE = 39975

        if shared is None:
            # set map
            self.opt = opt
            self._set_chunk_idx_to_file()
        else:
            self.chunk_idx_to_file = shared['chunk_idx_to_file']
        super().__init__(opt, shared)

    def _get_data_folder(self):
        return os.path.join(self.opt['datapath'], 'wikipedia/full/wiki_full_extracted')

    def get_num_samples(self, datatype) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        if 'train' in datatype:
            return self.TRAINSIZE, self.TRAINSIZE
        elif 'valid' in datatype:
            return self.VALIDSIZE, self.VALIDSIZE
        else:
            # test
            return self.TESTSIZE, self.TESTSIZE

    def _set_chunk_idx_to_file(self):
        folder = self._get_data_folder()
        all_subdirs = sorted([x for x in os.listdir(folder) if 'README' not in x])
        self.chunk_idx_to_file = {i: x for i, x in enumerate(all_subdirs)}

    def get_fold_chunks(self, datatype) -> List[int]:  # type: ignore
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        all_chunk_idxs = list(self.chunk_idx_to_file.keys())
        if 'train' in datatype:
            return all_chunk_idxs[:-2]
        elif 'valid' in datatype:
            return [all_chunk_idxs[-2]]
        else:
            return [all_chunk_idxs[-1]]

    def load_from_chunk(self, chunk_idx: int) -> List[Tuple[str, str]]:
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        output = []
        chunk_path = os.path.join(self.folder, self.chunk_idx_to_file[chunk_idx])
        for wiki_file in os.listdir(chunk_path):
            wiki_file_path = os.path.join(chunk_path, wiki_file)
            with open(wiki_file_path) as wf:
                for article_json in wf:
                    article = json.loads(article_json)
                    title = article['title']
                    text = article['text']
                    output.append((title, text))

        return output

    def create_message(self, queue_output: Tuple[str, ...]) -> 'Message':
        """
        Given the tuple output of the queue, return an act.
        """
        title, text = queue_output
        return Message(
            {'title': title, 'text': text, 'labels': [''], 'episode_done': True}
        )

    def share(self):
        shared = super().share()
        shared['chunk_idx_to_file'] = self.chunk_idx_to_file
        return shared


class SummaryTeacher(DialogTeacher):
    """
    Reads Wikipedia pages one at a time, only uses summaries.
    """

    def __init__(self, opt, shared=None):
        self.key_value = ':key-value' in opt['task']
        opt['task'] = 'wikipedia:summary'
        build(opt)
        opt['datafile'] = os.path.join(
            opt['datapath'], 'wikipedia/summary/summaries.json'
        )
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
