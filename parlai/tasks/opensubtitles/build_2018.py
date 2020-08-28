#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import glob
import gzip
import multiprocessing
import os
import re
import sys
import time
import tqdm
import xml.etree.ElementTree as ET
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager

RESOURCES = [
    DownloadableFile(
        'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/en.zip',
        'OpenSubtitles2018.zip',
        '917af90fcaa8b0ebb3d59d9f8d205f304f31bf92cbf15aa6e9ee030f6691755e',
    )
]

NUM_MOVIE_FOLDERS = 140044
NUM_SUBTITLES_FILES = 446612

MAX_TIME_DIFFERENCE_S = 2
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 20

# remove brackets
CLEAN_BRACKETS_REGEX = re.compile(
    r'<!--.*?-->|<[^>]*>|\([^\)]*\)|\[[^\]]*\]|\{[^\}]*\}|##|~'
)
# Usually, unbalanced brackets correspond to very noisy sentences
# '#' is usually pretty bad and means lyrics of the song
BRACKETS_CHARACTERS = ['[', ']', '(', ')', '{', '}', '<', '>', '#']

MULTI_WHITESPACES_REGEX = re.compile(r'\s+')

# Existing apostrophe tokenization in Open Subtitles is not compatible with nltk
APOSTROPHE_REPLACEMENT_REGEX = [
    (re.compile(r"(\s?)n(\s?)'(\s?)t(\s|$)"), "\\1n't\\4"),
    (re.compile(r"'(\s?)(s|re|em|im|bout|cause|ve|d|ll|ne)(\s+|$)"), " '\\2\\3"),
    # it's a common (in OpenSubtitles) spelling error to use 'il instead of 'll
    (re.compile(r"'(\s?)il(\s|$)"), " 'll\\2"),
    (re.compile(r"(\s|^)i(\s?)'(\s?)(m|mm)(\s|$)"), "\\1i 'm\\5"),
    (re.compile(r"in(\s?)'(\s|$)"), "ing\\2"),
    (re.compile(r"(\s|^)ma(\s?)'(\s?)am(\s|$)"), "\\1ma'am\\4"),
    (re.compile(r"(\s|^)c(\s?)'(\s?)mon(\s|$)"), "\\1c'mon\\4"),
    (re.compile(r"(\s|^)o(\s?)'(\s?)clock(\s|$)"), "\\1o'clock\\4"),
    (re.compile(r"(\s|^)y(\s?)'(\s?)all(\s|$)"), "\\1y'all\\4"),
]

# Some cleaning steps are taken from
CLEANUP_REGEX_RULES = [
    # remove speaker tag "xxx: "
    (re.compile(r'^\s*[A-z]*\s*:'), ''),
    # remove unnecessary symbols
    (re.compile(r"-{2,}"), ' '),
    # delete a space right before a period for titles
    (re.compile(r'(?<=( mr| jr| ms| dr| st|mrs)) \.'), '. '),
]

CLEANUP_REPLACE_RULES = [
    ('"', ' '),
    ("``", " "),
    ("''", " "),
    ("% %", " "),
    ("i̇", "i"),
]


def get_movie_id(filename_path):
    dirpath, filename = os.path.split(filename_path)
    _, movie_id_str = os.path.split(dirpath)
    return int(movie_id_str)


# OpenSubtitles2016 contains have several subtitles per movie,
# stored in a separate folders.
# We gather all subtitles files based on the movie they correspond to
# and apply deduplication for the extracted replicas
def get_list_of_files(top_path):
    result = {}
    for path, _dirs, files in os.walk(top_path):
        for filename in files:
            if filename.endswith('.xml'):
                full_filename = os.path.realpath(os.path.join(path, filename))
                assert PathManager.exists(full_filename), 'Bad file ' + full_filename
                movie_id = get_movie_id(full_filename)
                if movie_id not in result:
                    result[movie_id] = []
                result[movie_id].append(full_filename)
    return result


def parse_xml(filepath):
    extension = os.path.splitext(filepath)[1]
    if extension == '.gz':
        with gzip.open(filepath, 'r') as f:
            return ET.parse(f)
    else:
        return ET.parse(filepath)


def normalize_whitespaces(sentence):
    return MULTI_WHITESPACES_REGEX.sub(' ', sentence).strip()


def normalize_apostrophe(sentence):
    sentence = normalize_whitespaces(sentence)
    for rule in APOSTROPHE_REPLACEMENT_REGEX:
        sentence = rule[0].sub(rule[1], sentence)
    return sentence


def clean_text(words):
    if len(words) > 0 and words[-1] == ':':
        return None
    sentence = ' '.join(words).strip(' -').lower()

    sentence = CLEAN_BRACKETS_REGEX.sub('', sentence)
    if len([ch for ch in BRACKETS_CHARACTERS if ch in sentence]) > 0:
        return None

    sentence = sentence.replace('\\\'', '\'')
    if sentence.count('"') % 2 == 1:
        # There are unmatched double-quotes.
        # Usually, it means a quote got splitted into separate utterances,
        # so it's bad example of a dialog
        return None

    sentence = normalize_apostrophe(sentence)

    for (regex, replacement) in CLEANUP_REGEX_RULES:
        sentence = regex.sub(replacement, sentence)
    for (pattern, replacement) in CLEANUP_REPLACE_RULES:
        sentence = sentence.replace(pattern, replacement)

    words = normalize_whitespaces(sentence).split()

    if (
        len(words) > 0
        and any(map(lambda k: re.search(r'\w', k) is not None, words))
        and len(words) >= MIN_WORD_LENGTH
        and len(words) <= MAX_WORD_LENGTH
    ):
        return ' '.join(words)
    else:
        return None


def parse_time_str(time_value_str):
    if not (
        time_value_str is not None
        and len(time_value_str) == 12
        and time_value_str[2] == ':'
        and time_value_str[5] == ':'
        and time_value_str[8] == ','
    ):
        return None
    try:
        return (
            int(time_value_str[0:2]) * 3600
            + int(time_value_str[3:5]) * 60
            + int(time_value_str[6:8])
        )
    except ValueError:
        return None


def extract_data_from_xml(xml_object):
    previous_end_time = -1000
    conversation = []
    for sentence_node in xml_object.getroot():
        if sentence_node.tag != 's':
            continue

        words = []
        start_time, end_time = None, None

        for node in sentence_node:
            if node.tag == 'time':
                time_value = parse_time_str(node.get('value'))
                if time_value is None:
                    continue
                if node.get('id')[-1] == 'S':
                    start_time = (
                        time_value
                        if start_time is None
                        else min(time_value, start_time)
                    )
                elif node.get('id')[-1] == 'E':
                    end_time = (
                        time_value if end_time is None else max(time_value, end_time)
                    )
                else:
                    raise Exception('Unknown time-id for node: %s' % node)
            elif node.tag == 'w':
                if node.text is not None and len(node.text) > 0:
                    words.append(node.text)
            else:
                pass

        sentence = clean_text(words)

        start_time = start_time or previous_end_time
        end_time = end_time or previous_end_time
        # add to the conversation
        # flush and start new conversation
        if (
            sentence is not None
            and start_time - previous_end_time <= MAX_TIME_DIFFERENCE_S
        ):
            conversation.append(sentence)
        else:
            if len(conversation) > 1:
                yield conversation
            conversation = []
            if sentence is not None:
                conversation.append(sentence)

        previous_end_time = max(start_time, end_time)


def conversation_to_fb_format(conversation):
    assert len(conversation) > 1
    lines = []
    for i in range(0, len(conversation), 2):
        if i + 1 < len(conversation):
            lines.append(
                '%d %s\t%s' % (i / 2 + 1, conversation[i], conversation[i + 1])
            )
        else:
            lines.append('%d %s' % (i / 2 + 1, conversation[i]))
    return '\n'.join(lines)


def conversation_to_basic_format(conversation):
    assert len(conversation) > 1
    lines = []
    for i in range(len(conversation)):
        if i + 1 < len(conversation):
            lines.append('1 %s\t%s' % (conversation[i], conversation[i + 1]))
    return '\n'.join(lines)


class DataProcessor(object):
    def __init__(self, use_history):
        self.use_history = use_history

    def __call__(self, movie_id_with_files):
        movie_id, files = movie_id_with_files
        data = set()
        for filepath in files:
            try:
                xml_object = parse_xml(filepath)
                for conversation in extract_data_from_xml(xml_object):
                    if self.use_history:
                        data.add(conversation_to_fb_format(conversation))
                    else:
                        data.add(conversation_to_basic_format(conversation))
            except ET.ParseError:
                # TODO: We possibly can log these errors,
                # but I'm not sure how it would intervene with the PrograssLogger
                pass
            except Exception:
                print(
                    'Unexpected error for file %s:\n%s' % (filepath, sys.exc_info()[0]),
                    file=sys.stderr,
                )
                raise
        data_str = '\n'.join(data) + ('\n' if len(data) > 0 else '')
        return data_str


def create_fb_format(inpath, outpath, use_history):
    print('[building fbformat]')
    start_time = time.time()

    ftrain = open(os.path.join(outpath, 'train.txt'), 'w')
    fvalid = open(os.path.join(outpath, 'valid.txt'), 'w')
    ftest = open(os.path.join(outpath, 'test.txt'), 'w')

    movie_dirs = get_list_of_files(inpath)
    total_movie_dirs = len(movie_dirs)
    total_files = sum([len(l) for l in movie_dirs.values()])
    print(
        '[Found %d movie folders and %d subtitles within %s in %d seconds]'
        % (total_movie_dirs, total_files, inpath, time.time() - start_time)
    )

    assert total_movie_dirs == NUM_MOVIE_FOLDERS, 'Incorrect number of movies'
    assert total_files == NUM_SUBTITLES_FILES, 'Incorrect number of files'

    processor = DataProcessor(use_history)

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for i, s in enumerate(pool.imap(processor, tqdm.tqdm(movie_dirs.items()))):
            handle = ftrain
            # TODO: Shall we use smaller valid/test sets? Even 10% is A LOT here
            if i % 10 == 0:
                handle = ftest
            if i % 10 == 1:
                handle = fvalid
            handle.write(s)

    ftrain.close()
    fvalid.close()
    ftest.close()

    print(
        '[Data has been successfully extracted in %d seconds]'
        % (time.time() - start_time,)
    )


def build(datapath, use_history):
    dpath = os.path.join(datapath, 'OpenSubtitles2018')
    if not use_history:
        dpath += '_no_history'
    version = '1'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        untar_path = os.path.join(dpath, 'OpenSubtitles', 'xml', 'en')

        if len(glob.glob(untar_path + '/*/*/*.xml')) != NUM_SUBTITLES_FILES:
            # Download the data.
            for downloadable_file in RESOURCES:
                downloadable_file.download_file(dpath)

        create_fb_format(untar_path, dpath, use_history)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
    return dpath
