#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import parlai.tasks.opensubtitles.build_2018 as os_build

import glob
import os
import re


def clean_text(words):
    # character name
    if len(words) > 0 and words[-1] == ':':
        # this is a hack to filter out tagging that's not a name
        # the choice of 5 is arbitrary
        if len(words) > 5:
            return None
        sentence = ' '.join(words[:-1]).lower()
        sentence = '(' + sentence + '):'
        return sentence

    sentence = ' '.join(words).strip(' -').lower()

    sentence = os_build.CLEAN_BRACKETS_REGEX.sub('', sentence)
    if len([ch for ch in os_build.BRACKETS_CHARACTERS if ch in sentence]) > 0:
        return None

    sentence = sentence.replace('\\\'', '\'')
    if sentence.count('"') % 2 == 1:
        # There are unmatched double-quotes.
        # Usually, it means a quote got splitted into separate utterances,
        # so it's bad example of a dialog
        return None

    sentence = os_build.normalize_apostrophe(sentence)

    for (regex, replacement) in os_build.CLEANUP_REGEX_RULES:
        sentence = regex.sub(replacement, sentence)
    for (pattern, replacement) in os_build.CLEANUP_REPLACE_RULES:
        sentence = sentence.replace(pattern, replacement)

    words = os_build.normalize_whitespaces(sentence).split()

    if (
        len(words) > 0
        and any(map(lambda k: re.search(r'\w', k) is not None, words))
        and len(words) >= os_build.MIN_WORD_LENGTH
        and len(words) <= os_build.MAX_WORD_LENGTH
    ):
        return ' '.join(words)
    else:
        return None


def extract_data_from_xml_characters(xml_object):
    """
    Extract data from XML
    """
    previous_end_time = -1000
    conversation = []
    char_name = None
    for sentence_node in xml_object.getroot():
        if sentence_node.tag != 's':
            continue

        words = []
        start_time, end_time = None, None

        for node in sentence_node:
            if node.tag == 'time':
                time_value = os_build.parse_time_str(node.get('value'))
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

        if sentence is None:
            previous_end_time = max(start_time, end_time)
            continue

        # sentence is end of speech tag
        if sentence.endswith('):'):
            char_name = sentence
            print(sentence, start_time, end_time, previous_end_time)
            # need to start a new conversation
            if start_time - previous_end_time > os_build.MAX_TIME_DIFFERENCE_S:
                if len(conversation) > 1:
                    yield conversation
                conversation = []
            previous_end_time = max(start_time, end_time)
            continue

        if char_name is not None:
            print(char_name + ' ' + sentence, start_time, end_time, previous_end_time)

        if start_time - previous_end_time <= os_build.MAX_TIME_DIFFERENCE_S:
            # add to the conversation
            if char_name is not None and sentence is not None:
                sentence = char_name + ' ' + sentence
                conversation.append(sentence)
        else:
            if len(conversation) > 1:
                yield conversation
            conversation = []
            char_name = None

        previous_end_time = max(start_time, end_time)


class OSCharDataProcessor(os_build.DataProcessor):
    def xml_extract(self, xml_object):
        """
        Override to save character information
        """
        return extract_data_from_xml_characters(xml_object)


def check(conversation):
    prev_name = ''
    cnt = 0
    for i in range(len(conversation)):
        line = conversation[i]
        p = line.find('):')
        assert p > 0
        name = line[1:p]
        line = line[p + 3 :]
        if name == prev_name:
            cnt = cnt + 1
        else:
            prev_name = name
            cnt = 0
        if cnt > 2:
            return False
    return True


def output(fout, conversation):
    for i in range(0, len(conversation), 2):
        if i + 1 < len(conversation):
            fout.write(
                '%d %s\t%s\n' % (i / 2 + 1, conversation[i], conversation[i + 1])
            )
        else:
            fout.write('%d %s\n' % (i / 2 + 1, conversation[i]))


def filter_dialogue(data_path):
    out_path = data_path + '.filtered'
    fout = open(out_path, 'wt')
    # read conversations from file
    conversation = []
    count1 = 0
    count2 = 0
    with open(data_path, 'rt') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('1 '):
                if len(conversation) > 1 and check(conversation):
                    count1 += 1
                    count2 += len(conversation)
                    output(fout, conversation)
                conversation = []

            q = line.find(' ')
            assert q > 0
            line = line[q + 1 :]

            p = line.find('\t')
            if p > 0:
                text = line[:p]
                last_line = line[p + 1 :]
                conversation.append(text)
                conversation.append(last_line)
            else:
                conversation.append(line)

        if len(conversation) > 1 and check(conversation):
            count1 += 1
            count2 += len(conversation)
            output(fout, conversation)

    fout.close()

    out_stat_path = out_path + '.lengths'
    with open(out_stat_path, 'wt') as ff:
        ff.write(str(count1))
        ff.write('\n')
        ff.write(str(count2))


def build(datapath):
    dpath = os.path.join(datapath, 'md_gender', 'opensubtitles')
    if not os.path.exists(dpath):
        os.mkdir(dpath)

    version = '1'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        untar_path = os.path.join(dpath, 'OpenSubtitles', 'xml', 'en')

        if len(glob.glob(untar_path + '/*/*/*.xml')) != os_build.NUM_SUBTITLES_FILES:
            # Download the data.
            for downloadable_file in os_build.RESOURCES:
                downloadable_file.download_file(dpath)

        # Process and save the data in FB Format
        processor = OSCharDataProcessor(True)
        os_build.create_fb_format(untar_path, dpath, processor)

        # Now re-process and filter for gender purposes
        for fle in ['train.txt', 'test.txt', 'valid.txt']:
            path = os.path.join(dpath, fle)
            filter_dialogue(path)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

    return dpath
