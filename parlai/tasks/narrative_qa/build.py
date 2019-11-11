#!/usr/bin/env python3


# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://github.com/deepmind/narrativeqa/archive/master.zip',
        'narrative_qa.zip',
        '9f6c484664394e0275944a4630a3de6294ba839162765d2839cc3d31a0b47a0e',
    )
]


def get_rows_for_set(reader, req_set):
    selected_rows = [row for row in reader if row['set'].strip() == req_set]
    return selected_rows


def read_csv_to_dict_list(filepath):
    f = open(filepath, 'r')
    return csv.DictReader(f, delimiter=','), f


def write_dict_list_to_csv(dict_list, filepath):
    keys = list(dict_list[0].keys())

    with open(filepath, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in dict_list:
            writer.writerow(row)


def divide_csv_into_sets(csv_filepath, sets=('train', 'valid', 'test')):
    reader, fh = read_csv_to_dict_list(csv_filepath)

    base_filename = os.path.basename(csv_filepath).split('.')[0]
    base_path = os.path.dirname(csv_filepath)

    for s in sets:
        path = os.path.join(base_path, base_filename + '_' + s + '.csv')
        fh.seek(0)
        rows = get_rows_for_set(reader, s)
        write_dict_list_to_csv(rows, path)

    fh.close()


def make_folders(base_path, sets=('train', 'valid', 'test')):
    for s in sets:
        path = os.path.join(base_path, s)
        if not os.path.exists(path):
            os.mkdir(path)


def move_files(base_path, sets=('train', 'valid', 'test')):
    source = os.listdir(base_path)

    for f in source:
        for s in sets:
            if f.endswith('_' + s + '.csv'):
                final_name = f[: -(len('_' + s + '.csv'))] + '.csv'
                f = os.path.join(base_path, f)
                shutil.move(f, os.path.join(base_path, s, final_name))


# Returns false unless the story was already downloaded and
# has appropriate size
def try_downloading(directory, row):
    document_id, kind, story_url = row['document_id'], row['kind'], row['story_url']
    story_path = os.path.join(directory, document_id + '.content')

    actual_story_size = 0
    if os.path.exists(story_path):
        with open(story_path, 'rb') as f:
            actual_story_size = len(f.read())

    if actual_story_size <= 19000:
        if kind == 'gutenberg':
            time.sleep(2)

        build_data.download(story_url, directory, document_id + '.content')
    else:
        return True

    file_type = subprocess.check_output(['file', '-b', story_path])
    file_type = file_type.decode('utf-8')

    if 'gzip compressed' in file_type:
        gz_path = os.path.join(directory, document_id + '.content.gz')
        shutil.move(story_path, gz_path)
        build_data.untar(gz_path)

    return False


def download_stories(path):
    documents_csv = os.path.join(path, 'documents.csv')
    tmp_dir = os.path.join(path, 'tmp')
    build_data.make_dir(tmp_dir)

    with open(documents_csv, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            print("Downloading %s (%s)" % (row['wiki_title'], row['document_id']))
            finished = try_downloading(tmp_dir, row)
            count = 0
            while not finished and count < 5:
                if count != 0:
                    print("Retrying (%d retries left)" % (5 - count - 1))
                finished = try_downloading(tmp_dir, row)
                count += 1


def build(opt):
    dpath = os.path.join(opt['datapath'], 'NarrativeQA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        print('downloading stories now')
        base_path = os.path.join(dpath, 'narrativeqa-master')

        download_stories(base_path)

        # move from tmp to stories
        tmp_stories_path = os.path.join(base_path, 'tmp')
        new_stories_path = os.path.join(base_path, 'stories')
        shutil.move(tmp_stories_path, new_stories_path)

        # divide into train, valid and test for summaries
        summaries_csv_path = os.path.join(
            base_path, 'third_party', 'wikipedia', 'summaries.csv'
        )
        new_path = os.path.join(base_path, 'summaries.csv')
        shutil.move(summaries_csv_path, new_path)

        divide_csv_into_sets(new_path)

        # divide into sets for questions
        questions_path = os.path.join(base_path, 'qaps.csv')
        divide_csv_into_sets(questions_path)

        # divide into sets for documents
        documents_path = os.path.join(base_path, 'documents.csv')
        divide_csv_into_sets(documents_path)

        # move specific set's files into their set's folder
        make_folders(base_path)
        move_files(base_path)

        # move narrativeqa-master to narrative_qa
        new_path = os.path.join(dpath, 'narrative_qa')
        shutil.move(base_path, new_path)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
