#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
"""
Utilities for downloading and building data.
These can be replaced if your particular file system does not support them.
"""

import os

def built(path):
    return os.path.isfile(path + "/.built")


def remove_dir(path):
    os.system('rm -rf %s' % (path))


def make_dir(path):
    os.system('mkdir -p %s' % (path))


def move(path1, path2):
    os.system('mv %s %s' % (path1, path2))


def download(path, url):
    # TODO(jase): need to use normal wget
    os.system(
        ('cd %s' % path) +
        '; ~/bin/sh/proxy-exec wget ' + url)


def untar(path, fname):
    os.system(
        ('cd %s' % path) + ';' +
        'tar xvfz %s' % (path + fname))


def mark_done(path):
    os.system('date > %s/.built' % path)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def download_file_from_google_drive(gd_id, destination):
    import requests

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': gd_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': gd_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
