# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""
Utilities for downloading and building data.
These can be replaced if your particular file system does not support them.
"""

import datetime
import os
import requests
import shutil
import wget

def built(path):
    """Checks if '.built' flag has been set for that task."""
    return os.path.isfile(os.path.join(path, '.built'))

def download(path, url, redownload=True):
    """Downloads file using `wget`. If redownload is set to false, then will not
    download tar file again if it is present (default true).
    """
    if redownload or not os.path.isfile(path):
        filename = wget.download(url, out=path)
        print() # wget prints download status, without newline

def download_request(url, path, fname):
    """Downloads file using `requests`."""
    with requests.Session() as session:
        response = session.get(url, stream=True)
        CHUNK_SIZE = 32768
        with open(os.path.join(path, fname), 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        response.close()

def make_dir(path):
    """Makes the directory and any nonexistent parent directories."""
    os.makedirs(path, exist_ok=True)

def mark_done(path):
    """Marks the path as done by adding a '.built' file with the current
    timestamp.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))

def move(path1, path2):
    """Renames the given file."""
    shutil.move(path1, path2)

def remove_dir(path):
    """Removes the given directory, if it exists."""
    shutil.rmtree(path, ignore_errors=True)

def untar(path, fname, deleteTar=True):
    """Unpacks the given archive file to the same directory, then (by default)
    deletes the archive file.
    """
    print('unpacking ' + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_from_google_drive(gd_id, destination):
    """Uses the requests package to download a file from Google Drive."""
    URL = 'https://docs.google.com/uc?export=download'

    with requests.Session() as session:
        response = session.get(URL, params={'id': gd_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            response.close()
            params = {'id': gd_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        CHUNK_SIZE = 32768
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        response.close()
