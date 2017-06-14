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


def built(path, version_string=None):
    """Checks if '.built' flag has been set for that task.
    If a version_string is provided, this has to match, or the version
    is regarded as not built.
    """
    if version_string:
        fname = os.path.join(path, '.built')
        if not os.path.isfile(fname):
            return False
        else:
            with open(fname, 'r') as read:
                text = read.read().split('\n')
            return (len(text) == 2 and text[1] == version_string)
    else:
        return os.path.isfile(os.path.join(path, '.built'))

def mark_done(path, version_string=None):
    """Marks the path as done by adding a '.built' file with the current
    timestamp plus a version description string if specified.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)

def log_progress(curr, total, width=40):
    """Displays a bar showing the current progress."""
    done = min(curr * width // total, width)
    remain = width - done
    progress = '[{}{}] {} / {}'.format(
        ''.join(['|'] * done),
        ''.join(['.'] * remain),
        curr,
        total
    )
    print(progress, end='\r')


def download(url, path, fname, redownload=True):
    """Downloads file using `requests`. If ``redownload`` is set to false, then
    will not download tar file again if it is present (default ``True``)."""
    outfile = os.path.join(path, fname)
    if redownload or not os.path.isfile(outfile):
        with requests.Session() as session:
            response = session.get(url, stream=True)
            CHUNK_SIZE = 32768
            total_size = int(response.headers.get('Content-Length', -1))
            done = 0
            with open(outfile, 'wb') as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                    if total_size > 0:
                        done += len(chunk)
                        if total_size < done:
                            # don't freak out if content-length was too small
                            total_size = done
                        log_progress(done, total_size)
            if done < total_size:
                raise RuntimeWarning('Received less data than specified in ' +
                                     'Content-Length header for ' + url + '.' +
                                     ' There may be a download problem.')
            print()
            response.close()


def make_dir(path):
    """Makes the directory and any nonexistent parent directories."""
    os.makedirs(path, exist_ok=True)


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
