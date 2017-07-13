# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""
Utilities for downloading and building data.
These can be replaced if your particular file system does not support them.
"""

import time
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
            return (len(text) > 1 and text[1] == version_string)
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


def download(url, path, fname, redownload=False):
    """Downloads file using `requests`. If ``redownload`` is set to false, then
    will not download tar file again if it is present (default ``True``)."""
    outfile = os.path.join(path, fname)
    download = not os.path.isfile(outfile) or redownload

    retry = 5
    exp_backoff = [2 ** r for r in reversed(range(retry))]
    while download and retry >= 0:
        resume_file = outfile + '.part'
        resume = os.path.isfile(resume_file)
        if resume:
            resume_pos = os.path.getsize(resume_file)
            mode = 'ab'
        else:
            resume_pos = 0
            mode = 'wb'
        response = None

        with requests.Session() as session:
            try:
                header = {'Range': 'bytes=%d-' % resume_pos,
                        'Accept-Encoding': 'identity'} if resume else {}
                response = session.get(url, stream=True, timeout=5, headers=header)

                # negative reply could be 'none' or just missing
                if resume and response.headers.get('Accept-Ranges', 'none') == 'none':
                    resume_pos = 0
                    mode = 'wb'

                CHUNK_SIZE = 32768
                total_size = int(response.headers.get('Content-Length', -1))
                # server returns remaining size if resuming, so adjust total
                total_size += resume_pos
                done = resume_pos

                with open(resume_file, mode) as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                            log_progress(done, total_size)
                    break
            except requests.exceptions.ConnectionError:
                retry -= 1
                print(''.join([' '] * 60), end='\r')  # TODO Better way to clean progress bar?
                if retry >= 0:
                    print('Connection error, retrying. (%d retries left)' % retry)
                    time.sleep(exp_backoff[retry])
                else:
                    print('Retried too many times, stopped retrying.')
            finally:
                if response:
                    response.close()
    if retry < 0:
        raise RuntimeWarning('Connection broken too many times. Stopped retrying.')

    if download and retry > 0:
        print()
        if done < total_size:
            raise RuntimeWarning('Received less data than specified in ' +
                                 'Content-Length header for ' + url + '.' +
                                 ' There may be a download problem.')
        move(resume_file, outfile)


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
