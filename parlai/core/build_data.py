#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities for downloading and building data.
These can be replaced if your particular file system does not support them.
"""

import importlib
import time
import datetime
import os
import requests
import shutil
import tqdm


def built(path, version_string=None):
    """
    Checks if '.built' flag has been set for that task.

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
    """
    Marks the path as done by adding a '.built' file with the current timestamp
    plus a version description string if specified.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)


def download(url, path, fname, redownload=False):
    """
    Downloads file using `requests`. If ``redownload`` is set to false, then
    will not download tar file again if it is present (default ``True``).
    """
    outfile = os.path.join(path, fname)
    download = not os.path.isfile(outfile) or redownload
    print("[ downloading: " + url + " to " + outfile + " ]")
    retry = 5
    exp_backoff = [2 ** r for r in reversed(range(retry))]

    pbar = tqdm.tqdm(unit='B', unit_scale=True, desc='Downloading {}'.format(fname))

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
                pbar.total = total_size
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
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except requests.exceptions.ConnectionError:
                retry -= 1
                pbar.clear()
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
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeWarning('Received less data than specified in ' +
                                 'Content-Length header for ' + url + '.' +
                                 ' There may be a download problem.')
        move(resume_file, outfile)

    pbar.close()


def make_dir(path):
    """Makes the directory and any nonexistent parent directories."""
    # the current working directory is a fine path
    if path != '':
        os.makedirs(path, exist_ok=True)


def move(path1, path2):
    """Renames the given file."""
    shutil.move(path1, path2)


def remove_dir(path):
    """Removes the given directory, if it exists."""
    shutil.rmtree(path, ignore_errors=True)


def untar(path, fname, deleteTar=True):
    """
    Unpacks the given archive file to the same directory, then (by default)
    deletes the archive file.
    """
    print('unpacking ' + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)


def cat(file1, file2, outfile, deleteFiles=True):
    with open(outfile, 'wb') as wfd:
        for f in [file1, file2]:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd, 1024 * 1024 * 10)
                # 10MB per writing chunk to avoid reading big file into memory.
    if deleteFiles:
        os.remove(file1)
        os.remove(file2)


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


def download_models(opt, fnames, model_folder, version='v1.0', path='aws',
                    use_model_type=False):
    """
    Download models into the ParlAI model zoo from a url.

    :param fnames: list of filenames to download
    :param model_folder: models will be downloaded into models/model_folder/model_type
    :param path: url for downloading models; defaults to downloading from AWS
    :param use_model_type: whether models are categorized by type in AWS
    """

    model_type = opt.get('model_type', None)
    if model_type is not None:
        dpath = os.path.join(opt['datapath'], 'models', model_folder, model_type)
    else:
        dpath = os.path.join(opt['datapath'], 'models', model_folder)

    if not built(dpath, version):
        for fname in fnames:
            print('[building data: ' + dpath + '/' + fname + ']')
        if built(dpath):
            # An older version exists, so remove these outdated files.
            remove_dir(dpath)
        make_dir(dpath)

        # Download the data.
        for fname in fnames:
            if path == 'aws':
                url = 'http://parl.ai/downloads/_models/'
                url += model_folder + '/'
                if use_model_type:
                    url += model_type + '/'
                url += fname
            else:
                url = path + '/' + fname
            download(url, dpath, fname)
            if '.tgz' in fname or '.gz' in fname or '.zip' in fname:
                untar(dpath, fname)
        # Mark the data as built.
        mark_done(dpath, version)


def modelzoo_path(datapath, path):
    """
    If path starts with 'models', then we remap it to the model zoo path
    within the data directory (default is ParlAI/data/models).
    We download models from the model zoo if they are not here yet.
    """
    if path is None:
        return None
    if not path.startswith('models:'):
        return path
    else:
        # Check if we need to download the model
        animal = path[7:path.rfind('/')].replace('/', '.')
        if '.' not in animal:
            animal += '.build'
        module_name = 'parlai.zoo.{}'.format(animal)
        try:
            my_module = importlib.import_module(module_name)
            my_module.download(datapath)
        except (ImportError, AttributeError):
            pass

        return os.path.join(datapath, 'models', path[7:])
