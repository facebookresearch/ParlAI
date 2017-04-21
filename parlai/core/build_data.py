# Copyright 2004-present Facebook. All Rights Reserved.
"""
Utilities for downloading and building data.
These can be replaced if your particular file system does not support them.
"""

import os

def built(path):
    return os.path.isfile(path + "/.built")

def remove_dir(path):
    s = ('rm -rf "%s"' % (path))
    if os.system(s) != 0:
        raise RuntimeError('failed: ' + s)

def make_dir(path):
    s = ('mkdir -p "%s"' % (path))
    if os.system(s) != 0:
        raise RuntimeError('failed: ' + s)

def move(path1, path2):
    s = ('mv "%s" "%s"' % (path1, path2))
    if os.system(s) != 0:
        raise RuntimeError('failed: ' + s)

def download(path, url):
    s = ('cd "%s"' % path) + '; wget ' + url
    if os.system(s) != 0:
        raise RuntimeError('failed: ' + s)

def untar(path, fname, deleteTar=True):
    print('unpacking ' + fname)
    s = ('cd "%s"' % path) + ';' + 'tar xfz "%s"' % (path + fname)
    if os.system(s) != 0:
        raise RuntimeError('failed: ' + s)
    # remove tar file
    if deleteTar:
        s = ('cd "%s"' % path) + ';' + 'rm "%s"' % (path + fname)
        if os.system(s) != 0:
            raise RuntimeError('failed: ' + s)

def mark_done(path):
    s = ('date > "%s"/.built' % path)
    if os.system(s) != 0:
        raise RuntimeError('failed: ' + s)

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_file_from_google_drive(gd_id, destination):
    import requests

    URL = 'https://docs.google.com/uc?export=download'

    session = requests.Session()
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
