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
    os.system(
        ('cd %s' % path) +
        '; wget ' + url)


def untar(path, fname):
    os.system(
        ('cd %s' % path) + ';' +
        'tar xvfz %s' % (path + fname))


def mark_done(path):
    os.system('date > %s/.built' % path)
