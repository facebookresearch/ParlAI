#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for downloading and building data.

These can be replaced if your particular file system does not support them.
"""

import importlib
import json
import time
import datetime
import os
import requests
import shutil
import hashlib
import tqdm
import gzip
import math
import parlai.utils.logging as logging
from parlai.utils.io import PathManager

try:
    from torch.multiprocessing import Pool
except ImportError:
    from multiprocessing import Pool


class DownloadableFile:
    """
    A class used to abstract any file that has to be downloaded online.

    Any task that needs to download a file needs to have a list RESOURCES
    that have objects of this class as elements.

    This class provides the following functionality:

    - Download a file from a URL / Google Drive
    - Untar the file if zipped
    - Checksum for the downloaded file
    - Send HEAD request to validate URL or Google Drive link

    An object of this class needs to be created with:

    - url <string> : URL or Google Drive id to download from
    - file_name <string> : File name that the file should be named
    - hashcode <string> : SHA256 hashcode of the downloaded file
    - zipped <boolean> : False if the file is not compressed
    - from_google <boolean> : True if the file is from Google Drive
    """

    def __init__(self, url, file_name, hashcode, zipped=True, from_google=False):
        self.url = url
        self.file_name = file_name
        self.hashcode = hashcode
        self.zipped = zipped
        self.from_google = from_google

    def checksum(self, dpath):
        """
        Checksum on a given file.

        :param dpath: path to the downloaded file.
        """
        sha256_hash = hashlib.sha256()
        with PathManager.open(os.path.join(dpath, self.file_name), "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != self.hashcode:
                # remove_dir(dpath)
                raise AssertionError(
                    f"Checksum for {self.file_name} from \n{self.url}\n"
                    f"does not match the expected checksum:\n{sha256_hash.hexdigest()} != {self.hashcode}"
                    "\n\nPlease try again."
                )
            else:
                logging.debug("Checksum Successful")

    def download_file(self, dpath):
        if self.from_google:
            download_from_google_drive(self.url, os.path.join(dpath, self.file_name))
        else:
            download(self.url, dpath, self.file_name)

        self.checksum(dpath)

        if self.zipped:
            untar(dpath, self.file_name)

    def check_header(self):
        """
        Performs a HEAD request to check if the URL / Google Drive ID is live.
        """
        session = requests.Session()
        if self.from_google:
            URL = 'https://docs.google.com/uc?export=download'
            response = session.head(URL, params={'id': self.url}, stream=True)
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'
            }
            response = session.head(self.url, allow_redirects=True, headers=headers)
        status = response.status_code
        session.close()

        assert status == 200


def built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.

    If a version_string is provided, this has to match, or the version is regarded as
    not built.
    """
    if version_string:
        fname = os.path.join(path, '.built')
        if not PathManager.exists(fname):
            return False
        else:
            with PathManager.open(fname, 'r') as read:
                text = read.read().split('\n')
            return len(text) > 1 and text[1] == version_string
    else:
        return PathManager.exists(os.path.join(path, '.built'))


def mark_done(path, version_string=None):
    """
    Mark this path as prebuilt.

    Marks the path as done by adding a '.built' file with the current timestamp
    plus a version description string if specified.

    :param str path:
        The file path to mark as built.

    :param str version_string:
        The version of this dataset.
    """
    with PathManager.open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)


def download(url, path, fname, redownload=False, num_retries=5):
    """
    Download file using `requests`.

    If ``redownload`` is set to false, then will not download tar file again if it is
    present (default ``False``).
    """
    outfile = os.path.join(path, fname)
    download = not PathManager.exists(outfile) or redownload
    logging.info(f"Downloading {url} to {outfile}")
    retry = num_retries
    exp_backoff = [2 ** r for r in reversed(range(retry))]

    pbar = tqdm.tqdm(unit='B', unit_scale=True, desc='Downloading {}'.format(fname))

    while download and retry > 0:
        response = None

        with requests.Session() as session:
            try:
                response = session.get(url, stream=True, timeout=5)

                # negative reply could be 'none' or just missing
                CHUNK_SIZE = 32768
                total_size = int(response.headers.get('Content-Length', -1))
                # server returns remaining size if resuming, so adjust total
                pbar.total = total_size
                done = 0

                with PathManager.open(outfile, 'wb') as f:
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
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                retry -= 1
                pbar.clear()
                if retry > 0:
                    pl = 'y' if retry == 1 else 'ies'
                    logging.debug(
                        f'Connection error, retrying. ({retry} retr{pl} left)'
                    )
                    time.sleep(exp_backoff[retry])
                else:
                    logging.error('Retried too many times, stopped retrying.')
            finally:
                if response:
                    response.close()
    if retry <= 0:
        raise RuntimeError('Connection broken too many times. Stopped retrying.')

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeError(
                f'Received less data than specified in Content-Length header for '
                f'{url}. There may be a download problem.'
            )

    pbar.close()


def make_dir(path):
    """
    Make the directory and any nonexistent parent directories (`mkdir -p`).
    """
    # the current working directory is a fine path
    if path != '':
        PathManager.mkdirs(path)


def remove_dir(path):
    """
    Remove the given directory, if it exists.
    """
    shutil.rmtree(path, ignore_errors=True)


def untar(path, fname, delete=True, flatten_tar=False):
    """
    Unpack the given archive file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    if ".zip" in fname:
        return _unzip(path, fname, delete=delete)
    else:
        return _untar(path, fname, delete=delete, flatten=flatten_tar)


def _untar(path, fname, delete=True, flatten=False):
    """
    Unpack the given archive file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    import tarfile

    logging.debug(f'unpacking {fname}')
    fullpath = os.path.join(path, fname)
    # very painfully manually extract files so that we can use PathManger.open
    # instead, lest we are using fb internal file services

    with tarfile.open(fileobj=PathManager.open(fullpath, 'rb')) as tf:
        for item in tf:
            item_name = item.name
            while item_name.startswith("./"):
                # internal file systems will actually create a literal "."
                # directory, so we gotta watch out for that
                item_name = item_name[2:]
            if flatten:
                # flatten the tar file if there are subdirectories
                fn = os.path.join(path, os.path.split(item_name)[-1])
            else:
                fn = os.path.join(path, item_name)
            logging.debug(f"Extracting to {fn}")
            if item.isdir():
                PathManager.mkdirs(fn)
            elif item.isfile():
                with PathManager.open(fn, 'wb') as wf, tf.extractfile(item.name) as rf:
                    tarfile.copyfileobj(rf, wf)
            else:
                raise NotImplementedError("No support for symlinks etc. right now.")

    if delete:
        PathManager.rm(fullpath)


def ungzip(path, fname, deleteGZip=True):
    """
    Unzips the given gzip compressed file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool deleteGZip:
        If true, the compressed file will be deleted after extraction.
    """

    def _get_output_filename(input_fname):
        GZIP_EXTENSIONS = ('.gz', '.gzip', '.tgz', '.tar')
        for ext in GZIP_EXTENSIONS:
            if input_fname.endswith(ext):
                return input_fname[: -len(ext)]
        return f'{input_fname}_unzip'

    logging.debug(f'unzipping {fname}')
    fullpath = os.path.join(path, fname)

    with gzip.open(fullpath, 'rb') as fin, open(
        _get_output_filename(fullpath), 'wb'
    ) as fout:
        shutil.copyfileobj(fin, fout)

    if deleteGZip:
        os.remove(fullpath)


def _unzip(path, fname, delete=True):
    """
    Unpack the given zip file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    import zipfile

    logging.debug(f'unpacking {fname}')
    fullpath = os.path.join(path, fname)
    with zipfile.ZipFile(PathManager.open(fullpath, 'rb'), 'r') as zf:
        for member in zf.namelist():
            outpath = os.path.join(path, member)
            if zf.getinfo(member).is_dir():
                logging.debug(f"Making directory {outpath}")
                PathManager.mkdirs(outpath)
                continue
            logging.debug(f"Extracting to {outpath}")
            with zf.open(member, 'r') as inf, PathManager.open(outpath, 'wb') as outf:
                shutil.copyfileobj(inf, outf)
    if delete:
        PathManager.rm(fullpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_from_google_drive(gd_id, destination):
    """
    Use the requests package to download a file from Google Drive.
    """
    URL = 'https://docs.google.com/uc?export=download'

    with requests.Session() as session:
        response = session.get(URL, params={'id': gd_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            response.close()
            params = {'id': gd_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        CHUNK_SIZE = 32768
        with PathManager.open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        response.close()


def get_model_dir(datapath):
    return os.path.join(datapath, 'models')


def download_models(
    opt,
    fnames,
    model_folder,
    version='v1.0',
    path='aws',
    use_model_type=False,
    flatten_tar=False,
):
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
            logging.info(f'building data: {dpath}/{fname}')
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
                untar(dpath, fname, flatten_tar=flatten_tar)
        # Mark the data as built.
        mark_done(dpath, version)


def modelzoo_path(datapath, path):
    """
    Map pretrain models filenames to their path on disk.

    If path starts with 'models:', then we remap it to the model zoo path within the
    data directory (default is ParlAI/data/models). We download models from the model
    zoo if they are not here yet.
    """
    if path is None:
        return None
    if (
        not path.startswith('models:')
        and not path.startswith('zoo:')
        and not path.startswith('izoo:')
    ):
        return path
    elif path.startswith('models:') or path.startswith('zoo:'):
        zoo = path.split(':')[0]
        zoo_len = len(zoo) + 1
        model_path = path[zoo_len:]
        # Check if we need to download the model
        if "/" in path:
            animal = path[zoo_len : path.rfind('/')].replace('/', '.')
        else:
            animal = path[zoo_len:]
        if '.' not in animal:
            animal += '.build'
        module_name = 'parlai.zoo.{}'.format(animal)
        try:
            my_module = importlib.import_module(module_name)
            my_module.download(datapath)
        except (ImportError, AttributeError):
            try:
                # maybe we didn't find a specific model, let's try generic .build
                animal_ = '.'.join(animal.split(".")[:-1]) + '.build'
                module_name_ = 'parlai.zoo.{}'.format(animal_)
                my_module = importlib.import_module(module_name_)
                my_module.download(datapath)
            except (ImportError, AttributeError):
                # truly give up
                raise ImportError(
                    f'Could not find pretrained model in {module_name} or {module_name_}.'
                )

        return os.path.join(datapath, 'models', model_path)
    else:
        # Internal path (starts with "izoo:") -- useful for non-public
        # projects.  Save the path to your internal model zoo in
        # parlai_internal/.internal_zoo_path
        # TODO: test the internal zoo.
        zoo_path = 'parlai_internal/zoo/.internal_zoo_path'
        if not PathManager.exists('parlai_internal/zoo/.internal_zoo_path'):
            raise RuntimeError(
                'Please specify the path to your internal zoo in the '
                'file parlai_internal/zoo/.internal_zoo_path in your '
                'internal repository.'
            )
        else:
            with PathManager.open(zoo_path, 'r') as f:
                zoo = f.read().split('\n')[0]
            return os.path.join(zoo, path[5:])


def download_multiprocess(
    urls, path, num_processes=32, chunk_size=100, dest_filenames=None, error_path=None
):
    """
    Download items in parallel (e.g. for an image + dialogue task).

    WARNING: may have issues with OS X.

    :param urls:
        Array of urls to download
    :param path:
        directory to save items in
    :param num_processes:
        number of processes to use
    :param chunk_size:
        chunk size to use
    :param dest_filenames:
        optional array of same length as url with filenames.  Images will be
        saved as path + dest_filename
    :param error_path:
        where to save error logs
    :return:
        array of tuples of (destination filename, http status code, error
        message if any). Note that upon failure, file may not actually be
        created.
    """

    pbar = tqdm.tqdm(total=len(urls), position=0)

    # Resume TODO: isfile() may take too long ?? Should I try in a .tmp file
    if dest_filenames:
        if len(dest_filenames) != len(urls):
            raise Exception(
                'If specified, destination filenames must equal url array in length.'
            )
    else:

        def _naming_fn(url, url_metadata=None):
            return hashlib.md5(url.encode('utf-8')).hexdigest()

        dest_filenames = [_naming_fn(url) for url in urls]

    items = zip(urls, dest_filenames)
    remaining_items = [
        it for it in items if not PathManager.exists(os.path.join(path, it[1]))
    ]
    logging.info(
        f'Of {len(urls)} items, {len(urls) - len(remaining_items)} already existed; only going to download {len(remaining_items)} items.'
    )
    pbar.update(len(urls) - len(remaining_items))

    pool_chunks = (
        (remaining_items[i : i + chunk_size], path, _download_multiprocess_single)
        for i in range(0, len(remaining_items), chunk_size)
    )
    remaining_chunks_count = math.ceil(float(len(remaining_items) / chunk_size))
    logging.info(
        f'Going to download {remaining_chunks_count} chunks with {chunk_size} images per chunk using {num_processes} processes.'
    )

    pbar.desc = 'Downloading'
    all_results = []
    collected_errors = []

    with Pool(num_processes) as pool:
        for idx, chunk_result in enumerate(
            pool.imap_unordered(_download_multiprocess_map_chunk, pool_chunks, 2)
        ):
            all_results.extend(chunk_result)
            for dest_file, http_status_code, error_msg in chunk_result:
                if http_status_code != 200:
                    # msg field available as third item in the tuple
                    # not using b/c error log file would blow up
                    collected_errors.append(
                        {
                            'dest_file': dest_file,
                            'status_code': http_status_code,
                            'error': error_msg,
                        }
                    )
                    logging.error(
                        f'Bad download - chunk: {idx}, dest_file: {dest_file}, http status code: {http_status_code}, error_msg: {error_msg}'
                    )
            pbar.update(len(chunk_result))
    pbar.close()

    if error_path:
        now = time.strftime("%Y%m%d-%H%M%S")
        error_filename = os.path.join(
            error_path, 'parlai_download_multiprocess_errors_%s.log' % now
        )

        with PathManager.open(os.path.join(error_filename), 'w') as error_file:
            error_file.write(json.dumps(collected_errors))
            logging.error(f'Summary of errors written to {error_filename}')

    logging.info(
        f'Of {len(remaining_items)} items attempted downloading, '
        f'{len(collected_errors)} had errors.'
    )

    logging.debug('Finished downloading chunks.')
    return all_results


def _download_multiprocess_map_chunk(pool_tup):
    """
    Helper function for Pool imap_unordered.

    Apparently function must be pickable (which apparently means must be
    defined at the top level of a module and can't be a lamdba) to be used in
    imap_unordered. Has to do with how it's passed to the subprocess.

    :param pool_tup: is a tuple where first arg is an array of tuples of url
    and dest file name for the current chunk and second arg is function to be
    called.
    :return: an array of tuples
    """
    items = pool_tup[0]
    path = pool_tup[1]
    fn = pool_tup[2]
    return [fn(it[0], path, it[1]) for it in items]


def _download_multiprocess_single(url, path, dest_fname):
    """
    Helper function to download an individual item.

    Unlike download() above, does not deal with downloading chunks of a big
    file, does not support retries (and does not fail if retries are exhausted).

    :param url: URL to download from
    :param path: directory to save in
    :param dest_fname: destination file name of image
    :return tuple (dest_fname, http status)
    """

    status = None
    error_msg = None
    try:
        # 'User-Agent' header may need to be specified
        headers = {}

        # Use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(
            url, stream=False, timeout=10, allow_redirects=True, headers=headers
        )
    except Exception as e:
        # Likely a timeout during fetching but had an error in requests.get()
        status = 500
        error_msg = '[Exception during download during fetching] ' + str(e)
        return dest_fname, status, error_msg

    if response.ok:
        try:
            with PathManager.open(os.path.join(path, dest_fname), 'wb+') as out_file:
                # Some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)
            status = 200
        except Exception as e:
            # Likely a timeout during download or decoding
            status = 500
            error_msg = '[Exception during decoding or writing] ' + str(e)
    else:
        # We get here if there is an HTML error page (i.e. a page saying "404
        # not found" or anything else)
        status = response.status_code
        error_msg = '[Response not OK] Response: %s' % response

    return dest_fname, status, error_msg
