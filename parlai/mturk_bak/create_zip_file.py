# Copyright 2004-present Facebook. All Rights Reserved.
import os
import sys
import zipfile
import shutil
from subprocess import call


def setup_all_dependencies():
    devnull = open(os.devnull, 'w')

    # Set up all other dependencies
    command = "pip install --target=./lambda_server -r lambda_requirements.txt".split(" ")
    call(command, stdout=devnull, stderr=devnull)

    # Set up psycopg2
    command = "git clone https://github.com/yf225/awslambda-psycopg2.git".split(" ")
    call(command, stdout=devnull, stderr=devnull)
    shutil.copytree("./awslambda-psycopg2/with_ssl_support/psycopg2", "./lambda_server/psycopg2")
    shutil.rmtree("./awslambda-psycopg2")


def create_zip_file(files_to_copy=None, verbose=False):
    setup_all_dependencies()

    directory_path = 'lambda_server/'
    zip_file_name = 'lambda_server.zip'

    src = directory_path
    dst = zip_file_name

    if files_to_copy:
        for file_path in files_to_copy:
            shutil.copy2(file_path, src)

    zf = zipfile.ZipFile("%s" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            os.chmod(absname, 0o777)
            arcname = os.path.relpath(absname, abs_src)
            if verbose:
                print('zipping %s as %s' % (os.path.join(dirname, filename),
                                            arcname))
            zf.write(absname, arcname)
    zf.close()

    if verbose:
        print("Done!")

if __name__ == "__main__":
    create_zip_file()
