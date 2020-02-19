#!/usr/bin/env python3
# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Contains basic helper functions for running a parameter sweep on the FAIR
cluster and make HiPlot server render your own experiments.
"""

import os
import pandas as pd
import json
import collections
import hiplot as hip
import argparse
from threading import Timer
import glob
import time

# Axes to skipped for visualizing hyperparameter tuning
SKIP_KEYS = [
    'starttime',
    'lr',
    'model_file',
    'loss',
    'persona_path',
    'batchindex',
    'dict_file',
    'parlai_home',
    'model_name',
]

# Default hiplot server.
HIPLOT_SERVER_URL = 'http://127.0.0.1:5005/'


def flatten_nested_list(d, flatten_multi_item=False, parent_key='', sep='___') -> dict:
    """Flatten list-type dictionary values

    This function serve 2 ways of standarding dictionaries containing list-type values:
    (1) casting list to strings by default (flatten_multi_item = False)
    (2) iterating the dict by key/value, creats new keys for your new dictionary and creating
    the dictionary at final step. It also try to standardize dictionary containing
    one-element list and convert to scalar.

    :param d: the dictionary to be flattened/normalized.
    :param flatten_multi_item: if set True, the multi-element list will be flatten to
                               multiple (key, value) pairs to extended in d.
                               For example, {'d': [0.1,0.2]} to {'d___0': 0.1, 'd___1':0.2}.
    :param parent_key: key value to concat for the 0-layer
    :param sep: delimiter for creating new keys iteratively.
    :type d: dict
    :type flatten_multi_item: bool
    :type parent_key: str
    :type sep: str

    :returns: dataframe of training output
    :rtype: dict
    """

    for x in d:
        if type(d[x]) == list:
            if len(d[x]) == 1:
                d[x] = d[x][0]
            elif not flatten_multi_item:
                d[x] = '[' + ','.join(str(e) for e in d[x]) + ']'
    if not flatten_multi_item:
        return d
    res = []
    if not isinstance(d, collections.MutableMapping):
        res.append((parent_key, d))
        return dict(res)
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            res.extend(flatten_nested_list(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for idx, value in enumerate(v):
                res.extend(
                    flatten_nested_list(value, new_key + sep + str(idx), sep).items()
                )
        else:
            res.append((new_key, v))
    return dict(res)


def grep_results(
    root, trainstats_file_name='model.trainstats', opt_file_name='model.opt', keep='all'
) -> pd.DataFrame:
    """Grep results from training logs

    This util function parse various evaluation metrics for validation sets from .trainstats files and model hyperparameters settings
    from .opt files under the directory root and merge them two.

    :param root: root path containing all the raw output
    :param trainstats_file_name: names of the files containing validation reports
    :param opt_file_name: names of the files containing model hyperparameter settings
    :param keep: how to deal with tie-breaking for model performances in a single hyperparameter settings.
            first : take the first occurrence.
            last : take the last occurrence.
            all : do not drop any duplicates, even it means selecting more than n items.

    :returns: dataframe that merges evaluation metrics for validation set and model settings
    :rtype: dataframe
    """
    # parsed output from .trainstats files
    out = pd.DataFrame()

    # final results to be returned
    results = pd.DataFrame()

    # primary validation metrics for model performance, parsed from .opt file
    primary_metric = None

    for dn, _, _ in os.walk(root):
        # Load .opt file containing model hyperparameters
        try:
            with open(os.path.join(dn, opt_file_name), "r") as f:
                opt = json.load(f)
            if 'override' in opt:
                ov = opt['override']
                del opt['override']
                opt = {**opt, **ov}
        except IOError:
            print(os.path.join(dn, opt_file_name), IOError)
            continue

        # Load .trainstats file summarizing model performance
        out = out.iloc[0:0]
        try:
            with open(os.path.join(dn, trainstats_file_name), "r") as json_file:
                metrics_dict = json.load(json_file)
            col_flatten = [k for k in metrics_dict if isinstance(metrics_dict[k], list)]
            if len(col_flatten) > 0:
                out = pd.io.json.json_normalize(metrics_dict, record_path=col_flatten)
            else:
                out = pd.DataFrame.from_dict(metrics_dict)
            if (
                'validation_metric' in opt
                and opt['validation_metric'] in out.columns
                and 'validation_metric_mode' in opt
            ):
                if opt['validation_metric_mode'] == 'min':
                    out = out.nsmallest(
                        1, columns=[opt['validation_metric']], keep=keep
                    )
                elif opt['validation_metric_mode'] == 'max':
                    out = out.nlargest(1, columns=[opt['validation_metric']], keep=keep)
                ## metric sanity check:
                if primary_metric and primary_metric != opt['validation_metric']:
                    print("Validation metric mismatch between configs", dn)
                    return results
                elif not primary_metric:
                    primary_metric = opt['validation_metric']

            else:
                print(dn, 'Validation metric mismatch:')
                continue
        except IOError:
            print(os.path.join(dn, trainstats_file_name), IOError)
            continue

        # Merge the two output files
        try:
            opt = flatten_nested_list(opt)
            s = pd.Series(opt)
            result = out.merge(
                pd.DataFrame(data=[s.values] * len(s), columns=s.index),
                left_index=True,
                right_index=True,
            )
            result['model_name'] = os.path.basename(dn)
        except Exception as e:
            print(dn, " fail to merge trainstats and opt files ", e)
        else:
            results = results.append(result, ignore_index=True)

    # Reorganize the columns such that those related to performance metrics are on the rightmost side
    cols = [x for x in results.columns if x not in out.columns]
    cols.extend(out.columns)
    results = results.loc[:, cols]
    return results


def fetcher(uri) -> hip.Experiment:
    """Prepare param sweep output for hiplot

    This function collect the param sweeping results and simplify them for easy display using hiplot

    :param uri: root dir that containing all the param_sweeping results.

    :returns: hiplot Experiment Object for display
    """
    df = grep_results(uri)
    if len(df) == 0:
        print("Errors parsing trainstats and opt files", uri)
        return None

    # Drop constant columns & those specified in SKIP_KEYS
    df.drop(df.columns[df.nunique() <= 1], inplace=True, axis=1)
    df.drop([x for x in df.columns if x in SKIP_KEYS], axis=1, inplace=True)

    output_file_name = str((int)(time.time())) + '_param_sweep.csv'

    files_present = glob.glob(output_file_name)
    # if no matching files, write to csv, if there are matching files, print warning
    if not files_present:
        df.to_csv(os.path.join(uri, output_file_name))
        print(
            "Store parameter sweeping results to ", os.path.join(uri, output_file_name)
        )
    else:
        print('WARNING: This file already exists!')

    data = df.to_dict('records')
    return hip.Experiment.from_iterable(data)


def open_browser():
    import webbrowser

    webbrowser.open(HIPLOT_SERVER_URL, new=2, autoraise=True)


def main():
    Timer(1, open_browser).start()
    # By running the following command, a hiplot server will be rendered to display your experiment results
    # using the udf fetcher passed to hiplot.
    os.system('hiplot param_sweep_plot.fetcher')


if __name__ == '__main__':
    main()
