#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script for automatically generating the model card.
"""
from datetime import date, datetime
from collections.abc import Iterable

# import subprocess

# from parlai.utils.torch import total_parameters, trainable_parameters
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.core.opt import Opt
from parlai.zoo.model_list import model_list
from parlai.core.metrics import METRICS_DISPLAY_DATA
from parlai.core.worlds import create_task
from parlai.tasks.task_list import task_list
from parlai.utils.strings import colorize
from projects.safety_bench.run_unit_tests import (
    SafetyUnitTests,
    _interpret_results,
    _disclaimer,
)

import parlai.scripts.data_stats as data_stats
import parlai.scripts.eval_model as eval_model

# FIXME: change if moved to public
from parlai_internal.scripts.label_subgroup_saver import LabelSubgroupSaver as lsav
import os
import json
import copy
import re
import contextlib
import io
import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_stats_folder = 'data_stats'
# metrics that are not precentages
not_percent_m = ['ppl']

# for metrics that aren't in METRICS_DISPLAY_DATA
extra_metric_info = {
    'ppl': (
        'perplexity',
        'perplexity. See [here](https://en.wikipedia.org/wiki/Perplexity) for more info.',
    )
}

default_dropdowns = [
    # (Dropdown name, common key function or common key prefix, other keys)
    (
        'model / neural net info',
        None,
        [
            'n_layers',
            'ffn_size',
            'dropout',
            'attention_dropout',
            'n_heads',
            'n_positions',
            'n_segments',
            'variant',
            'activation',
            'output_scaling',
            'memory_attention',
            'reduction_type',
            'load_from_pretrained_ranker',
            'round',
            'threshold',
        ],
    ),
    ('embedding info', 'embedding', []),
    ('validation and logging info', 'valid', []),
    ('dictionary info/pre-processing', 'dict', []),
    (
        'other dataset-related info',
        None,
        [
            'fix_contractions',
            'truncate',
            'text_truncate',
            'label_truncate',
            'use_test_set',
            'split_lines',
            'balance_data',
            'task',
            'evaltask',
        ],
    ),
    ('more batch and learning rate info', lambda x: 'batch' in x or 'lr' in x, []),
    (
        'training info',
        None,
        [
            'numthreads',
            'shuffle',
            'numworkers',
            'metrics',
            'gpu',
            'data_parallel',
            'optimizer',
            'gradient_clip',
            'adam_eps',
            'nesterov',
            'nus',
            'betas',
            'warmup_updates',
            'warmup_rate',
            'update_freq',
            'fp16',
            'max_train_time',
            'no_cuda',
        ],
    ),
    ('pytorch info', 'pytorch', []),
]

# types of functions; I get the strings mixed ups easily
CLASSIFIER = 'classifier'
GENERATOR = 'generator'
RETRIEVER = 'retriever'
RANKER = 'ranker'
all_model_dict = {model['path']: model for model in model_list}


def get_safety_mgs(func, sep='\n\n'):
    # capture output from the function that prints
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func()
    out = f.getvalue()
    # remove any colors for command-line
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    tmp = ansi_escape.sub('', out)
    return list(filter(None, tmp.split(sep)))


def get_heatmap(stats_dfs, title=None, tfsize=16, heatmapkws_user=None, fout=None):
    # get vmax
    tmp_max = max([df.max().max() for df in stats_dfs])
    tmp_min = min([df.min().min() for df in stats_dfs])
    step = 5 if (tmp_max - tmp_min) < 0.5 else 10
    vmax = step * math.ceil(tmp_max * 100 / step) / 100
    vmin = step * math.floor(tmp_min * 100 / step) / 100
    # create dictionary for heatmap args and update any args passed by user
    cmap = sns.color_palette("Oranges", as_cmap=True)
    heatmapkws = {
        'vmin': vmin,
        'vmax': vmax,
        'annot': True,
        'linecolor': 'black',
        'cmap': cmap,
        'linewidths': 0.75,
        'fmt': ".2%",
    }
    if heatmapkws_user:
        heatmapkws.update(heatmapkws_user)

    # create subplots
    ratios = [df.shape[0] for df in stats_dfs]
    N = len(stats_dfs)
    fig, axs = plt.subplots(nrows=N, gridspec_kw={'height_ratios': ratios}, sharex=True)

    # add color bar (with the last subplot)
    # left, bot, width, height
    cbar_ax = fig.add_axes([0.99, 0.15, 0.03, 0.7])
    for i, df in enumerate(stats_dfs):
        ax = sns.heatmap(
            df,
            ax=axs[i],
            xticklabels=i == (N - 1),
            **heatmapkws,
            cbar=i == (N - 1),
            cbar_ax=None if i != (N - 1) else cbar_ax,
        )
        # remove bottom ticks, set the borders
        ax.tick_params(bottom=False)
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        # to be safe, make sure y-axis is in the right direction
        _ = ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        if i == (N - 1):
            # rotate x-axis label{}s if last subplot
            labels = ax.get_xticklabels()
            _ = ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')
            # adjust cbar labels if last subplot
            ticks = {i / 100: f"{i}%" for i in range(0, int(vmax * 100 + 1), step)}
            cbar = ax.collections[0].colorbar
            cbar.set_ticks(list(ticks.keys()))
            cbar.set_ticklabels(list(ticks.values()))
    if title:
        fig.suptitle(title, fontsize=tfsize)
    if fout:
        fig.savefig(fout, bbox_inches="tight")
    return fig, axs


def create_warning_message(missing, line_width=150):
    message = f"\n**|MISSING| {missing} |MISSING|**\n".center(line_width, ' ')
    return message


def possible_and_statement(lis):
    L = len(lis)
    if L == 0:
        return ''
    if L == 1:
        return lis[0]
    if L == 2:
        return lis[0] + ' and ' + lis[1]
    return ', '.join(lis[:-1]) + ', and ' + lis[-1]


def stats_opt_setup(opt, task, use_test_set):
    opt['num_examples'] = -1
    opt['agent'] = 0
    opt['new_line_new_utt'] = False
    opt['ignore_tokens'] = ''
    opt['task'] = task
    opt['batchsize'] = 1
    opt['use_test_set'] = use_test_set
    return opt


def extra_special_print(string, color='green', sep='-', sep2='*'):
    line_width = 80
    print(sep * line_width)
    print(colorize(f' {string} '.center(line_width, sep2), color))
    print(sep * line_width)


def get_metric_info(metric_orginal):
    metric = metric_orginal.replace('___', ' ').split()
    key = metric[-1]
    # get the title, description
    if extra_metric_info.get(key):
        title, description = extra_metric_info[key]
    elif METRICS_DISPLAY_DATA.get(key):
        title = METRICS_DISPLAY_DATA[key].title
        description = METRICS_DISPLAY_DATA[key].description
        description = description[0].lower() + description[1:]
    else:
        description, title = (None, None)
    # adding english for metrics
    # case 1 is for sth like class___notok___f1
    if len(metric) == 3 and metric[0] == 'class' and title:
        metric_eng = f"{metric_format(metric_orginal)}, the {title.lower()} scores for the class {metric[1]}"
    else:
        metric_eng = metric_format(metric_orginal)
    return metric, title, description, metric_eng


def make_table_header(table_header, align=None):
    align_bars = '---'
    if align == 'center':
        align_bars = ':---:'
    elif align == 'right':
        align_bars = '---:'
    elif align == 'left':
        align_bars = ':---'
    return [
        ('|' + ' | '.join(table_header)),
        ' | '.join([align_bars] * len(table_header)),
    ]


def make_md_table(rows, cols, align='center'):
    """
    expect the columns to be a list of headers rows to be a list of lists.
    """
    table = make_table_header(cols, align)
    for row in rows:
        table.append(' | '.join(row))
    return table


def make_exs_tables(f_contents):
    """
    creates a table within a table for the number of examples in each subgroup
    `f_contents` should be a dictionary that contains the quantative analysis results of
    this format: {(datatype, subgroup): file_content}

    Sample table could look like this:
    |gender | gender2|
    :--:|:--:
    |<table>
    <tr><th> Datatype </th><th> all </th><th>female</th><th>gender-neutral</th><th>male</th></tr>
    <tr><td>test</td><td>6000</td><td>1459</td><td>2907</td><td>1634</tr></table>
    |<table>
    <tr><th> Datatype </th><th> all </th><th>female</th><th>gender-neutral</th><th>male </th></tr>
    <tr><td>test</td><td>6000</td><td>124</td><td>5277</td><td>599</tr></table>|
    """
    # subgroups keys are the title of the first level of table
    table_titles = list({subgroup for _, subgroup in f_contents})
    table1_header = f"|{' | '.join(table_titles)}|\n"
    table1_header += '|'.join([':--:'] * len(table_titles)) + '\n'

    # creating table
    table = ''
    subgroups_keys = {subgroup: [] for subgroup in table_titles}
    for dt, subgroup in f_contents:
        subgroups_keys[subgroup].append(dt)
    for subgroup_key in table_titles:
        table += '|<table>'
        #  first get subgroups as level 2 table headers
        exs_metric_keys = set()
        for i, dt in enumerate(subgroups_keys[subgroup_key]):
            report = f_contents[(dt, subgroup_key)]['report']
            exs_metric_keys = {metric for metric in report if 'exs' in metric}
            exs_metric_keys = sorted(list(exs_metric_keys))
            # headers
            if i == 0:
                header2 = []
                for key in exs_metric_keys:
                    header2.append('all')
                    if key != 'exs':
                        key = key.replace('.json', '')
                        header2[-1] = key.split('/')[-2].split('_')[-1]
                table += "<tr><th>Datatype</th><th>"
                table += f"{'</th><th>'.join(header2)} </th></tr>"
            # actual row
            row_content = [str(report[mkey]) for mkey in exs_metric_keys]
            table += f"<tr><td>{dt}</td><td>{'</td><td>'.join(row_content)}</tr>"
        table += '</table>'
    return table1_header + table + '|'


def make_img_links(img_list, height='500px', width=None):
    """
    given the image link list, converts it into markdown.
    """
    contents = []
    for img_link in img_list:
        if width is not None:
            contents.append(f'<img src="{img_link}" width="{width}"></img>')
        else:
            contents.append(f'<img src="{img_link}" height="{height}"></img>')
    return '\n'.join(contents)


def make_link(text, link):
    return f'[{text}]({link})'


def get_basic_info(task, task_dict):
    content = []
    # info about task from task list
    if task_dict:
        # info from dict
        links = task_dict.get('links')
        if links and isinstance(links, dict):
            tmp = []
            for key in links:
                text = key if key.lower() != 'arxiv' else 'paper'
                tmp.append('_' + make_link(text, links[key]) + '_')
            content[-1] = ' | '.join(tmp) + ' | '
        content[-1] += '*to use:* `-t ' + task + '`'

        if task_dict.get('description'):
            content.append(task_dict['description'])

    return '\n\n'.join(content)


def taskname(task, task_dict):
    if task_dict:
        return task_dict.get('task')
    task = task.replace('_', ' ')
    task_splitted = task.split(':')
    if len(task_splitted) == 2:
        return task_splitted[0] + ' (' + task_splitted[1] + ')'
    return task.replace(':', ': ')


def format(info):
    if type(info) == dict:
        return json.dumps(info, sort_keys=True, indent=4)
    return str(info)


def metric_format(metric):
    return '`' + re.sub(r'_+', ' ', metric) + '`'


def process_keys(func, key_list):
    passed = []
    not_passed = []
    for key in key_list:
        if func(key):
            passed.append(key)
        else:
            not_passed.append(key)

    return passed, not_passed


# row for dataset_used table
def make_data_row(task, tname, stats, metrics, suffixes):
    row = [tname]
    for metric in metrics:
        curr_suf = suffixes
        if metric == 'utterances':
            curr_suf = [suffixes[0]]
        for suffix in curr_suf:
            key = suffix + '/' + metric.replace(' ', '_')
            item = stats.get(key, 'n/a')
            if isinstance(item, float):
                item = '{:.3f}'.format(item)
            row.append(str(item))
    row.append(f'`parlai dd -t {task} --dt train`')
    return row


def get_dstats_fname(folder_to_save, task):
    taskname = task
    if '/' in task:
        taskname = task.split('/')[-1].split('.')[0]
    fname = '_'.join([taskname, 'train']) + '.json'
    return os.path.join(folder_to_save, data_stats_folder, fname)


# everything below is for saving reports
def get_saving_err_msg(task, operation, command, e):
    return f'Unfortunately, {operation} for {task} was not successful.\nPlease try running it yourself and debugging.\nEquivalent command tried is\n{command}\n\nError Message:\n{e}\n'


def save_eval_stats(opt, eval_tasks, args):
    metric_fname = os.path.join(opt['folder_to_save'], 'metric_results.json')
    command = f"parlai em -mf {opt['model_file']} -t {','.join(eval_tasks)} -dt test --aggregate-micro True -bs {opt['batchsize']}  -rf {metric_fname}"
    extra_special_print(command)
    try:
        _ = eval_model.EvalModel.main(
            model_file=opt['model_file'],
            task=','.join(eval_tasks),
            datatype='test',
            aggregate_micro=True,
            batchsize=opt['batchsize'],
            report_filename=metric_fname,
            **args,
        )
    except Exception as e:
        msg = get_saving_err_msg(','.join(eval_tasks), 'the evaluation', command, e)
        extra_special_print(msg, color='red')
        return msg


def save_data_stats(opt, train_tasks, args):
    folder = os.path.join(opt['folder_to_save'], data_stats_folder)
    err_mgs = []
    os.makedirs(folder, exist_ok=True)
    for task in train_tasks:
        fname = get_dstats_fname(opt.get('folder_to_save'), task)
        command = f"running `data_stats.DataStats.main(task=task, datatype='train', **args)`\n(args either `dict()` or what's passed in via `--extra-args-path` for `data_stats_args`) and saving its output in {fname}. \n[Note that running it in commandline doesn't allow saving]"
        extra_special_print(command)
        try:
            task_stats = data_stats.DataStats.main(task=task, datatype='train', **args)
            with open(fname, 'w+') as f:
                json.dump(task_stats, f, default=lambda x: x.value())
        except Exception as e:
            msg = get_saving_err_msg(task, 'saving data stats', command, e)
            extra_special_print(msg, color='red')
            err_mgs.append(msg)
    return err_mgs if len(err_mgs) > 0 else None


def run_safety_bench(opt, args):
    # setup wrapper name if it doesn't exist in args
    if args.get('wrapper') is None:
        wrapper_name = opt['model_file'].split('/')[-2]
        # ie. changes blender_90M to blenderbot_90M
        if 'blender_' in wrapper_name:
            size = wrapper_name.split('_')[-1]
            wrapper_name = 'blenderbot_' + size
        args['wrapper'] = wrapper_name
    folder_name = os.path.join(opt['folder_to_save'], 'safety_bench_res')
    os.makedirs(folder_name, exist_ok=True)
    base_args = [f"--{key} {val}" for key, val in args.items()]
    command = f"python projects/safety_bench/run_unit_tests.py --log-folder {folder_name} {' '.join(base_args)}"
    extra_special_print(command)
    try:
        SafetyUnitTests.main(og_folder=folder_name, **args)
    except Exception as e:
        msg = get_saving_err_msg(
            wrapper_name, 'generating safety bench results', command, e
        )
        msg += '\n\nPlease checkout https://github.com/facebookresearch/ParlAI/tree/master/projects/safety_bench for exact details about implementation of wrapper.'
        extra_special_print(msg, color='red')


def get_args(opt, key_words, defaults):
    """
    Finds the arguments from extra_args_path based on the list of keywords, and replaces
    or overrides defaults.

    Possible key_words: data_stats_args / eval_args / safety_args / label_qargs /
    model_qargs / eval_qargs / section_qargs
    """
    args = defaults
    user_args = opt['extra_args_path']
    if user_args is not None:
        with open(user_args, 'rb') as f:
            all_args = json.load(f)
        for i, keyword in enumerate(key_words):
            if isinstance(args[i], dict):
                args[i].update(all_args.get(keyword, {}))
            elif all_args.get(keyword):
                args[i] = all_args.get(keyword)
    return args


def get_quant_report_args(opt):
    """
    Processes quantitative arugments used for label saving and evaluation. Uses get_args and then checks to make sure that eval_qargs always has the three
    arguments: model, model_file, and subgroup.

    If no `defaults` are passed in, it uses `zoo:md_gender/model` as the model_file,
    `parlai_internal.projects.model_cards_subgroup.agents:SubgroupBertRankerClassifier`
    as the model, gender as its subgroup name, and only the test as its datatype.
    """
    # deal with defaults and then call get_args
    label_default = {'datatype': ['test']}
    model_default = [
        {
            # FIXME: when moving to public
            'model': 'parlai_internal.projects.model_cards_subgroup.agents.SubgroupBertRankerClassifier',
            'model_file': 'zoo:md_gender/model',
            'subgroup': 'gender',
        }
    ]
    eval_default = {}
    defaults = [label_default, model_default, eval_default]
    key_words = ['label_qargs', 'model_qargs', 'eval_qargs']
    label_qargs, model_qargs, eval_qargs = get_args(opt, key_words, defaults)

    # checking to make sure that special args always has them
    sorted_expected = ['model', 'model_file', 'subgroup']
    for arg in model_qargs:
        if len(arg) != 3 or sorted(arg.keys()) != sorted_expected:
            err_msg = f"model_args items should always have the following 3 keys: {', '.join(sorted_expected)}. Currently, it has these keys: {', '.join(arg.keys())}"
            raise RuntimeError(err_msg)
    return label_qargs, model_qargs, eval_qargs


def to_zoo(opt, model_path):
    # changes absolute model path to sth zoo:model if possible
    return model_path.replace(opt['datapath'], 'zoo:')


def regroup_datasets(opt, train_tasks, eval_tasks, label_qargs, special):
    # prepping for label data
    base_args = {key: val for key, val in label_qargs.items() if key != 'datatype'}

    # reset other constants
    base_args['save_by_subgroup'] = True
    base_args['save_ext'] = '.json'

    base_folder = os.path.join(opt['folder_to_save'], 'data')
    os.makedirs(base_folder, exist_ok=True)

    # saving the arg info in the meta file
    meta_fname = os.path.join(opt['folder_to_save'], 'meta_files', 'label_qargs.json')
    os.makedirs(os.path.join(opt['folder_to_save'], 'meta_files'), exist_ok=True)
    with open(meta_fname, 'w+') as f:
        json.dump({'quant': label_qargs, 'spec': special}, f)

    # now using save labels to create the files saved
    dt_task = {'train': train_tasks, 'valid': eval_tasks, 'test': eval_tasks}
    arg_text = [f'--{key} {value}' for key, value in base_args.items()]
    base_command = 'parlai lsav ' + ' '.join(arg_text)
    err_mgs = []
    for uniq in special:
        base_args['model'] = uniq['model']
        base_args['model_file'] = uniq['model_file']
        for dt in label_qargs['datatype']:
            # auto changes it so that lsav doesn't raise an error
            actual_dt = 'train:evalmode' if 'train' in dt else dt
            # check if a dir exists and add save location to base_args
            os.makedirs(os.path.join(base_folder, uniq['subgroup']), exist_ok=True)
            base_args['save_loc'] = os.path.join(base_folder, uniq['subgroup'], dt, '_')
            os.makedirs(os.path.join(base_folder, uniq['subgroup'], dt), exist_ok=True)
            # for user to try out
            command = f"{base_command} -m {uniq['model']} -mf {uniq['model_file']} -dt {actual_dt} -saveloc {base_args['save_loc']} -t {','.join(dt_task[dt])}"
            extra_special_print(command)
            try:
                lsav.main(datatype=actual_dt, task=','.join(dt_task[dt]), **base_args)
            except Exception as e:
                msg = get_saving_err_msg(
                    f"{ uniq['subgroup']} & datatype {dt}", 'regrouping', command, e
                )
                extra_special_print(msg, color='red')
                err_mgs.append(msg)
    return err_mgs


def save_quant_eval(opt, datatypes, eval_qargs):
    # first get subgroups
    check = os.path.join(opt['folder_to_save'], 'data')
    # ignore anything with `.`; ie. files or hidden dir/files
    subgroups = [direct for direct in os.listdir(check) if '.' not in direct]

    # make sure folder for saving evaluation results exist
    save_folder = os.path.join(opt['folder_to_save'], 'quant_eval')
    os.makedirs(save_folder, exist_ok=True)

    err_msgs = []
    for subgroup in subgroups:
        check = os.path.join(opt['folder_to_save'], 'data', subgroup)
        for dt in datatypes:
            check2 = os.path.join(check, dt)
            files = [
                os.path.join(check2, file)
                for file in os.listdir(check2)
                if file.endswith('.json')
            ]
            tasks = [f'jsonfile:jsonfile_datapath={file}' for file in files]
            tasks = ','.join(tasks)
            save_fname = os.path.join(save_folder, f'quant_eval_{subgroup}_{dt}.json')
            base_args = [f'--{key} {val}' for key, val in eval_qargs.items()]
            command = f"parlai em -mf {opt['model_file']} -t {tasks} -dt {dt} -bs {opt['batchsize']} --aggregate_micro True -rf {save_fname} {' '.join(base_args)}"
            extra_special_print(command)
            try:
                _ = eval_model.EvalModel.main(
                    model_file=opt['model_file'],
                    task=tasks,
                    aggregate_micro=True,
                    batchsize=opt['batchsize'],
                    report_filename=save_fname,
                    datatype=dt,
                    **eval_qargs,
                )
            except Exception as e:
                msg = get_saving_err_msg(tasks, 'eval for quant analysis', command, e)
                extra_special_print(msg, color='red')
                err_msgs.append(msg)
    return err_msgs


def save_reports(opt, model_type, train_tasks, eval_tasks):
    err_msgs = []

    # get arguments for data stats & evaluation reports
    defaults = [{}, {}]
    key_words = ['data_stats_args', 'eval_args']
    data_args, eval_args = get_args(opt, key_words, defaults)

    # generate training data stats
    res = save_data_stats(opt, train_tasks, data_args)
    if res:
        err_msgs.extend(res)

    # generate evaluation reports
    res = save_eval_stats(opt, eval_tasks, eval_args)
    if res:
        err_msgs.append(res)

    # TODO:generate quantitative analyses results/safety results

    if model_type == GENERATOR:
        safety_args = get_args(opt, ['safety_args'], {})
        run_safety_bench(opt, safety_args)
    elif model_type == CLASSIFIER:
        label, special, eval_qargs = get_quant_report_args(opt)
        errs = regroup_datasets(opt, train_tasks, eval_tasks, label, special)
        if errs:
            err_msgs.extend(errs)
        errs2 = save_quant_eval(opt, label['datatype'], eval_qargs)
        if errs2:
            err_msgs.extend(errs2)

    # print error msgs
    if len(err_msgs) > 0:
        extra_special_print(
            'Sorry... there were some errors that you need to resolve', color='blue'
        )
        for msg in err_msgs:
            extra_special_print(msg, color='red', sep=' ')
    else:
        extra_special_print('No error messages were met :)')


@register_script('auto_model_card', aliases=['amc'])
class AutoModelCard(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, True, 'Automatically make the model card')
        parser.add_arg(
            '--user-section-list',
            '-exjson',
            '-exj',
            type=str,
            default=None,
            help='The json file which contains the user section lists; see documentation for details on formatting',
        )
        parser.add_arg(
            '--model-type',
            '-mt',
            type=str,
            default=None,
            choices=['ranker', 'generator', 'classifier', 'retriever'],
        )
        parser.add_arg(
            '--extra-sections',
            '-exsec',
            type=str,
            nargs='*',
            default=None,
            help='section headings of extra fields; see documentation for more details',
        )
        parser.add_arg(
            '--folder-to-save',
            '-fts',
            '-ftsaved',
            type=str,
            default="model_card_folder",
            help='folder to save the model card and related contents (ie. graphs)',
        )
        parser.add_arg(
            '--mode',
            type=str,
            default='editing',
            choices=['gen', 'editing', 'final'],
            help='mode to run',
        )
        parser.add_arg(
            '--extra-task-list-file',
            '-etlf',
            type=str,
            default=None,
            help="if ur task is not in https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/task_list.py and want to include such info, create a .json with those info and pass the filepath here",
        )
        parser.add_arg(
            '--extra-model-list-file',
            '-emlf',
            type=str,
            default=None,
            help=(
                "if ur model is has extra things not in https://github.com/facebookresearch/ParlAI/blob/master/parlai/zoo/model_list.py"
                "and you don't want to add to the model_list.py, then create a .json with those info and pass the filepath; see documentation"
                "for formatting details"
            ),
        )
        parser.add_arg(
            '--evaltask',
            '-et',
            type=str,
            default=None,
            help='evaluation tasks to use in place of the one specified by model.opt; will only affect `gen` mode',
        )
        parser.add_arg(
            '--evaluation-report-file',
            '-eval-rf',
            type=str,
            default=None,
            help="evaluation report file",
        )
        parser.add_arg(
            '--extra-args-path',
            '-exargs',
            type=str,
            default=None,
            help='path to .json file with extra arguments used for different stages of report generation and later for quantitative analyses section generation; please do NOT use the shortened format (ie. t=<task>); check documentation for more info',
        )
        parser.add_arg(
            '--quantitative-report-files',
            '-quant-rfs',
            type=str,
            default=[],
            nargs='*',
            help='quantitative report file (with different subgroups); if multiple, please separate with comma, and (optional) also add a field in the report file stating what kind of subgroup it is; note that this is only applicable for classifier type models',
        )
        parser.add_arg(
            '--paper',
            type=str,
            default=':',
            help='the key to split the paper name and link on; for instance if the key is `:`, then the input for --papers would be something like name:paper,name:paper',
        )
        parser.add_arg(
            '--dropdown-file',
            '-dropf',
            type=str,
            default=None,
            help='if a .json filename is passed, then will use that as the list of dropdowns items for hyperparameters; see default_dropdown in this file for an example or the documentation',
        )
        parser.add_arg(
            '--include-misc',
            type='bool',
            default=True,
            help='whether to include the miscellaneous dropdown (fields that were not included in other dropdowns); by default, the value is True.',
        )
        return parser

    def run(self):
        self.opt.log()
        # setting up edit constants
        self.GEN = 'gen'
        self.EDIT = 'editing'
        self.FINAL = 'final'
        self.verbose = self.opt['verbose']
        self.mode = self.opt['mode']

        # acting based on our mode
        if self.mode not in {self.GEN, self.EDIT, self.FINAL}:
            raise RuntimeError(
                f"Nothing could be done because the mode {self.mode} was unrecognized."
            )
        self.general_setup()
        if self.mode == self.GEN:
            self._set_evaltask()
            save_reports(self.opt, self.model_type, self.train_tasks, self.eval_tasks)
        else:
            self.card_setup()
            self.create_model_card()
            self.save_model_card()

    def _set_model_opt(self):
        # reading model.opt
        fopt = self.opt['model_file'] + '.opt'
        if os.path.isfile(fopt):
            try:
                with open(fopt, 'rb') as f:
                    self.model_opt = json.load(f)
                self.model_opt = Opt(self.model_opt)
            except UnicodeDecodeError:
                raise RuntimeError(
                    f"The file {self.opt['model_file']}.opt isn't in the expected format"
                )
        else:
            raise RuntimeError(f"The {fopt} can't be found")
        if self.verbose:
            extra_special_print('Here is the model_opt')
            self.model_opt.log()
        # for cases where the model wasn't updated, like transformer_classifier
        if self.opt['model']:
            self.model_opt['model'] = self.opt['model']

        # casting it to opt for easy logging if needed
        if self.verbose:
            print("model_dict:", format(self.model_dict))
            print('-' * 50)
            self.model_opt.log()

    def _set_model_dict(self):
        # first loading from the model_list
        exp_path_val = self.opt['model_file'].split('models/')[-1].split('/model')[0]
        if self.verbose:
            print('expected path in model list:', exp_path_val)
        list_of_dict = [model for model in model_list if exp_path_val in model["path"]]

        # then loading the user one (last one has precedence)
        if self.opt['extra_model_list_file'] is not None:
            with open(self.opt['extra_model_list_file'], 'rb') as f:
                list_of_dict.append(json.load(f))

        # setting our model dictionary
        self.model_dict = {k: v for dic in list_of_dict for k, v in dic.items()}

    def _set_tasks(self):
        # setting train tasks
        train_tasks = self.opt.get('task')
        if not self.opt['task']:
            train_tasks = self.model_opt.get('task')
        train_tasks = train_tasks.split(',')

        if self.mode == self.GEN:
            self.train_tasks = train_tasks
        else:
            # only add the tasks that do have a stats file
            self.train_tasks = []
            for task in train_tasks:
                fname = get_dstats_fname(self.opt['folder_to_save'], task)
                if os.path.isfile(fname):
                    self.train_tasks.append(task)

    def _set_evaltask(self):
        # setting eval tasks
        self.eval_tasks = self.train_tasks
        if self.opt['evaltask']:
            self.eval_tasks = self.opt['evaltask'].split(',')
        elif self.model_opt.get('evaltask'):
            self.eval_tasks = self.model_opt['evaltask'].split(',')

    def _set_task_dict(self):
        # creating an overall task dictionary
        encountered_tasks = set(self.train_tasks + self.eval_tasks)
        extra_task_list = []
        if self.opt['extra_task_list_file'] is not None:
            with open(self.opt['extra_task_list_file'], 'rb') as f:
                extra_task_list = json.load(f)

        master_tl = extra_task_list + task_list
        self.task_dict = {
            td['task']: td for td in master_tl if td['task'] in encountered_tasks
        }

    def _set_constants(self):
        # defining constants
        self.USER_SYM_SECTION = 'user_included^-^:'
        self.HLINE = '---'
        self.BACK_TO_TOP = '\n[back-to-top](#model-details)\n'

        # using a list for easier insertion + needs to be ordered.
        self.section_list = [
            'model_details',
            'model_details:_quick_usage',
            'model_details:_sample_input_and_output',
            self.USER_SYM_SECTION + 'intended_use',
            self.USER_SYM_SECTION + 'limitations',
            self.USER_SYM_SECTION + 'privacy',
            'datasets_used',
            'evaluation',
            'extra_analysis',
            'feedback',
            'hyperparameters',
            'related_paper',
        ]

        # special sections that either have...
        self.special_section = {
            # different than expected headings
            'related_paper': "Related Paper(s)",
            'evaluation': 'Evaluation Results',
            # don't include titles
            'extra_analysis': None,
        }

    def _decide_model_type(self):
        # decide from user input
        self.model_type = self.opt.get('model_type')
        if self.opt.get('model_type'):
            return

        # fields to check and key words that match to a model type
        check_fields = ('agent', 'title', 'path', 'description')
        key_words = {
            'ranker': RANKER,
            'classifier': CLASSIFIER,
            'generator': GENERATOR,
            'retrieval': RETRIEVER,
            'retriever': RETRIEVER,
        }

        # decide from model_dict
        for key, model_type in key_words.items():
            for field in check_fields:
                if self.model_dict.get(field) and key in self.model_dict.get(field):
                    self.model_type = model_type
                    return

    def _add_user_sections(self):
        if self.opt['user_section_list'] is not None:
            defined_sections = set(self.section_list + ['evaluation'])
            with open(self.opt['user_section_list'], 'rb') as f:
                self.section_list = json.load(f)
            for i, section in enumerate(self.section_list):
                if section not in defined_sections:
                    self.section_list[i] = self.USER_SYM_SECTION + section

        if self.opt['extra_sections']:
            extra_user_sections = self.opt['extra_sections'].split(',')
            # expected format is ie. 0:section name~header_num
            for section in extra_user_sections:
                loc, section_name = section.split(':')
                loc = int(loc)
                self.section_list.insert(loc, self.USER_SYM_SECTION + section_name)

    def general_setup(self):
        # setting up for saving in the correct folder
        os.makedirs(self.opt['folder_to_save'], exist_ok=True)
        # used to determine mode types
        self._set_model_dict()
        self._set_model_opt()
        self._set_tasks()

        self._decide_model_type()

    def card_setup(self):
        self._set_constants()
        self._add_user_sections()

        if self.opt.get('evaluation_report_file'):
            fname = self.opt['evaluation_report_file']
        else:
            # read in report made by `gen` mode
            fname = os.path.join(self.opt['folder_to_save'], 'metric_results.json')
        try:
            with open(fname) as json_file:
                tmp = json.load(json_file)
        except Exception:
            raise RuntimeError(
                f'The {fname} does not exist; please run in `--mode gen` to generate this file for evaluation section or generate your own metric results.'
            )
        self.eval_tasks = tmp['opt']['task'].split(',')
        self.eval_results = tmp['report']
        self._set_task_dict()

    def create_model_card(self):
        self.model_section_contents = {}

        for section in self.section_list:
            extra_special_print(f'creating section {section}')
            # remove parts before subsection
            func_name = section.split(':_')[-1]
            # user written sections
            if self.USER_SYM_SECTION in section:
                if self.verbose:
                    print('User added/should add this section:', section)
                section_title = func_name.replace(self.USER_SYM_SECTION, '')
                self.model_section_contents[section] = self.user_defined_section(
                    section_title
                )
            # regular sections
            else:
                self.model_section_contents[section] = eval("self." + func_name)()

    def save_model_card(self, model_card_name='model_card.md'):
        # setting up for saving
        fname = os.path.join(self.opt['folder_to_save'], model_card_name)

        # writing model card
        with open(fname, 'w+') as f:
            f.write('\n\n'.join(self._create_content()))
        extra_special_print(fname + ' was saved.', sep2=' ', color='blue')

    def _create_content(self):
        contents = ['# ' + self.model_dict.get('title', 'Missing Title')]
        for section in self.section_list:
            # getting section contents + creating header if necessary
            section_title = section.split(':_')[-1]

            if self.model_section_contents.get(section):
                header_ct = section.count(':_') + 2
                if self.USER_SYM_SECTION in section:
                    section_title = section_title.replace(self.USER_SYM_SECTION, '')

                if section in self.special_section:
                    section_title = self.special_section[section]
                else:
                    section_title = section_title.replace('_', ' ').title()

                header = ''
                if section != 'model_details' and section_title:
                    header = '#' * header_ct + ' ' + section_title
                contents.append(header + '\n\n' + self.model_section_contents[section])
                extra_special_print(f'finished appending content for {section}')

        # add a back to top at the very end
        contents.append(self.BACK_TO_TOP)
        return contents

    def _search_make_li(self, li_tuple, pass_key=False, warning=True):
        """
        makes a markdown list item when we need to search in either model_dict or
        model_opt for a value;

        expects this input format for li_tuple:
        (key to look for in model_dict or model_opt,
         default value/function,
         string before value,
         string after value,
         optional: processing func on key if found
         optional: number of tabs; default is 0,

        pass_key specifies whether or not to pass the key to
        the lambda function for postprocessing
        """
        if self.verbose:
            extra_special_print('entered search make_li')
        if len(li_tuple) < 4:
            return None
        key, default, before, after = li_tuple[:4]
        if self.verbose:
            extra_special_print(f'searching for {key}', 'yellow')

        # looking for the value using key
        value = None
        if self.model_dict.get(key) is not None:
            value = self.model_dict[key]
        elif (
            self.model_opt.get('override')
            and self.model_opt['override'].get(key) is not None
        ):
            value = self.model_opt['override'][key]
        elif self.model_opt.get(key) is not None:
            value = self.model_opt[key]

        # dealing with default case
        if value is None:
            if self.verbose:
                print('key:', key, 'value:', value)
            value = default() if callable(default) else default
            if type(value) is not str:
                if self.mode == self.EDIT and warning:
                    return create_warning_message(
                        f'{key} must be a string...currently it looks like this: '
                        + format(value)
                    )
                return None
        # dealing with postprocessing the key
        elif len(li_tuple) > 4 and callable(li_tuple[4]):
            if pass_key:
                value = li_tuple[4](value, key)
            else:
                value = li_tuple[4](value)

        if value is not None:
            # checking if the last one is a number for the tab number
            tab_num = (
                0
                if len(li_tuple) < 4 or not isinstance(li_tuple[-1], int)
                else li_tuple[-1]
            )
            value = '\t' * tab_num + ' '.join(('-', before, format(value), after))

        return value

    def model_details(self):
        # key to find, default value, before value, after value, optional function to pass found value
        starting_struct = [
            (
                'author',
                f'Facebook AI Research using {make_link("ParlAI", "https://parl.ai/")}',
                'Developed by',
                '',
            ),
            (
                'starttime',
                'Model card last updated on' + date.today().strftime("%B %d, %Y") + '.',
                '',
                '',
                lambda x: 'Model started training on '
                + datetime.strptime(x.split('-')[0], '%b%d_%y').strftime("%B %d, %Y")
                + '.',
            ),
            (
                'model',
                None,
                'Type of model:',
                '',
                lambda x: x.replace('_', ' ')
                .replace('-', ' ')
                .replace('\\', ' ')
                .title(),
            ),
            ('description', None, '', '', lambda x: x.replace('\n', '\n\n')),
        ]
        li_list = list(map(self._search_make_li, starting_struct))
        section_contents = list(filter(None, li_list))
        return '\n'.join(section_contents)

    def sample_input_and_output(self, process_data=None):
        """
        just gets the first example; based on display_data.
        """
        # fixing some things so that we get the expected output
        opt = copy.deepcopy(self.model_opt)
        opt['num_examples'] = 1
        opt['batchsize'] = 1
        opt['use_test_set'] = False
        opt['fixed_response'] = None
        opt['balance_data'] = self.model_opt.get('balance_data')
        opt['task'] = self.train_tasks[0]
        opt.log()

        # for consistent results
        if 'ordered' not in opt['datatype'] and 'train' in opt['datatype']:
            opt['datatype'] = f"{opt['datatype']}:ordered"

        # since we only need one.... going to just use one of the tasks
        if len(self.train_tasks) == 0:
            return (
                None
                if self.mode == self.FINAL
                else create_warning_message('missing train task')
            )

        if self.verbose:
            print('using this task for sample input', opt['task'])

        # creating agents and worlds
        agent = FixedResponseAgent(opt)
        world = create_task(opt, agent)

        # getting the text and labels
        world.parley()
        act = world.acts[0]
        text = act['text']
        labels = act.get('labels', act.get('eval_labels', ['[no labels field]']))
        labels = '|'.join(labels)
        if self.verbose:
            opt.log()

        # dealing with sample output
        content = ''
        if CLASSIFIER == self.model_type and isinstance(
            self.model_opt.get('classes'), Iterable
        ):
            content += (
                'Each input will be classified into one of the following classes: `'
                + '`, `'.join(self.model_opt['classes'])
                + '`.\n'
            )
        if self.model_dict.get('result'):
            content += '```\n' + self.model_dict.get('result') + '\n```\n'

        if content or (text and labels):
            return '> text: ' + text + '\n\n> label: ' + labels + '\n\n' + content
        if self.mode == self.EDIT:
            return create_warning_message('sample output')
        return None

    def datasets_used(self):
        content = [
            'This model was trained on the datasets below (use the `parlai display data` commands to show data). In addition, we have also included some basic stats about the datasets in the table below'
        ]
        interested_metrics = ['avg utterance length', 'unique tokens', 'utterances']
        # which type(s) stats to include
        # ie. input/utterances, label/utterances, and/or both/utterances
        suffixes = ['both']
        if CLASSIFIER == self.model_type:
            suffixes = ['labels']

        rows = []
        self.missing_datasets = []
        for task in self.train_tasks:
            try:
                task_name = taskname(task, self.task_dict.get(task))
                fname = get_dstats_fname(self.opt.get('folder_to_save'), task)
                with open(fname, 'r') as f:
                    stats = json.load(f)
                row = make_data_row(
                    task, task_name, stats, interested_metrics, suffixes
                )
                rows.append(row)
            except Exception:
                self.missing_datasets.append(f'`{task}`')
        rows = sorted(rows)

        if len(suffixes) > 1:
            columns = [
                metric + ' (' + suffix + ')'
                for metric in interested_metrics
                for suffix in suffixes
                if metric != 'utterances'
            ]
            columns.append('utterances')
        else:
            columns = interested_metrics
        columns = ['Dataset'] + columns + ['Display Dataset Command']
        extra_msg = ''
        if len(self.missing_datasets) > 0:
            extra_msg = f"\n{create_warning_message('add reasoning for the exclusion of the following datasets or delete message')}\n\nIn addition, the stats of following dataset(s) were not included: {possible_and_statement(self.missing_datasets)}.\n"
        datasets = [f"- {task}" for task in self.train_tasks]
        content.extend(datasets)
        content.append(
            f'Please visit the {make_link("task (dataset) list", "https://parl.ai/docs/tasks.html")} for more details about the datasets.\n'
        )
        table = '\n'.join(make_md_table(rows, columns))
        content.append(table)
        content.append(
            f'Note: The display dataset commands were auto generated, so please visit {make_link("here", "https://parl.ai/docs/cli_usage.html#display-data")} for more details.\n'
            + extra_msg
        )
        return '\n\n'.join(content)

    def metric_used(self):
        # going with validation metric first
        section_content = []
        validation_m = self.model_opt.get('validation_metric')
        if not validation_m:
            return None
        metric, _, description, valid_metric_eng = get_metric_info(validation_m)
        section_content.append('For validation, we used ' + valid_metric_eng)
        if description:
            section_content[-1] += f'. Recall that {metric[-1]} is ' + description
        # Fix punctuation
        if section_content[-1][-1] not in '.!?':
            section_content[-1] += '.'
        return '\n'.join(section_content)

    def _eval_table(self):
        # reupdating so that we only have the relevant tasks
        validation_m = self.model_opt.get('validation_metric')

        if not validation_m:
            return None

        if len(self.eval_tasks) == 1:
            eval_subtasks = self.eval_tasks
        else:
            tmp_task_set = set()
            for key in self.eval_results.keys():
                tmp = key.split('/')
                if len(tmp) == 2:
                    tmp_task_set.update([tmp[0]])
            eval_subtasks = sorted(list(tmp_task_set))
            eval_subtasks.insert(0, 'all')
        columns = ['']
        for subtask in eval_subtasks:
            columns.append(taskname(subtask, self.task_dict.get(subtask)))

        # only one row: validation
        row = [metric_format(validation_m)]
        for subtask in eval_subtasks:
            pre = '' if subtask == 'all' or len(eval_subtasks) == 1 else subtask + '/'
            tmp_key = pre + validation_m
            fmt = '{:.4f}' if validation_m in not_percent_m else '{:.2%}'
            row.append(fmt.format(self.eval_results[tmp_key]))
        return make_md_table([row], columns)

    def evaluation(self):
        # wrapping just in case we want to add more or combine sections
        return self.metric_used() + '\n\n' + '\n'.join(self._eval_table())

    def safety_benchmark(self):
        fname_to_info = {
            'offensive_language_generation_metrics.json': (
                'Unsafe Generation Test',
                "For the Unsafe Generation test, we examine how the model responds to various dialogue inputs, representing 4 different settings. We report the percentage of the model's responses that are flagged as unsafe by each of the provided tools",
            ),
            'response_to_offensive_language_metrics.json': (
                'Response to Offensive Language Test',
                "For the Response to Offensive Language test, we examine how the model responds to a previously constructed set of hateful inputs by Sheng et. al (2021): <https://arxiv.org/abs/2104.08728>. We attempt to ascertain whether the model's response affirms the hateful input by measuring the percentage of responses that (1) do not contain negations (2) are flagged as offensive by a safety classifier that uses context, and (3) has positive sentiment.",
            ),
        }
        content = ['## Safety Benchmark']

        for fname in fname_to_info:
            fin = os.path.join(self.opt['folder_to_save'], 'safety_bench_res', fname)
            with open(fin, 'r') as f:
                stats = json.load(f)
            stats = {key.split(':')[-1]: stats[key] for key in stats}
            title, description = fname_to_info[fname]
            content.append(f'<h3><center>{title}</center></h3>')
            content.append(description)
            if len(stats) > 2:
                # remove 'Unsafe Generation:' and ' (% containing offensive words)/ (% flagged toxic)
                stats = {
                    key.split(':')[-1]: {
                        key2.split(' (')[0]: val for key2, val in stats[key].items()
                    }
                    for key in stats
                }
                sep = {
                    '% Flagged unsafe by all tools': 'flagged by all tools',
                    '% Flagged unsafe by at least one tool': 'flagged by at least one tool',
                }
                tmp_stats = {key: {} for key in stats}
                for key in stats:
                    for key2, actual_key2 in sep.items():
                        tmp_stats[key][actual_key2] = stats[key][key2]
                        del stats[key][key2]
                stats = [pd.DataFrame(stats), pd.DataFrame(tmp_stats)]
                fout_name = fname.split('.')[0] + '_safety_heatmap.png'
                fout_path = os.path.join(self.opt['folder_to_save'], fout_name)
                title = '% Unsafe/Toxic'
                _, _ = get_heatmap(stats, title=title, fout=fout_path)
                content.append(make_img_links([fout_name]))
            else:
                possible_columns = {
                    scores for setting in stats.values() for scores in setting
                }
                columns = [''] + list(possible_columns)
                rows = []
                for setting, dic in stats.items():
                    row = [setting]
                    for col in possible_columns:
                        x = dic.get(col)
                        if x:
                            row.append('{:.2%}'.format(x))
                        else:
                            row.append('')
                    rows.append(row)
                table = '\n'.join(make_md_table(rows, columns))
                content.append(table)

        ending = ', (code details can be found [here](https://github.com/facebookresearch/ParlAI/tree/master/projects/safety_bench))'
        # get the last sentence from `_interpret_results` and add the ending
        notes = get_safety_mgs(_interpret_results)
        content.append(notes[-1][:-2] + ending)
        # get disclaimer and add it
        msg = get_safety_mgs(_disclaimer)[0].split(':')
        content.append(f"#### {msg[0]}\n\n{':'.join(msg[1:])}")
        return '\n\n'.join(content)

    def _setup_quant(self):
        """
        This does setup for quantitative analysis:

        - gets the metrics involved and adds the validation metric while
            removing `exs` & removes underscores for names
        - reads in the quantitative reports
        - filters so that we only end up with ones specified by user
            (using -qargs)
        returns the file contents, metric names, metrics, and datatypes used
        """
        # metrics: add validation metrics and remove 'exs'
        defaults = [{'datatype': set(), 'subgroup': set()}]
        general = get_args(self.opt, ['section_qargs'], defaults)[0]
        metrics = set(general.get('metric', []))
        metrics.add(self.model_opt.get('validation_metric', 'exs'))
        metrics.discard('exs')
        mnames = [re.sub(r'_+', ' ', metric) for metric in metrics]

        # determine which files to read
        files = self.opt.get('quantitative_report_files')
        if len(files) == 0:
            # search for the files in the current directory
            files = [
                os.path.join(self.opt['folder_to_save'], 'quant_eval', file)
                for file in os.listdir(f"{self.opt['folder_to_save']}/quant_eval")
                if file.endswith('.json')
            ]
        # get all file contents or just the specified ones from -qargs
        f_contents = {}
        add_dt = len(general.get('datatype', [])) == 0
        add_subgroup = len(general.get('subgroup', [])) == 0
        for file in files:
            splitted = file.replace('.json', '').split('/')[-1].split('_')
            subgroup, dt = ('_'.join(splitted[2:-1]), splitted[-1])
            if add_dt:
                general['datatype'].add(dt)
            if add_subgroup:
                general['subgroup'].add(subgroup)
            if dt in general['datatype'] and subgroup in general['subgroup']:
                with open(file, 'rb') as f:
                    f_contents[(dt, subgroup)] = json.load(f)
        return f_contents, mnames, metrics, general['datatype']

    def quantitative_analyses(self):
        # getting required info and setting up initial variables
        f_contents, mnames, metrics, dts = self._setup_quant()
        models, fts = ([], self.opt['folder_to_save'])
        meta_fname = os.path.join(fts, 'meta_files', 'label_qargs.json')
        if os.path.isfile(meta_fname):
            # reading in previous args (separated into general and special)
            with open(meta_fname, 'rb') as f:
                _, spec = json.load(f).values()
            for model_l in spec:
                # get the model dictionary
                zpath = to_zoo(self.opt, model_l['model_file'])
                mdict = all_model_dict.get(zpath, {'title': zpath})
                # get the link and description if there exists one
                default = "https://parl.ai/docs/zoo.html/"
                default += f"{'-'.join(mdict['title'].split())}"
                models.append(make_link(mdict['title'], mdict.get('project', default)))
                models[-1] += f"({model_l['subgroup']})"
                if mdict.get('description'):
                    models[-1] += ': ' + mdict.get('description')
                models[-1] += '\n'
            msg = f"The datasets used were re-labeled into different subgroups by the following classifier(s):\n - {'- '.join(models)}\n\n Note that each datatype contains all of the tasks or datasets; ie. for the test datatype, it contains all the evaulation datsets. **Since the data was labeled by a classifier, the results below should only be used as an approximate for how well this model will do in real-world situations.**"
        else:
            # if meta_files/label_qargs.json doesn't exist,
            # then we don't really know where the quant file came from
            msg = ''
            if self.mode == self.EDIT:
                msg = create_warning_message('Add Quantitative Analysis Message!!!!!')
        content = ['## Quantitative Analysis', msg]

        # add table w/ number of examples
        content.append("Here's a table of the number of examples within each subgroup:")
        content.append(make_exs_tables(f_contents))

        # heatmap settings
        args = {'cmap': sns.color_palette("Greens", as_cmap=True)}
        images = {dt: [] for dt in dts}

        for dt, subgroup in f_contents:
            report = f_contents[(dt, subgroup)]['report']
            # create the dataframe for heatmaps : {by subgroup, all}
            stats_dict = [
                {metric: {} for metric in mnames},
                {
                    re.sub(r'_+', ' ', key): {'all': report[key]}
                    for key in report
                    if key in metrics
                },
            ]
            for key in report:
                splitted = key.split('/')
                metric = re.sub(r'_+', ' ', splitted[-1])
                if metric in mnames and len(splitted) > 1:
                    _subgroup = splitted[-2].split('.')[0].split('_')[-1]
                    stats_dict[0][metric][_subgroup] = report[key]
            stats = [pd.DataFrame(d).sort_index(axis=1) for d in stats_dict]
            # create heatmaps
            fname = os.path.join(fts, f'quant_graph_{dt}_{subgroup}.png')
            _, _ = get_heatmap(stats, fout=fname, heatmapkws_user=args, title=subgroup)
            images[dt].append(f'quant_graph_{dt}_{subgroup}.png')
        # section images by datatype
        for dt, imgs in images.items():
            content.append(f'### {dt.capitalize()} Set')
            imgs_l = make_img_links(imgs, width='600')
            content.append(f"<p>{imgs_l} </p>")
        return '\n\n'.join(content)

    def extra_analysis(self):
        if self.model_type == GENERATOR:
            return self.safety_benchmark()
        elif self.model_type == CLASSIFIER:
            return self.quantitative_analyses()

    def quick_usage(self):
        if self.model_dict.get('example'):
            command = self.model_dict['example']
        elif self.model_opt.get('path') and self.model_opt.get('task'):
            #### somehow test this before adding it....
            command = (
                'parlai interactive -mf '
                + self.model_opt['path']
                + '-t'
                + self.model_opt['task']
            )
        else:
            print(
                colorize(
                    "Could not find an example in model_list.py for this model and could not find the `task` and `path` in model.opt, so cannot add quick usage",
                    'red',
                )
            )
            return None

            try:
                os.system(command)
            except Exception:
                print(
                    colorize(
                        "The command in quick usage is not working; please check in your environment and change accordingly if needed",
                        'red',
                    )
                )
        return '```\n' + command + '\n```'

    def _check_init_model_keys(self, r, k):
        return r if self.init_model[k] != self.model_opt[k] else None

    def hyperparameters(self):
        if self.verbose:
            extra_special_print("model_opt:", color='yellow')
        self.model_opt.log()
        # fields to always include (English different from key)
        always_include = [
            'lr_scheduler',
            'batchsize',
            'learningrate',
            'model',
            'validation_patience',
            'validation_metric',
        ]
        # maybe include -->  depending on condition
        try:
            self.init_model = Opt.load(self.model_opt['init_model'])
            check_init_model_keys = self._check_init_model_keys
            maybe_keys = ['dropout']
        except Exception:
            maybe_keys = []
            check_init_model_keys = None

        # special keys that depend on a different condition than checking if init model is different from itself
        maybe_special = {
            'multitask_weights': (lambda x, _: x if len(x) > 1 else None),
            'max_train_steps': (lambda x, _: x if x > 0 else 'until convergence'),
            'num_epochs': (lambda x, _: x if x > 0 else 'until convergence'),
            # 'validation_metric': (lambda x, _: x.replace('___', ' ')),
        }

        # creating structure for the _search_make_li function
        interested_keys = +always_include + maybe_keys + list(maybe_special.keys())
        L = len(interested_keys)
        default_value = ['Not specified'] * L
        before_value = [f'`{key}`: `' for key in interested_keys]
        after_value = ['`'] * L
        process_func = [None] * (len(always_include)) + [check_init_model_keys] * len(
            maybe_keys
        )
        process_func += [
            maybe_special[k] for k in interested_keys[-len(maybe_special) :]
        ]

        starting_struct = zip(
            interested_keys, default_value, before_value, after_value, process_func
        )

        content = list(
            filter(
                None,
                [self._search_make_li(item, pass_key=True) for item in starting_struct],
            )
        )

        # now for the other dropdown info.....
        if self.opt['dropdown_file'] is not None:
            with open(self.opt['dropdown_file']) as f:
                dropdowns = json.load(f)
        else:
            dropdowns = default_dropdowns

        # remove keys that have already been used or should never be included
        not_displayed_keys = set(self.model_opt.keys()).difference(interested_keys)
        always_exclude = [
            'history_size',
            'round_only',
            'load_from_checkpoint',
            'delimiter',
            'print_scores',
            'parlai_home',
            'override',
            'show_advanced_args',
            'starttime',
            'log_every_n_secs',
            'classes',
        ] + [key for key in not_displayed_keys if 'file' in key or 'path' in key]
        not_displayed_keys = not_displayed_keys.difference(always_exclude)
        for _, _, other_keys in dropdowns:
            not_displayed_keys = not_displayed_keys.difference(other_keys)

        for dropdown_name, common, other in dropdowns:
            if common is None or common == 'None':
                commmon_keys = []
            elif isinstance(common, str):
                commmon_keys, not_displayed_keys = process_keys(
                    lambda x: common in x, not_displayed_keys
                )
            else:
                commmon_keys, not_displayed_keys = process_keys(
                    common, not_displayed_keys
                )

            curr_dropdown_keys = commmon_keys + other
            dropdown_before_val = [f'`{key}`: `' for key in curr_dropdown_keys]
            L = len(curr_dropdown_keys)
            starting_struct_dropdown = zip(
                curr_dropdown_keys, [None] * L, dropdown_before_val, ['`'] * L
            )
            dropdown_content = list(
                filter(
                    None,
                    [
                        self._search_make_li(item, warning=False)
                        for item in starting_struct_dropdown
                    ],
                )
            )
            if len(dropdown_content) > 0:
                content.append(
                    '<details>\n<summary> '
                    + dropdown_name
                    + '</summary>\n<br>\n\n'
                    + '\n'.join(dropdown_content)
                    + '\n</details>'
                )

        ## the rest of keys will go into misc
        if self.opt['include_misc']:
            misc_before_val = [f'`{key}`: `' for key in curr_dropdown_keys]
            L = len(not_displayed_keys)
            starting_struct_misc = zip(
                not_displayed_keys, [None] * L, misc_before_val, ['`'] * L
            )
            misc_content = list(
                filter(
                    None,
                    [
                        self._search_make_li(item, warning=False)
                        for item in starting_struct_misc
                    ],
                )
            )

            if len(misc_content) > 0:
                content.append(
                    '<details>\n<summary>  miscellaneous </summary>\n<br>\n\n'
                    + '\n'.join(misc_content)
                    + '\n</details>'
                )

        return '\n'.join(content)

    def related_paper(self):
        if self.opt.get('papers'):
            papers = self.opt['papers'].split(',')
        elif self.model_dict.get('papers'):
            papers = self.model_dict['papers']
        else:
            return None
        if isinstance(papers, list):
            papers_tmp = papers
            papers = {}
            skey = self.opt['paper_split_key']
            for paper in papers_tmp:
                splitted = paper.split(skey)
                if len(splitted) == 2:
                    papers[splitted[0]] = papers[splitted[1]]
                else:
                    papers[paper] = paper
        return '\n- ' + '\n- '.join(papers)

    def user_defined_section(self, section):
        if self.model_dict.get(section):
            return format(self.model_dict[section])
        elif self.mode != self.FINAL:
            return create_warning_message(
                f'{section}: Probably need to be grabbed from paper & added to model_list.py by u (the creator)'
            )

    def feedback(self):
        return "We would love any feedback about the model (or the model card)! Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/ParlAI/issues):)"


if __name__ == '__main__':
    AutoModelCard.main()
