#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script to generate the model card automatically.
"""
from datetime import date, datetime

from parlai.core.metrics import METRICS_DISPLAY_DATA
from parlai.core.worlds import create_task
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.opt import Opt
from parlai.tasks.task_list import task_list
from parlai.utils.strings import colorize
from parlai.zoo.model_list import model_list
import parlai.scripts.data_stats as data_stats
import parlai.scripts.eval_model as eval_model
import traceback
import contextlib
import copy
import io
import json
import math
import os
import re


def make_link(text, link):
    return f'[{text}]({link})'


##########################################
# Constants for generating model cards
##########################################

# arguments that always are true for generating samples
sample_always_args = {
    'num_examples': 1,
    'batchsize': 1,
    'use_test_set': False,
    'fixed_response': None,
}

# for classifiers, keys to add from model.opt
classifier_keys = {
    'classes',
    'classes_from_file',
    'use_test_set',
    'balance_data',
    'single_turn',
}
opt_ignore_keys = classifier_keys.union({'is_debug'})

# model details stucture for _search_make_li
# key to search, default value, before value, after value, processing function (optional), # of tabs (optional)
model_details_struct = [
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
        lambda x: x.replace('_', ' ').replace('-', ' ').replace('\\', ' ').title(),
    ),
]

# metrics that are not precentages
not_percent = [
    'ppl',
    'exs',
    'clen',
    'ctpb',
    'ctps',
    'ctrunclen',
    'exps',
    'llen',
    'lr',
    'ltpb',
    'ltps',
    'ltrunclen',
    'total_train_updates',
    'tpb',
    'tps',
    'ups',
]

# for possible validation metrics that aren't
# in METRICS_DISPLAY_DATA and we want to still include info
extra_metric_info = {
    'ppl': (
        'perplexity',
        'perplexity. Click [here](https://en.wikipedia.org/wiki/Perplexity) for more info',
    )
}

# for safety bench section: {filename: (Section title, description)}
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

# used later to access the other keys
OTHER = 1
# {Dropdown name: (common key function or common key prefix, other keys)
default_hyperparams = {
    'always_include': (
        None,
        [
            'lr_scheduler',
            'batchsize',
            'learningrate',
            'model',
            'validation_patience',
            'validation_metric',
        ],
    ),
    'always_exclude': (
        None,
        (
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
        ),
    ),
    'maybe_special': (
        None,
        {
            'multitask_weights': lambda x, _: x if len(x) > 1 else None,
            'max_train_steps': (lambda x, _: x if x > 0 else 'until convergence'),
            'num_epochs': (lambda x, _: x if x > 0 else 'until convergence'),
        },
    ),
    'model / neural net info': (
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
    'embedding info': ('embedding', []),
    'validation and logging info': ('valid', []),
    'dictionary info/pre-processing': ('dict', []),
    'other dataset-related info': (
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
    'more batch and learning rate info': (lambda x: 'batch' in x or 'lr' in x, []),
    'training info': (
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
    'pytorch info': ('pytorch', []),
}


USER_SYM_SECTION = 'user_included:'

# using a list for easier insertion + needs to be ordered.
section_list = [
    "model_details",
    "model_details:_quick_usage",
    "model_details:_sample_input_and_output",
    USER_SYM_SECTION + "intended_use",
    USER_SYM_SECTION + "limitations",
    USER_SYM_SECTION + "privacy",
    "datasets_used",
    "evaluation",
    "extra_analysis",
    USER_SYM_SECTION + "related_paper",
    "hyperparameters",
    "feedback",
]

# sections that have unique functions
defined_sections = {
    'model_details',
    'quick_usage',
    'sample_input_and_output',
    'datasets_used',
    'evaluation',
    'extra_analysis',
    'quantitative_analyses',
    'safety_benchmark',
    'feedback',
    'hyperparameters',
}

# default messages for certain sections
defaults = {
    "intended_use": "This model is intended for the use of....\t",
    "privacy": "This model has the following privacy concerns....\t",
    "limitations": "This model has has these limitations: ...\t",
}

# special sections that either have...
special_section = {
    # different than expected headings
    'related_paper': "Related Paper(s)",
    'evaluation': 'Evaluation Results',
    # don't include titles
    'extra_analysis': None,
}

# types of functions; I get the strings mixed ups easily
CLASSIFIER = 'classifier'
GENERATOR = 'generator'
RETRIEVER = 'retriever'
RANKER = 'ranker'

# different modes
M_gen = 'gen'
M_edit = 'editing'
M_final = 'final'

# dictionary of all models with their path as the key
all_models = {model['path']: model for model in model_list}
# dictionary of all tasks with their task field as the key
all_tasks = {task['task']: task for task in task_list}

data_stats_folder = 'data_stats'
task_site = 'https://parl.ai/docs/tasks.html'

#######################################
# Printing/String related functions
#######################################


def format_io(sample, keys):
    """
    used to format messages for sample input/output.
    """
    contents = []
    for k in keys:
        v = sample[k]
        if isinstance(v, (int, float)):
            v = str(v)
        if isinstance(v, str):
            v = v.strip()
        elif isinstance(v, list):
            v = ', '.join(v)
        if v:
            contents.append(f"[{k}]: {v}")
    return contents


def to_sublink(heading):
    refine = heading.replace('#', '').strip()
    return '#' + refine.lower().replace(' ', '-')


def make_task_links(task, sep=' | '):
    """
    given a task, generates links based on its all_tasks contents.
    """
    content = []
    if all_tasks.get(task) and all_tasks[task].get('links'):
        content = [make_link(k, v) for k, v in all_tasks[task]['links'].items()]
    return sep.join(content)


def clean_mgs(func, sep='\n\n'):
    """
    first captures what's printed by a function and then cleans up special command-line
    stuff (ie.

    colors)
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func()
    out = f.getvalue()
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    tmp = ansi_escape.sub('', out)
    return list(filter(None, tmp.split(sep)))


def create_dropdown(title, dropdown_list):
    newline = '\n'
    return f"<details> {newline} <summary> {title} </summary>{newline} <br>{newline}{newline}{newline.join(dropdown_list)}{newline}</details>"


def taskname(task):
    """
    gets the display name from all_tasks or tries its best to create a more readable
    name.
    """
    task_dict = all_tasks.get(task)
    if task_dict:
        return task_dict.get('display_name')
    task = task.replace('_', ' ')
    task_splitted = task.split(':')
    if len(task_splitted) == 2:
        return task_splitted[0] + ' (' + task_splitted[1] + ')'
    return task.replace(':', ': ')


def format(info):
    """
    a format function for user defined sections.
    """
    if type(info) == dict:
        return json.dumps(info, sort_keys=True, indent=4)
    return str(info)


def possible_and_statement(lis):
    L = len(lis)
    if L == 0:
        return ''
    if L == 1:
        return lis[0]
    if L == 2:
        return lis[0] + ' and ' + lis[1]
    return ', '.join(lis[:-1]) + ', and ' + lis[-1]


def create_warning(missing):
    return "> :warning: " + missing + ":warning:"


def extra_msg(string, color='green', sep='-', sep2='*', line_width=80):
    msg = sep * line_width + '\n'
    msg += colorize(f' {string} '.center(line_width, sep2), color)
    msg += '\n' + sep * line_width + '\n'
    return msg


def extra_special_print(string, color='green', sep='-', sep2='*', width=80):
    print(extra_msg(string, color, sep, sep2, width))


def metric_format(metric):
    return '`' + re.sub(r'_+', ' ', metric) + '`'


def to_zoo(opt, model_path):
    """
    changes absolute model path to zoo:model if possible.
    """
    zoo = os.path.join(opt['datapath'], 'models') + '/'
    return model_path.replace(zoo, 'zoo:')


def get_dstats_fname(folder_to_save, task, dt='train'):
    """
    gets the data_stats file name.
    """
    taskname = task
    if '/' in task:
        taskname = task.split('/')[-1].split('.')[0]
    fname = '_'.join([taskname, dt]) + '.json'
    return os.path.join(folder_to_save, data_stats_folder, fname)


def get_report_msg(task, fname, e, extra=''):
    return f"ran `{task}` script \n and tried to save its output in {fname}.\n{extra}\n However, encountered this error:\n{e}"


def make_img_links(img_list, height='500px', width=None):
    """
    given the image link list, converts it into markdown.

    Note: uses either height or width, but not both b/c images
    can become out of proportion if we use both
    """
    contents = []
    for img_link in img_list:
        if width is not None:
            contents.append(f'<img src="{img_link}" width="{width}"></img>')
        else:
            contents.append(f'<img src="{img_link}" height="{height}"></img>')
    return '\n'.join(contents)


def get_dataset_info(tasks):
    """
    dataset info comes from guessing where it would be at the tasks site and the
    task_list.py + anything else from the user.
    """
    curr_task_info = []
    for task in tasks:
        # adding the name + attempted link
        tname = taskname(task)
        tsite = task_site + to_sublink(tname)
        curr_task_info.append(f"- [{tname}]({tsite})")
        # adding link
        links = make_task_links(task)
        curr_task_info[-1] += f" ({links})" if links else ''
        # adding description
        if all_tasks.get(task) and all_tasks[task].get('description'):
            curr_task_info[-1] += f": {all_tasks[task]['description']}"
    return curr_task_info


#################################
# Table-Related Functions
#################################
def make_data_row(task, stats, metrics, prefix):
    """
    a row for the datasets_used section.
    """
    row = [taskname(task)]
    for metric in metrics:
        key = prefix + '/' + metric.replace(' ', '_')
        item = stats.get(key, 'n/a')
        if isinstance(item, float):
            item = '{:.3f}'.format(item)
        row.append(str(item))
    row.append(f'`parlai dd -t {task}`')
    return row


def datasets_table(train_tasks, metrics, prefix, fts):
    """
    making the table in datasets_used section
    note: metrics should be w/o '_'
    """
    train_tasks = sorted(train_tasks)
    rows, missing_datasets = ([], [])
    for task in train_tasks:
        try:
            fname = get_dstats_fname(fts, task)
            with open(fname, 'r') as f:
                stats = json.load(f)
            rows.append(make_data_row(task, stats, metrics, prefix))
        except Exception:
            missing_datasets.append(f'`{task}`')
    columns = ['Dataset'] + metrics + ['Display Dataset Command']
    table = '\n'.join(make_md_table(rows, columns))
    return table, missing_datasets


def make_table_header(table_header, align=None, extra='|'):
    align_bars = '---'
    if align == 'center':
        align_bars = ':---:'
    elif align == 'right':
        align_bars = '---:'
    elif align == 'left':
        align_bars = ':---'
    header = extra + ' | '.join(table_header)
    line = ' | '.join([align_bars] * len(table_header))
    return [header, line]


def make_md_table(rows, cols, align='center', extra='|'):
    """
    expect the columns to be a list of headers rows to be a list of lists.
    """
    table = make_table_header(cols, align, extra)
    for row in rows:
        table.append(' | '.join(row))
    return table


def make_html_table(rows, header):
    table = ['<table><tr><th>' + '</th><th>'.join(header) + '</th></tr>']
    for row in rows:
        table.append('<tr><td>' + '</td><td>'.join(row) + '</td></tr>')
    table.append('</table>')
    return "\n".join(table)


#################################
# Graphing functions
#################################


def get_heatmap(stats_dfs, title=None, tfsize=16, heatmapkws_user=None, fout=None):
    # imports
    import seaborn as sns
    import matplotlib.pyplot as plt

    # get vmax and vmin and step
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


#################################
# Setup-related functions
#################################


def get_group_keys(group):
    keys = set()
    for action in group._actions:
        keys.add(action.dest)
    return keys


def get_new_parser(parser, opt, ignore_keys=(), always_keys=()):
    """
    rewrites parser with opt.
    """
    for key in opt:
        add_condition = key not in ignore_keys and key in parser
        if key in always_keys or add_condition:
            parser[key] = opt[key]
    if 'override' in parser:
        for k, v in parser['override'].items():
            if key not in ignore_keys:
                parser[k] = v
        del parser['override']
    return parser


def change_parser_req(parser, key):
    """
    used to change requirements in the parser to its inverse (ie.

    safety bench requires wrapper --> wrapper not required)
    """
    for action in parser._actions:
        if action.dest == key:
            action.required = not action.required
            return parser
    raise ValueError(f"{key} doesn't exist in the given parser")


def setup_args(parser=None) -> ParlaiParser:
    if parser is None:
        parser = ParlaiParser(True, True, 'Automatically generate the model card')
        parser = eval_model.setup_args()
        parser = data_stats.setup_args(parser)

        try:
            import projects.safety_bench.run_unit_tests as safety_tests

            parser = safety_tests.setup_args(parser)
            parser = change_parser_req(parser, 'wrapper')
        except Exception:
            # only adding the wrapper; for the building the website
            parser.add_argument(
                "-w", "--wrapper", type=str, help="Registered name of model wrapper"
            )
    gmc = parser.add_argument_group('Model Card Generation arguments')
    gmc.add_argument(
        '--model-type',
        '-mt',
        type=str,
        default=None,
        choices=['ranker', 'generator', 'classifier', 'retriever'],
        help='type of model',
    )
    gmc.add_argument(
        '--folder-to-save',
        '-fts',
        '-ftsaved',
        type=str,
        default="model_card_folder",
        help='folder to save the model card and related contents (ie. graphs)',
    )
    gmc.add_argument(
        '-et',
        '--evaltask',
        default=None,
        type=str,
        help='task to use for valid/test (defaults to the one used for training)',
    )
    gmc.add_argument(
        '--mode',
        type=str,
        default='editing',
        help='possible modes: gen (generation), editing, final.\nIn addition, for gen mode, we can also add the following to specify which exact reports to run: data_stats, eval, safety, sample, and quant)\n For instance, --mode gen:data_stats:eval',
    )
    gmc.add_argument(
        '--ignore-unfound-tasks',
        '--ignore',
        default=True,
        type='bool',
        help='whether or not to ignore the fromfile, jsonfile, etc. tasks if the task can be found; by default, we will (so True).',
    )
    gmc.add_argument(
        '--evaluation-report-file',
        '-eval-rf',
        type=str,
        default=None,
        help="evaluation report file",
    )
    gmc.add_argument(
        '--extra-args-path',
        '-exargs',
        type=str,
        default=None,
        help='path to .json file with extra arguments used for different stages of report generation and later for quantitative analyses section generation; please do NOT use the shortened format (ie. t=<task>); check documentation for more info',
    )
    gmc.add_argument(
        '--quantitative-report-files',
        '-quant-rfs',
        type=str,
        default=[],
        nargs='*',
        help='quantitative report file (with different subgroups); if multiple, please separate with comma, and (optional) also add a field in the report file stating what kind of subgroup it is; note that this is only applicable for classifier type models',
    )
    gmc.add_argument(
        '--include-misc',
        type='bool',
        default=True,
        help='whether to include the miscellaneous dropdown (fields that were not included in other dropdowns); by default, the value is True.',
    )
    gmc.add_argument(
        '--quant-metrics',
        type=str,
        default=[],
        help='Other metrics to include in the quantitative analysis',
    )
    return parser


def decide_model_type(opt, model_dict):
    """
    based on key words in the model dictionary, we try to figure out what type of model
    this is.

    Note: can later try to make it more accurate by tracing the model's ancestors
    """
    # decide from user input
    if opt.get('model_type'):
        return opt['model_type']

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
            if model_dict.get(field) and key in model_dict.get(field):
                return model_type


###################################
# Script for generating model card
###################################


@register_script('generate_model_card', aliases=['gmc'])
class GenerateModelCard(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        self.opt.log()
        self.verbose = self.opt['verbose']
        self.mode = self.opt['mode'].split(':')[0]

        self.general_setup()
        if self.mode == M_gen:
            self._set_evaltask()
            jobs, args = self._gen_jobs()
            self.save_reports(jobs, args)
        elif self.mode in {M_edit, M_final}:
            # card setting up
            self._set_sections_info()
            self._set_eval()
            self._set_validation_metric()

            # creating & saving content
            self.create_model_card()
            self.save_model_card()

    ##########################################
    # general setup-related class functions
    ##########################################

    def general_setup(self):
        self._setup_args()
        # setting up for saving in the correct folder
        os.makedirs(self.opt['folder_to_save'], exist_ok=True)
        self._add_user_model_tasks()
        self._set_model_dict()
        self._set_model_opt()
        self.ignore_task = self.opt['ignore_unfound_tasks']
        self._set_train_tasks()
        # actually deciding model type
        self.model_type = decide_model_type(self.opt, self.model_dict)

    def _setup_args(self):
        """
        gets the extra arguments.
        """
        user_args = self.opt['extra_args_path']
        if user_args is None:
            user_args = os.path.join(self.opt['folder_to_save'], 'args.json')

        try:
            # now setting up args.json
            with open(user_args, 'rb') as f:
                self.all_args = json.load(f)
        except Exception:
            self.all_args = {}

    def _add_user_model_tasks(self):
        """
        updates all_models and all_tasks based on the extra args being passed in.
        """
        # get relevant args using `get_args`
        keywords = {'extra_models': {}, 'extra_tasks': {}}
        user_input = self.get_args(keywords)
        # adding user input to all_models and all_tasks
        for path, dict in user_input['extra_models'].items():
            all_models[path].update(dict)
        for task, dict in user_input['extra_models'].items():
            all_tasks[task].update(dict)

    def _set_model_dict(self):
        """
        find model dictionary from model_list.py; should be run after
        `self._add_user_model_tasks`
        """
        mf, self.model_dict = (self.opt['model_file'], {})
        exp_path = to_zoo(self.opt, mf)
        if self.verbose:
            print('expected path in model list:', exp_path, 'or', mf)
        if all_models.get(exp_path):
            self.model_dict.update(all_models[exp_path])
        elif all_models.get(mf):
            self.model_dict.update(all_models[mf])

    def _set_model_opt(self):
        """
        read and save model.opt as an attribute Also updates the model with the
        overridden sections for easier access later on.

        Also updates the --model if it's not updated
        """
        # reading model.opt
        fopt = self.opt['model_file'] + '.opt'
        if not os.path.isfile(fopt):
            raise RuntimeError(f"The {fopt} can't be found")
        try:
            with open(fopt, 'rb') as f:
                model_opt = json.load(f)
            self.model_opt = Opt(model_opt)
        except UnicodeDecodeError:
            raise RuntimeError(fopt + " isn't in the expected format")

        # override with the override field
        self.model_opt.update(self.model_opt.get('override', {}))

        # make sure that if there's a classes, then there should be a classes_from_file
        if 'classes' in self.model_opt and 'classes_from_file' not in self.model_opt:
            self.model_opt['classes_from_file'] = None

        if self.verbose:
            extra_special_print('model.opt')
            self.model_opt.log()
        # for cases where the model wasn't updated, like transformer_classifier
        if self.opt['model']:
            self.model_opt['model'] = self.opt['model']

    def _set_train_tasks(self):
        # setting train tasks
        train_tasks = self.opt.get('task')
        if not train_tasks:
            train_tasks = self.model_opt.get('task', '')
        while not train_tasks or len(train_tasks) == 0:
            msg = "Please enter training dataset/test (none were passed in or found in model.opt): "
            train_tasks = input(msg)

        # process tasks from any internal to external
        self.train_tasks, tmp = ([], train_tasks.split(','))

        for task in tmp:
            processed = self.process_task(task)
            if processed:
                self.train_tasks.append(processed)
            else:
                msg = f"dropping training task {task}"
                extra_special_print(msg, 'yellow')

        if self.mode != M_gen:
            # only add the tasks that do have a stats file
            self.train_tasks, train_tasks = ([], self.train_tasks)
            for task in train_tasks:
                fname = get_dstats_fname(self.opt['folder_to_save'], task)
                if os.path.isfile(fname):
                    self.train_tasks.append(task)

    def process_task(self, task):
        """
        tries to remap tasks to their external version, and then may ignore the tasks
        w/o ext.

        version depending on `ignore_task`
        """
        # processing tasks so that no arguments are included
        # unless it's a fromfile or jsonfile one
        if 'fromfile:' in task or 'jsonfile:' in task or 'internal:' in task:
            return None if self.ignore_task else task
        return task

    ##########################################
    # generation setup-related class functions
    ##########################################

    def _set_evaltask(self):
        # setting eval tasks
        eval_tasks = self.train_tasks
        if self.opt.get('evaltask'):
            eval_tasks = self.opt['evaltask'].split(',')
        elif self.model_opt.get('evaltask'):
            eval_tasks = self.model_opt['evaltask'].split(',')

        # since train tasks were already processed, we can return
        if eval_tasks == self.train_tasks:
            self.eval_tasks = eval_tasks
            return

        self.eval_tasks = []
        for task in eval_tasks:
            processed = self.process_task(task)
            if processed:
                self.eval_tasks.append(processed)
            else:
                msg = f"dropping evaluation task {task}"
                extra_special_print(msg, 'yellow')

    def _gen_jobs(self):
        """
        generating the jobs to be done in the report saving mode.
        """
        splitted = self.opt['mode'].split(':')[1:]
        # job name: None or default struct for getting arguments
        all_jobs = {
            'data_stats': None,
            'eval': None,
            'safety_bench': None,
            'sample': None,
        }
        if len(splitted) > 0:
            jobs = {job for job in splitted if job in all_jobs}
        else:
            jobs = copy.deepcopy(set(all_jobs.keys()))
        if self.model_type != GENERATOR:
            jobs.discard('safety_bench')
        key_defaults = {(job + '_args'): all_jobs[job] for job in jobs}
        # adding a general field for later use
        key_defaults['general'] = {}
        args = self.get_args(key_defaults)
        return jobs, args

    #################################
    # Report Saving Functions
    #################################

    def save_data_stats(self):
        # setting up things needed for each task
        err_mgs = []
        tasks = set(self.train_tasks + self.eval_tasks)
        folder = os.path.join(self.opt['folder_to_save'], data_stats_folder)
        os.makedirs(folder, exist_ok=True)
        for task in tasks:
            fname = get_dstats_fname(self.opt['folder_to_save'], task)
            # setting up args for data_stats
            parser = data_stats.setup_args().parse_args([])
            if self.model_type == CLASSIFIER:
                parser = get_new_parser(
                    parser, self.model_opt, always_keys=classifier_keys
                )
            parser = get_new_parser(parser, self.opt, opt_ignore_keys)
            parser['task'] = task
            parser['batchsize'] = 1  # if it's changed, it will give sometimes an error

            if self.verbose:
                extra_special_print(f"{task}: passing in the following")
                Opt(parser).log()
            try:
                # run the script and save the stats
                task_stats = data_stats.obtain_stats(parser)
                with open(fname, 'w+') as f:
                    json.dump(task_stats, f, default=lambda x: x.value())
            except Exception:
                e = traceback.format_exc()
                extra_msg = "[Note that running it in commandline doesn't allow saving]"
                msg = get_report_msg('data_stats', fname, e, extra_msg)
                extra_special_print(msg, color='red')
                err_mgs.append(msg)
        return err_mgs if len(err_mgs) > 0 else None

    def save_eval(self):
        fname = os.path.join(self.opt['folder_to_save'], 'eval_results.json')
        # setting up args for evaluation
        parser = eval_model.setup_args().parse_args([])
        if self.model_type == CLASSIFIER:
            parser = get_new_parser(parser, self.model_opt, always_keys=classifier_keys)
        parser = get_new_parser(
            parser, self.opt['override'], opt_ignore_keys, always_keys={'data_parallel'}
        )
        parser['override'] = self.opt['override']
        parser['ignore_labels'] = self.opt.get('ignore_labels')
        parser['task'] = ','.join(self.eval_tasks)
        parser['datatype'] = 'test'
        parser['aggregate_micro'] = True
        parser['report_filename'] = fname

        if self.verbose:
            extra_special_print(f"passing in the following")
            Opt(parser).log()

        try:
            # running evaluation
            _ = eval_model.eval_model(parser)
        except Exception:
            e = traceback.format_exc()
            msg = get_report_msg('eval_model', fname, e)
            extra_special_print(msg, color='red')
            return [msg]

    def save_safety_bench(self):
        from projects.safety_bench.utils.wrapper_loading import setup_wrapper_registry
        import projects.safety_bench.run_unit_tests as safety_tests

        folder_name = os.path.join(self.opt['folder_to_save'], 'safety_bench_res')
        os.makedirs(folder_name, exist_ok=True)
        # adding this so that we can easily access the wrappers
        # setting up args for safety_bench
        wrapper = f"-w {self.opt.get('wrapper', '')}"
        parser = safety_tests.setup_args().parse_args(wrapper.split())
        parser = get_new_parser(parser, self.opt, ignore_keys={'batchsize'})
        parser['log_folder'] = folder_name

        if self.verbose:
            extra_special_print(f"passing in the following")
            Opt(parser).log()

        try:
            setup_wrapper_registry()
            safety_tests.run_safety_unit_tests(parser)
        except Exception:
            e = traceback.format_exc()
            msg = get_report_msg('safety_bench', folder_name, e)
            msg += '\n\nPlease checkout https://github.com/facebookresearch/ParlAI/tree/main/projects/safety_bench for exact details about implementation of wrapper.'
            extra_special_print(msg, color='red')
            return [msg]

    def save_sample(self):
        # setting up args
        parser = copy.deepcopy(self.model_opt)
        ignore_keys = {'model'}
        ignore_keys.update(opt_ignore_keys)
        parser = get_new_parser(parser, self.opt, ignore_keys)
        parser.update(sample_always_args)
        parser['task'] = self.train_tasks[0]

        # for consistent results
        if 'ordered' not in parser['datatype'] and 'train' in parser['datatype']:
            parser['datatype'] = f"{parser['datatype']}:ordered"

        # removing model file so that create agents must
        # use the current parser, not model.opt from file b/c
        # sometimes there are missing args that we want to add
        parser['model_file'] = None

        if self.verbose:
            extra_special_print("sample's model.opt")
            parser.log()
            print('using this task for sample input', parser['task'])

        # creating agents and worlds
        agent = create_agent(parser)
        world = create_task(parser, agent)

        # getting text and saving labels
        world.parley()
        act = world.acts[0]
        fname = os.path.join(self.opt['folder_to_save'], 'sample.json')
        with open(fname, 'w+') as f:
            json.dump(act, f)

    def save_reports(self, jobs, args):
        """
        jobs should be {job_names: (arguments)} and args should be.

        {job_names + '_args': arguments read from user}

        Possible job_names: data_stats, eval, safety_bench, sample, quant_analysis

        Note: self.save_job_name should be defined (ie. self.save_data_stats)
        """
        err_msgs = []
        for job_name in jobs:
            # get function name and argument keys
            func = 'self.save_' + job_name
            arg_key = job_name + '_args'
            extra_special_print(f"Working on {func}")
            # decide to pass keys or not
            if args.get(arg_key):
                user_args = copy.deepcopy(args[arg_key])
                errs = eval(func)(user_args)
            else:
                errs = eval(func)()

            if errs:
                err_msgs.extend(errs)

        #  print error messages or okay message
        if len(err_msgs) > 0:
            extra_special_print('There are some errors to be resolve', color='blue')
            for msg in err_msgs:
                extra_special_print(msg, color='red', sep=' ')
        else:
            extra_special_print('No error messages were met.')

    ##########################################
    # model card setup-related class functions
    ##########################################

    def _set_sections_info(self):
        """
        gets the user's list of sections, dropdowns re-ordering, and info about
        datasets_used and set ups relevant attributes along the way.
        """
        key_defaults = {'user_section_list': [], 'dropdowns': {}, 'datasets_used': {}}
        args = self.get_args(key_defaults)

        # set up dropdowns
        self.hyperparams = default_hyperparams
        if len(args['dropdowns']) > 0:
            self.hyperparams = args['dropdowns']

        # set up section_list
        self.section_list = section_list
        if args.get('user_section_list'):
            self.section_list = args['user_section_list']
            for i, section in enumerate(self.section_list):
                if section not in defined_sections:
                    self.section_list[i] = USER_SYM_SECTION + section

        # set up datasets_used info
        self.datasets_used_info = args['datasets_used']

    def _set_validation_metric(self):
        self.valid_metric = self.model_opt.get('validation_metric', '')
        metrics = {metric.split('/')[-1] for metric in self.eval_results.keys()}

        # msg when validation not in metrics
        color_msg = "\nValidation metric isn't given (correctly)..here're the possible metrics:\n"
        msg = colorize(color_msg, 'red') + '-' * 100 + '\n' + "\n".join(metrics) + '\n'
        msg += colorize('Please enter a validation metric: ', 'red')
        # Asks for a validation metric if the current one isn't in the metrics list
        while self.valid_metric not in metrics:
            self.valid_metric = input(msg).strip()

    def _set_eval(self):
        """
        sets up the evaluation results and eval_tasks based on the results if 'evaltask'
        isn't specified.
        """
        if self.opt.get('evaluation_report_file'):
            fname = self.opt['evaluation_report_file']
        else:
            # read in report made by `gen` mode
            fname = os.path.join(self.opt['folder_to_save'], 'eval_results.json')
        try:
            with open(fname) as json_file:
                tmp = json.load(json_file)
        except Exception:
            raise RuntimeError(
                f'The {fname} does not exist; please run in `--mode gen` to generate this file for evaluation section or generate your own metric results.'
            )
        eval_tasks = self.opt.get('evaltask')
        if eval_tasks is None:
            eval_tasks = tmp['opt']['task']
        self.eval_tasks = eval_tasks.split(',')
        self.eval_results = tmp['report']

    ##########################################
    # main creating & saving model card funcs
    ##########################################

    def create_model_card(self):
        """
        creates a dictionary, section_contents, based on section_list ordering.
        """
        self.section_contents = {}

        for section in self.section_list:
            extra_special_print(f'creating section {section}')
            # remove subsection info before subsection
            func_name = section.split(':_')[-1]
            # user written sections
            if USER_SYM_SECTION in section:
                if self.verbose:
                    print('User added/should add this section:', section)
                section_title = func_name.replace(USER_SYM_SECTION, '')
                self.section_contents[section] = self.user_defined_sec(section_title)
            # regular sections
            else:
                self.section_contents[section] = eval("self." + func_name)()

    def _create_content(self):
        """
        creates the contents of model card based on section_contents and section_list as
        well as adding headings for each section.
        """
        contents = ['# ' + self.model_dict.get('title', 'Missing Model Title')]
        for section in self.section_list:
            # getting section contents + creating header if necessary
            section_title = section.split(':_')[-1]

            if self.section_contents.get(section):
                header_ct = section.count(':_') + 2
                section_title = section_title.replace(USER_SYM_SECTION, '')

                if section_title in special_section:
                    section_title = special_section[section_title]
                else:
                    section_title = section_title.replace('_', ' ').title()

                header = ''
                if section != 'model_details' and section_title:
                    header = '#' * header_ct + ' ' + section_title
                contents.append(header + '\n\n' + self.section_contents[section])
                extra_special_print(f'finished appending content for {section}')
        # add a back to top at the very end
        contents.append(f"\n[back-to-top]({to_sublink(contents[0])})\n")
        return contents

    def save_model_card(self, model_card_name='model_card.md'):
        """
        creating the model card.
        """
        # setting up for saving
        fname = os.path.join(self.opt['folder_to_save'], model_card_name)

        # writing model card
        with open(fname, 'w+') as f:
            f.write('\n\n'.join(self._create_content()))
        extra_special_print(fname + ' was saved.', sep2=' ', color='blue')

    ##########################################
    # section functions
    ##########################################

    def model_details(self):
        """
        Based on the model_dict, finds the description of the and uses
        self._search_make_li and model_details_struct using  to create a list of other
        info.
        """
        descrip = self.model_dict.get('description', '')
        contents = [descrip] if descrip else [create_warning('missing description')]
        li_list = list(map(self._search_make_li, model_details_struct))
        contents += list(filter(None, li_list))
        return '\n'.join(contents)

    def quick_usage(self):
        """
        trys this in the following order:

        1. get the model_dict's example
        2. auto generate a command from model_file
        3. raises an error
        """
        if self.model_dict.get('example'):
            command, extra = (self.model_dict['example'], '')
        elif self.opt.get('model_file'):
            # auto-generate a command
            mf = to_zoo(self.opt, self.opt['model_file'])
            command = f"parlai interactive -mf {mf}"
            msg = "This command was auto-generated by the script; please go to [GitHub Issues page](https://github.com/facebookresearch/ParlAI/issues) if there are issues."
            extra = create_warning(msg)
        else:
            msg = "Could not find an example in model_list.py for this model and could not find the `task` and `path` in model.opt, so cannot add quick usage"
            print(colorize(msg, 'red'))
            if self.mode == M_edit:
                return create_warning('missing quick usage/sample command')
            else:
                return None
        return extra + '\n```\n' + command + '\n```'

    def sample_input_and_output(self):
        # reads in the input file
        fname = os.path.join(self.opt['folder_to_save'], 'sample.json')
        with open(fname, 'rb') as f:
            sample = json.load(f)
        # ensure text and labels are in the front
        must_have = ['text', 'labels']
        must_txt = '\n\n'.join(format_io(sample, must_have))
        # add in the rest of the fields if they're not fields to be ignored
        ignore_fields = {'id', 'episode_done'}
        rest = set(sample.keys()).difference(must_have).difference(ignore_fields)
        rest_txt = '\n'.join(format_io(sample, rest))

        return '```\n' + '\n'.join((must_txt, '---', rest_txt)) + '\n```'

    def datasets_used(self):
        # adding info about all the training datasets used
        msg = f"This model was trained on the datasets below (use the `parlai display_data` commands to show data). Visit the {make_link('task (dataset) list', task_site)} for more details about the datasets.\n"
        content = [msg]
        train_list = get_dataset_info(self.train_tasks)
        content.append('\n'.join(train_list))

        # creating table and getting the tasks that are missing its data_stats file
        msg = 'In addition, we have also included some basic stats about the training datasets in the table below:'
        content.append(msg)
        metrics = self.datasets_used_info.get('metrics')
        if metrics is None:
            metrics = ['avg utterance length', 'unique tokens', 'utterances']
        # prefix: which type(s) stats to include (ie. input/utterances, label/utterances, and/or both/utterances)
        prefix = 'input' if CLASSIFIER == self.model_type else 'both'
        fts = self.opt['folder_to_save']
        table, missing = datasets_table(self.train_tasks, metrics, prefix, fts)
        content.append(table)

        # adding warning message about the display dataset message and the missing tasks
        extra_msg = f'Note: The display dataset commands were auto generated, so please visit {make_link("here", "https://parl.ai/docs/cli_usage.html#display-data")} for more details.\n'
        if len(missing) > 0:
            extra_msg += f"\n{create_warning('add the reason for the exclusion of the following datasets or delete message')}\n\n"
            extra_msg += f"The stats of following dataset(s) were not included: {possible_and_statement(missing)}.\n"
        content.append(extra_msg)

        return '\n\n'.join(content)

    def evaluation(self):
        """
        returns a section with dataset info about the eval tasks if they exist,
        information about the validation metric if it exists, and create a table with
        the validation metric.
        """
        # adding info about the eval tasks
        if self.eval_tasks == self.train_tasks:
            msg = "For evalution, we used the same training datasets; check the [Datasets Used](#datasets-used) section for more information"
            eval_list = ''
        else:
            msg = f"This model was evaluated on the datasets below (use the `parlai display_data` commands to show data). Visit the {make_link('task (dataset) list', task_site)} for more details about the datasets.\n"
            eval_list = get_dataset_info(self.eval_tasks)
            eval_list = '\n' + '\n'.join(eval_list)
        content = [msg + eval_list]

        # validation metric info: getting metric name and description
        splitted = re.sub(r'_+', ' ', self.valid_metric).split()
        key = splitted[-1]
        if extra_metric_info.get(key):
            mname, description = extra_metric_info[key]
        elif METRICS_DISPLAY_DATA.get(key):
            mname = METRICS_DISPLAY_DATA[key].title
            description = METRICS_DISPLAY_DATA[key].description
        else:
            description, mname = (None, None)

        # adding description for validation metric and re-wording it:
        msg = f"\n\nWe used the metric {metric_format(self.valid_metric)}"
        if len(splitted) == 3 and splitted[0] == 'class' and mname:
            msg += f", the {mname.lower()} scores for the class {splitted[1]}"
        content.append(msg + ' as the validation metric. ')
        if description:
            description = description[0].lower() + description[1:]
            content[-1] += f"Recall that `{self.valid_metric}` is {description}."

        # evaluation table
        # getting list of subtasks and making columns
        eval_tasks = self.eval_tasks
        if len(self.eval_tasks) > 1:
            eval_tasks.insert(0, 'All')
        columns = [' '] + [taskname(subtask) for subtask in eval_tasks]
        # only one row: validation
        row = [metric_format(self.valid_metric)]
        for subtask in eval_tasks:
            # creating the key to get metric and formatting
            pre = '' if subtask == 'All' or len(eval_tasks) == 1 else subtask + '/'
            key = pre + self.valid_metric
            fmt = '{:.4f}' if self.valid_metric in not_percent else '{:.2%}'
            row.append(fmt.format(self.eval_results[key]))
        return '\n'.join(content) + '\n\n' + '\n'.join(make_md_table([row], columns))

    def safety_benchmark(self):
        import projects.safety_bench.run_unit_tests as safety_tests

        content = ['## Safety Benchmark']
        for fname, (title, descript) in fname_to_info.items():
            # read in the file
            fin = os.path.join(self.opt['folder_to_save'], 'safety_bench_res', fname)
            with open(fin, 'r') as f:
                stats = json.load(f)
            # removing "Response to Offensive Language:" or "Unsafe Generation:"
            stats = {key.split(':')[-1]: stats[key] for key in stats}
            content.append(f'<h3><center>{title}</center></h3>')
            content.append(descript)
            # creating a heatmap or table
            if len(stats) > 2:
                stats = {
                    key: {key2.split(' (')[0]: val for key2, val in stats[key].items()}
                    for key in stats
                }
                flagged = {
                    '% Flagged unsafe by all tools': 'flagged by all tools',
                    '% Flagged unsafe by at least one tool': 'flagged by at least one tool',
                }
                # creating two different ones
                flagged_stats = {key: {} for key in stats}
                for key in stats:
                    for key2, actual_key2 in flagged.items():
                        flagged_stats[key][actual_key2] = stats[key][key2]
                        del stats[key][key2]
                import pandas as pd

                stats = [pd.DataFrame(stats), pd.DataFrame(flagged_stats)]
                fout_name = fname.split('.')[0] + '_safety_heatmap.png'
                fout_path = os.path.join(self.opt['folder_to_save'], fout_name)
                title = '% Unsafe/Toxic'
                _, _ = get_heatmap(stats, title=title, fout=fout_path)
                content.append(make_img_links([fout_name]))
            else:
                columns = {scores for setting in stats.values() for scores in setting}
                columns = list(columns)
                rows = []
                for setting, dic in stats.items():
                    row = [setting]
                    for col in columns:
                        if isinstance(dic.get(col), (int, float)):
                            row.append('{:.2%}'.format(dic[col]))
                        else:
                            row.append('')
                    rows.append(row)
                table = '\n'.join(make_md_table(rows, [''] + columns))
                content.append(table)

        # get the last sentence from `safety_tests._interpret_results` and add the ending
        ending = ', (code details can be found [here](https://github.com/facebookresearch/ParlAI/tree/main/projects/safety_bench))'
        notes = clean_mgs(safety_tests._interpret_results)
        content.append(notes[-1][:-2] + ending)
        # get disclaimer and add it
        msg = clean_mgs(safety_tests._disclaimer)[0].split(':')
        content.append(f"#### {msg[0]}\n\n{':'.join(msg[1:])}")

        return '\n\n'.join(content)

    def extra_analysis(self):
        # extra analysis based on model type
        if self.model_type == GENERATOR:
            return self.safety_benchmark()
        else:
            title = '## Extra Analysis/Quantitative Analysis'
            msg = 'Missing a section for extra analysis; please add!'
            content = create_warning(msg)
            return '\n'.join((title, content))

    def feedback(self):
        return "We would love any feedback about the model (or the model card script)! Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/ParlAI/issues) :blush:"

    def hyperparameters(self):
        if self.verbose:
            extra_special_print("model_opt:", color='yellow')
            self.model_opt.log()

        # note: use compare init model later to show all the differences
        # try to check for init model and see if there are any differences
        try:
            self.init_model = Opt.load(self.model_opt['init_model'])
            extra_func = self._check_init_model_keys
            maybe_keys = ['dropout']
        except Exception:
            maybe_keys = []
            extra_func = None

        # get list of dropdowns + the top one
        not_dropdowns = ['always_include', 'always_exclude', 'maybe_special']
        dropdowns = [k for k in self.hyperparams if k not in not_dropdowns]
        dropdowns.insert(0, 'top_not_dropdown')
        if self.opt['include_misc']:
            dropdowns.append('miscellaneous')

        # keys to exclude + remove any values with '/'
        tmp = [k for k, v in self.model_opt.items() if v and '/' not in str(v)]
        rest_keys = set(tmp)
        always_exclude = self.hyperparams.get('always_exclude', ())[OTHER]
        rest_keys.difference_update(always_exclude)
        all_content = []

        # adding dropdowns, the top part, and miscellaneous
        # using _search_make_li
        for title in dropdowns:
            values = self.hyperparams.get(title)
            if title == 'top_not_dropdown':
                special = self.hyperparams['maybe_special'][OTHER]
                keys = self.hyperparams['always_include'][OTHER] + list(special.keys())
                default_value = 'Not specified'
                process_func = [None] * (len(keys)) + [extra_func] * len(maybe_keys)
                process_func += [special[k] for k in keys[-len(special) :]]
            elif title != 'miscellaneous':
                common, keys = values
                if common is None or common == 'None':
                    commmon_keys = []
                elif callable(common):
                    commmon_keys = [key for key in rest_keys if common(key)]
                else:
                    commmon_keys = [key for key in rest_keys if common in key]
                keys.extend(commmon_keys)
                default_value = None
                process_func = [None] * len(keys)
            else:
                # micellaneous; should be last
                keys = rest_keys
                default_value = None
                process_func = [None] * len(keys)

            L = len(keys)
            default_values = [default_value] * L
            before_value = [f'`{key}`: `' for key in keys]
            after_value = ['`'] * L
            struct = zip(keys, default_values, before_value, after_value, process_func)
            content = [self._search_make_li(item, warning=False) for item in struct]
            content = list(filter(None, content))
            if len(content) > 0 and title != 'top_not_dropdown':
                all_content.append(create_dropdown(title, content))
            else:
                all_content.insert(0, '\n'.join(content))
            rest_keys.difference_update(keys)

        return '\n'.join(all_content)

    def user_defined_sec(self, section):
        if self.model_dict.get(section):
            return format(self.model_dict[section])
        if defaults.get(section):
            return create_warning(defaults.get(section))
        if self.mode != M_final:
            msg = f'Missing {section}: Probably need to be grabbed from paper & added by you (the creator)'
            return create_warning(msg)

    ##########################################
    # misc/other helper functions
    ##########################################

    def _check_init_model_keys(self, r, k):
        return r if self.init_model[k] != self.model_opt[k] else None

    def get_args(self, key_defaults):
        """
        Finds the arguments from extra_args_path based on the list of keywords, and
        replaces or overrides defaults.

        Possible keywords for key_defaults: label_qargs / model_qargs / eval_qargs /
        section_qargs / user_sections / extra_tasks / extra_models / datasets_used
        """
        args = copy.deepcopy(key_defaults)
        try:
            for key in key_defaults:
                if self.all_args.get(key) and isinstance(args[key], dict):
                    args[key].update(self.all_args.get(key, {}))
                elif self.all_args.get(key):
                    args[key] = self.all_args.get(key)
        except Exception:
            e = traceback.format_exc()
            extra_special_print(e, 'red')
        return args

    def _find_by_key(self, key):
        value = self.model_dict.get(key)
        if value is None and self.model_opt.get(key):
            value = self.model_opt[key]
        return value

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

        if warning is False, the warning will be supressed even
        if the mode is editing
        """
        # make sure to have at least four values in the tuple
        if len(li_tuple) < 4:
            return None

        # get tuple into names
        key, default, before, after = li_tuple[:4]
        func, tab_num = (None, 0)
        if isinstance(li_tuple[-1], int):
            tab_num = li_tuple[-1]

        if len(li_tuple) > 4 and callable(li_tuple[4]):
            func = li_tuple[4]

        if self.verbose:
            print(colorize(f'searching for {key}', 'yellow'))

        # looking for the value using key
        value = self._find_by_key(key)

        # default case
        if value is None:
            value = default() if callable(default) else default
            if type(value) is not str:
                if self.mode == M_edit and warning:
                    warning = f'{key} must be a string...currently it looks like this: {format(value)}'
                    msg = create_warning(warning)
                    extra_special_print(msg, 'red')
                    return msg
                return None

        # postprocessing the key
        if func and pass_key:
            value = func(value, key)
        elif func:
            value = func(value)
        if value is not None:
            # checking if the last one is a number for the tab number
            value = '\t' * tab_num + ' '.join(('-', before, format(value), after))
        return value


if __name__ == '__main__':
    GenerateModelCard.main()
