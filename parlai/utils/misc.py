#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
File for miscellaneous utility functions and constants.
"""

from collections import deque, OrderedDict
from typing import Union, Optional, Set, Any, Dict, List, Tuple
from datetime import timedelta
import functools
import math
import time
import re
import shutil
import json
import os

from parlai.core.message import Message
from parlai.utils.strings import colorize
from parlai.utils.io import PathManager
import parlai.utils.logging as logging

try:
    import torch

    __TORCH_AVAILABLE = True
except ImportError:
    # silence the error, we'll have other problems later if it's super necessary
    __TORCH_AVAILABLE = False


SPECIAL_FORMATED_DISPLAY_MESSAGE_FIELDS = {
    'episode_done',
    'id',
    'image',
    'text',
    'labels',
    'eval_labels',
    'label_candidates',
    'text_candidates',
    'reward',
    'token_losses',
}

MUST_SHOW_MESSAGE_FIELDS = {'image', 'text', 'labels', 'eval_labels', 'reward'}


def maintain_dialog_history(
    history,
    observation,
    reply='',
    historyLength=1,
    useReplies='label_else_model',
    dict=None,
    useStartEndIndices=True,
    splitSentences=False,
):
    """
    Keep track of dialog history, up to a truncation length.

    Either includes replies from the labels, model, or not all using param
    'replies'.

    DEPRECATED. USE PARLAI.CORE.TORCH_AGENT INSTEAD.
    """

    def parse(txt, splitSentences):
        if dict is not None:
            if splitSentences:
                vec = [dict.txt2vec(t) for t in txt.split('\n')]
            else:
                vec = dict.txt2vec(txt)
            return vec
        else:
            return [txt]

    if 'dialog' not in history:
        history['dialog'] = deque(maxlen=historyLength)
        history['episode_done'] = False
        history['labels'] = []

    if history['episode_done']:
        history['dialog'].clear()
        history['labels'] = []
        useReplies = 'none'
        history['episode_done'] = False

    if useReplies != 'none':
        if useReplies == 'model' or (
            useReplies == 'label_else_model' and len(history['labels']) == 0
        ):
            if reply:
                if useStartEndIndices:
                    reply = dict.start_token + ' ' + reply
                history['dialog'].extend(parse(reply, splitSentences))
        elif len(history['labels']) > 0:
            r = history['labels'][0]
            history['dialog'].extend(parse(r, splitSentences))

    obs = observation
    if 'text' in obs:
        if useStartEndIndices:
            obs['text'] = dict.end_token + ' ' + obs['text']
        history['dialog'].extend(parse(obs['text'], splitSentences))

    history['episode_done'] = obs['episode_done']

    labels = obs.get('labels', obs.get('eval_labels', None))
    if labels is not None:
        if useStartEndIndices:
            history['labels'] = [dict.start_token + ' ' + l for l in labels]
        else:
            history['labels'] = labels

    return history['dialog']


def load_cands(path, lines_have_ids=False, cands_are_replies=False):
    """
    Load global fixed set of candidate labels that the teacher provides.

    Every example will include these as candidates. The true labels for a specific
    example are also added to this set, so that it's possible to get the right answer.
    """
    if path is None:
        return None
    cands = []
    cnt = 0
    with PathManager.open(path) as read:
        for line in read:
            line = line.strip().replace('\\n', '\n')
            if len(line) > 0:
                cnt = cnt + 1
                # If lines are numbered we strip them of numbers.
                if cnt == 1 and line[0:2] == '1 ':
                    lines_have_ids = True
                # If tabs then the label_candidates are all the replies.
                if '\t' in line and not cands_are_replies:
                    cands_are_replies = True
                    cands = []
                if lines_have_ids:
                    space_idx = line.find(' ')
                    line = line[space_idx + 1 :]
                    if cands_are_replies:
                        sp = line.split('\t')
                        if len(sp) > 1 and sp[1] != '':
                            cands.append(sp[1])
                    else:
                        cands.append(line)
                else:
                    cands.append(line)
    return cands


class Timer(object):
    """
    Computes elapsed time.
    """

    def __init__(self):
        """
        Initialize timer.
        """
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        """
        Reset timer to zero.
        """
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        """
        Resume timer.
        """
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        """
        Pause timer.
        """
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        """
        Get current timer time.
        """
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class TimeLogger:
    """
    Class for logging time progress against a goal.
    """

    def __init__(self):
        """
        Set up timer.
        """
        self.timer = Timer()
        self.tot_time = 0

    def total_time(self):
        """
        Return time elapsed at last log call.
        """
        return self.tot_time

    def time(self):
        """
        Return current timer time.
        """
        return self.timer.time()

    def log(self, done, total, report=None):
        """
        Log report, time elapsed, and percentage progress towards goal.

        :param done: number of examples completed so far
        :param total: total number of elements to be completed. if total > 0,
                      calculates the time remaining and percentage complete.
        :param report: dict of pairs to log

        :returns: tuple log string, log dict
            log string contains time elapsed and string representation of
            the log dict
            log dict contains pairs of all items to log, which includes
            percentage complete and projected time left if total > 0
        """
        from parlai.core.metrics import Metric  # delay import to prevent circular dep

        if isinstance(done, Metric):
            done = done.value()
        self.tot_time += self.timer.time()
        self.timer.reset()
        if report:
            report['exs'] = done
        if total > 0 and done > 0:
            progress = done / total
            seconds_left = max(0, self.tot_time / progress - self.tot_time)
            eta = timedelta(seconds=int(seconds_left + 0.5))
        else:
            progress = 0
            eta = "unknown"
        elapsed = timedelta(seconds=int(self.tot_time))

        text = (
            f'{progress:.1%} complete ({done:,d} / {total:,d}), '
            f'{elapsed} elapsed, {eta} eta'
        )
        if report:
            report_s = nice_report(report)
            text = f'{text}\n{report_s}'
        return text, report


class AttrDict(dict):
    """
    Helper class to have a dict-like object with dot access.

    For example, instead of `d = {'key': 'value'}` use
    `d = AttrDict(key='value')`.
    To access keys, instead of doing `d['key']` use `d.key`.

    While this has some limitations on the possible keys (for example, do not
    set the key `items` or you will lose access to the `items()` method), this
    can make some code more clear.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize AttrDict using input dict.
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class NoLock(object):
    """
    Empty `lock`.

    Does nothing when you enter or exit.
    """

    def __enter__(self):
        """
        No-op.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        No-op.
        """
        pass


class SimpleCounter:
    """
    Simple counter object.
    """

    def __init__(self, value=0):
        self.val = value

    def increment(self, value=1):
        self.val += value

    def value(self):
        return self.val


def _report_sort_key(report_key: str) -> Tuple[str, str]:
    """
    Sorting name for reports.

    Sorts by main metric alphabetically, then by task.
    """
    # if metric is on its own, like "f1", we will return ('', 'f1')
    # if metric is from multitask, we denote it.
    # e.g. "convai2/f1" -> ('convai2', 'f1')
    # we handle multiple cases of / because sometimes teacher IDs have
    # filenames.
    fields = report_key.split("/")
    main_key = fields.pop(-1)
    sub_key = '/'.join(fields)
    return (sub_key or 'all', main_key)


def float_formatter(f: Union[float, int]) -> str:
    """
    Format a float as a pretty string.
    """
    if f != f:
        # instead of returning nan, return "" so it shows blank in table
        return ""
    if isinstance(f, int):
        # don't do any rounding of integers, leave them alone
        return str(f)
    if f >= 1000:
        # numbers > 1000 just round to the nearest integer
        s = f'{f:.0f}'
    else:
        # otherwise show 4 significant figures, regardless of decimal spot
        s = f'{f:.4g}'
    # replace leading 0's with blanks for easier reading
    # example:  -0.32 to -.32
    s = s.replace('-0.', '-.')
    if s.startswith('0.'):
        s = s[1:]
    # Add the trailing 0's to always show 4 digits
    # example: .32 to .3200
    if s[0] == '.' and len(s) < 5:
        s += '0' * (5 - len(s))
    return s


def _line_width():
    if os.environ.get('PARLAI_FORCE_WIDTH'):
        try:
            return int(os.environ['PARLAI_FORCE_WIDTH'])
        except ValueError:
            pass
    try:
        # if we're in an interactive ipython notebook, hardcode a longer width
        __IPYTHON__
        return 128
    except NameError:
        return shutil.get_terminal_size((88, 24)).columns


def nice_report(report) -> str:
    """
    Render an agent Report as a beautiful string.

    If pandas is installed,  we will use it to render as a table. Multitask
    metrics will be shown per row, e.g.

    .. code-block:
                 f1   ppl
       all     .410  27.0
       task1   .400  32.0
       task2   .420  22.0

    If pandas is not available, we will use a dict with like-metrics placed
    next to each other.
    """
    if not report:
        return ""

    from parlai.core.metrics import Metric

    try:
        import pandas as pd

        use_pandas = True
    except ImportError:
        use_pandas = False

    sorted_keys = sorted(report.keys(), key=_report_sort_key)
    output: OrderedDict[Union[str, Tuple[str, str]], float] = OrderedDict()
    for k in sorted_keys:
        v = report[k]
        if isinstance(v, Metric):
            v = v.value()
        if use_pandas:
            output[_report_sort_key(k)] = v
        else:
            output[k] = v

    if use_pandas:
        line_width = _line_width()

        df = pd.DataFrame([output])
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.stack().transpose().droplevel(0, axis=1)
        result = "   " + df.to_string(
            na_rep="",
            line_width=line_width - 3,  # -3 for the extra spaces we add
            float_format=float_formatter,
            index=df.shape[0] > 1,
        ).replace("\n\n", "\n").replace("\n", "\n   ")
        result = re.sub(r"\s+$", "", result)
        return result
    else:
        return json.dumps(
            {
                k: round_sigfigs(v, 4) if isinstance(v, float) else v
                for k, v in output.items()
            }
        )


def round_sigfigs(x: Union[float, 'torch.Tensor'], sigfigs=4) -> float:
    """
    Round value to specified significant figures.

    :param x: input number
    :param sigfigs: number of significant figures to return

    :returns: float number rounded to specified sigfigs
    """
    x_: float
    if __TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x_ = x.item()
    else:
        x_ = x  # type: ignore

    try:
        if x_ == 0:
            return 0
        return round(x_, -(math.floor(math.log10(abs(x_)) - sigfigs + 1)))
    except (ValueError, OverflowError) as ex:
        if x_ in [float('inf'), float('-inf')] or x_ != x_:  # inf or nan
            return x_
        else:
            raise ex


single_nolock = NoLock()


def no_lock():
    """
    Build a nolock for other classes to use for no-op locking.
    """
    return single_nolock


def clip_text(text, max_len):
    """
    Clip text to max length, adding ellipses.
    """
    if len(text) > max_len:
        begin_text = ' '.join(text[: math.floor(0.8 * max_len)].split(' ')[:-1])
        end_text = ' '.join(
            text[(len(text) - math.floor(0.2 * max_len)) :].split(' ')[1:]
        )
        if len(end_text) > 0:
            text = begin_text + ' ...\n' + end_text
        else:
            text = begin_text + ' ...'
    return text


def _ellipse(lst: List[str], max_display: int = 5, sep: str = '|') -> str:
    """
    Like join, but possibly inserts an ellipsis.

    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '... ({} of {} shown)'.format(max_display, len(choices))
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)


def display_messages(
    msgs: List[Dict[str, Any]],
    prettify: bool = False,
    ignore_agent_reply: bool = False,
    add_fields: str = '',
    max_len: int = 1000,
    verbose: bool = False,
) -> Optional[str]:
    """
    Return a string describing the set of messages provided.

    If prettify is true, candidates are displayed using prettytable. add_fields provides
    a list of fields in the msgs which should be displayed if verbose is off.
    """

    def _token_losses_line(
        msg: Dict[str, Any], fields_to_show: List[str], space: str
    ) -> Optional[str]:
        """
        Displays the loss associated with each token. Can be used for debugging
        generative models.

        See TorchGeneratorAgent._construct_token_losses for an example implementation.
        """
        key = 'token_losses'
        token_losses = msg.get(key, None)
        if key not in fields_to_show or not token_losses:
            return None
        # Reduce losses to 4 significant figures
        formatted_tl = ' | '.join(
            [f"{tl[0]} {float('{:.4g}'.format(tl[1]))}" for tl in token_losses]
        )
        return _pretty_lines(space, key, formatted_tl, 'text2')

    def _pretty_lines(indent_space, field, value, style):
        line = '{}{} {}'.format(
            indent_space, colorize('[' + field + ']:', 'field'), colorize(value, style)
        )
        return line

    lines = []
    episode_done = False
    extra_add_fields_ = add_fields.split(',')
    for index, msg in enumerate(msgs):
        if msg is None or (index == 1 and ignore_agent_reply):
            # We only display the first agent (typically the teacher) if we
            # are ignoring the agent reply.
            continue

        if msg.get('episode_done'):
            episode_done = True
        # Possibly indent the text (for the second speaker, if two).
        space = ''
        if len(msgs) == 2 and index == 1:
            space = '   '

        agent_id = msg.get('id', '[no id field]')
        if verbose:
            line = _pretty_lines(
                indent_space=space, field='id', value=agent_id, style='id'
            )
            lines.append(line)

        # Only display rewards !=0 as they are confusing in non-RL tasks.
        if msg.get('reward', 0) != 0:
            lines.append(space + '[reward: {r}]'.format(r=msg['reward']))

        fields_to_show = []
        if verbose:
            fields_to_show = [field for field in msg]
        else:
            fields_to_show = [
                field
                for field in msg
                if field in list(MUST_SHOW_MESSAGE_FIELDS) + extra_add_fields_
            ]
        fields_to_show.sort()

        # Display fields without special format
        for field in fields_to_show:
            if field not in SPECIAL_FORMATED_DISPLAY_MESSAGE_FIELDS:
                if type(msg[field]) is list:
                    value = _ellipse(msg[field], sep='\n  ')
                else:
                    value = clip_text(str(msg.get(field)), max_len)
                line = _pretty_lines(
                    indent_space=space, field=field, value=value, style='text2'
                )
                lines.append(line)

        # Display fields WITH special format requirements
        # Display Image
        if type(msg.get('image')) in [str, torch.Tensor]:
            lines.append(f'[ image ]: {msg["image"]}')
        # Display Text
        if msg.get('text', ''):
            value = clip_text(msg['text'], max_len)
            style = 'bold_text' if index == 0 else 'labels'
            field = 'text' if verbose else agent_id
            line = _pretty_lines(
                indent_space=space, field=field, value=value, style=style
            )
            lines.append(line)
        # Display Label Fields
        for field in {'labels', 'eval_labels', 'label_candidates', 'text_candidates'}:
            if msg.get(field) and field in fields_to_show:
                line = _pretty_lines(
                    indent_space=space,
                    field=field,
                    value=_ellipse(msg[field]),
                    style=field,
                )
                lines.append(line)
        # Handling this separately since we need to clean up the raw output before displaying.
        token_loss_line = _token_losses_line(msg, fields_to_show, space)
        if token_loss_line:
            lines.append(token_loss_line)

    if episode_done:
        lines.append(
            colorize('- - - - - - - END OF EPISODE - - - - - - - - - -', 'highlight')
        )

    return '\n'.join(lines)


def str_to_msg(txt, ignore_fields=''):
    """
    Convert formatted string to ParlAI message dict.

    :param txt:
        formatted string to convert. String format is tab-separated fields,
        with colon separating field name and contents.
    :param ignore_fields:
        (default '') comma-separated field names to not
        include in the msg dict even if they're in the string.
    """

    def tostr(txt):
        txt = str(txt)
        txt = txt.replace('\\t', '\t')
        txt = txt.replace('\\n', '\n')
        txt = txt.replace('__PIPE__', '|')
        return txt

    def tolist(txt):
        vals = txt.split('|')
        for v in vals:
            v = tostr(v)
        return vals

    def convert(key, value):
        if key == 'text' or key == 'id':
            return tostr(value)
        elif (
            key == 'label_candidates'
            or key == 'labels'
            or key == 'eval_labels'
            or key == 'text_candidates'
        ):
            return tolist(value)
        elif key == 'episode_done':
            return bool(value)
        else:
            return tostr(value)

    if txt == '' or txt is None:
        return None

    msg = {}
    for t in txt.split('\t'):
        ind = t.find(':')
        key = t[:ind]
        value = t[ind + 1 :]
        if key not in ignore_fields.split(','):
            msg[key] = convert(key, value)
    msg['episode_done'] = msg.get('episode_done', False)
    return Message(msg)


def msg_to_str(msg, ignore_fields=''):
    """
    Convert ParlAI message dict to string.

    :param msg:
        dict to convert into a string.
    :param ignore_fields:
        (default '') comma-separated field names to not include in the string
        even if they're in the msg dict.
    """

    def filter(txt):
        txt = str(txt)
        txt = txt.replace('\t', '\\t')
        txt = txt.replace('\n', '\\n')
        txt = txt.replace('|', '__PIPE__')
        return txt

    def add_field(name, data):
        if name == 'reward' and data == 0:
            return ''
        if name == 'episode_done' and data is False:
            return ''
        txt = ''
        if type(data) == tuple or type(data) == set or type(data) == list:
            # list entries
            for c in data:
                txt += filter(c) + "|"
            txt = txt[:-1]
        else:
            # single fields
            txt = filter(data)
        return name + ":" + txt + '\t'

    default_fields = [
        'id',
        'text',
        'labels',
        'label_candidates',
        'episode_done',
        'reward',
    ]
    txt = ""
    ignore_fields = ignore_fields.split(',')
    for f in default_fields:
        if f in msg and f not in ignore_fields:
            txt += add_field(f, msg[f])
    for f in msg.keys():
        if f not in default_fields and f not in ignore_fields:
            txt += add_field(f, msg[f])
    return txt.rstrip('\t')


# DEPRECATION DAY: DELETE
def set_namedtuple_defaults(namedtuple, default=None):
    """
    Set *all* of the fields for a given nametuple to a singular value.

    Additionally removes the default docstring for each field.
    Modifies the tuple in place, but returns it anyway.

    More info:
    https://stackoverflow.com/a/18348004

    :param namedtuple: A constructed collections.namedtuple
    :param default: The default value to set.

    :returns: the modified namedtuple
    """
    namedtuple.__new__.__defaults__ = (default,) * len(namedtuple._fields)
    for f in namedtuple._fields:
        del getattr(namedtuple, f).__doc__
    return namedtuple


_seen_logs: Set[str] = set()


def warn_once(msg: str) -> None:
    """
    Log a warning, but only once.

    :param str msg: Message to display
    """
    global _seen_logs
    if msg not in _seen_logs:
        _seen_logs.add(msg)
        logging.warn(msg)


def error_once(msg: str) -> None:
    """
    Log an error, but only once.

    :param str msg: Message to display
    """
    global _seen_logs
    if msg not in _seen_logs:
        _seen_logs.add(msg)
        logging.error(msg)


def recursive_getattr(obj, attr, *args):
    """
    Recursive call to getattr for nested attributes.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
