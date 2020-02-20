#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
File for miscellaneous utility functions and constants.
"""

from collections import deque
import math
import random
import time
from typing import Union, Optional, Set, Any, Dict, List
import warnings
import json

from parlai.core.message import Message

try:
    import torch

    __TORCH_AVAILABLE = True
except ImportError:
    # silence the error, we'll have other problems later if it's super necessary
    __TORCH_AVAILABLE = False


DISPLAY_MESSAGE_DEFAULT_FIELDS = {
    'episode_done',
    'id',
    'image',
    'text',
    'labels',
    'eval_labels',
    'label_candidates',
    'text_candidates',
    'reward',
    'eval_labels_vec',
    'text_vec',
    'label_candidates_vecs',
    'token_losses',
}


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
    with open(path) as read:
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


class Predictor(object):
    """
    Wrapper to set up running version of model and request predictions.

    Note that this maintains no World state (does not use a World), merely
    providing the observation directly to the model and getting a response.

    This is limiting when it comes to certain use cases, but allows for quick
    model deployment.
    """

    def __init__(self, args=None, **kwargs):
        """
        Initialize the predictor, setting up opt automatically if needed.

        Args is expected to be in the same format as sys.argv: e.g. a list in
        the form ['--model', 'seq2seq', '-hs', 128, '-lr', 0.5].

        kwargs is interpreted by appending '--' to it and replacing underscores
        with hyphens, so 'dict_file=/tmp/dict.tsv' would be interpreted as
        '--dict-file /tmp/dict.tsv'.
        """
        from parlai.core.params import ParlaiParser
        from parlai.core.agents import create_agent

        if args is None:
            args = []
        for k, v in kwargs.items():
            args.append('--' + str(k).replace('_', '-'))
            args.append(str(v))
        parser = ParlaiParser(True, True)
        self.opt = parser.parse_args(args)
        self.agent = create_agent(self.opt)

    def predict(self, observation):
        """
        From a ParlAI-standard message dict, get model prediction.
        """
        if 'episode_done' not in observation:
            observation['episode_done'] = True
        self.agent.observe(observation)
        reply = self.agent.act()
        return reply


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
        log = {}
        log['exs'] = done
        if total > 0:
            log['%done'] = done / total
            if log["%done"] > 0:
                time_left = self.tot_time / log['%done'] - self.tot_time
                log['time_left'] = str(int(time_left)) + 's'
            z = '%.2f' % (100 * log['%done'])
            log['%done'] = str(z) + '%'

        if report:
            log = {**report, **log}

        int_time = int(self.tot_time)
        report_s = json.dumps(nice_report(log))
        text = f'{int_time}s elapsed: {report_s}'
        return text, log


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


def nice_report(report):
    from parlai.core.metrics import Metric

    output = {}
    for k, v in report.items():
        if isinstance(v, Metric):
            v = v.value()
        if isinstance(v, float):
            v = round_sigfigs(v, 4)
        output[k] = v
    return output


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
        return round(x_, -math.floor(math.log10(abs(x_)) - sigfigs + 1))
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


class PaddingUtils(object):
    """
    Helps with padding input and target tensors.

    DEPRECATED. USE PARLAI.CORE.TORCH_AGENT INSTEAD.
    """

    # DEPRECATIONDAY: delete!

    @classmethod
    def pad_text(
        cls,
        observations,
        dictionary,
        end_idx=None,
        null_idx=0,
        dq=False,
        eval_labels=True,
        truncate=None,
    ):
        """
        Pad observations to max width.

        We check that examples are valid, pad with zeros, and sort by length
        so that we can use the pack_padded function. The list valid_inds
        keeps track of which indices are valid and the order in which we sort
        the examples.

        dq -- whether we should use deque or list
        eval_labels -- whether or not we want to consider eval labels
        truncate -- truncate input and output lengths

        DEPRECATED. USE PARLAI.CORE.TORCH_AGENT INSTEAD.
        """

        def valid(obs):
            # check if this is an example our model should actually process
            return 'text' in obs and len(obs['text']) > 0

        try:
            # valid examples and their indices
            valid_inds, exs = zip(
                *[(i, ex) for i, ex in enumerate(observations) if valid(ex)]
            )
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None, None, None, None, None

        # `x` text is already tokenized and truncated
        # sort by length so we can use pack_padded
        if any(['text2vec' in ex for ex in exs]):
            parsed_x = [ex['text2vec'] for ex in exs]
        else:
            parsed_x = [dictionary.txt2vec(ex['text']) for ex in exs]

        if len(parsed_x) > 0 and not isinstance(parsed_x[0], deque):
            if dq:
                parsed_x = [deque(x, maxlen=truncate) for x in parsed_x]
            elif truncate is not None and truncate > 0:
                parsed_x = [x[-truncate:] for x in parsed_x]

        x_lens = [len(x) for x in parsed_x]
        ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

        exs = [exs[k] for k in ind_sorted]
        valid_inds = [valid_inds[k] for k in ind_sorted]
        parsed_x = [parsed_x[k] for k in ind_sorted]
        end_idxs = [x_lens[k] for k in ind_sorted]

        eval_labels_avail = any(['eval_labels' in ex for ex in exs])
        labels_avail = any(['labels' in ex for ex in exs])
        if eval_labels:
            some_labels_avail = eval_labels_avail or labels_avail
        else:
            some_labels_avail = labels_avail

        max_x_len = max(x_lens)

        # pad with zeros
        if dq:
            parsed_x = [
                x
                if len(x) == max_x_len
                else x + deque((null_idx,)) * (max_x_len - len(x))
                for x in parsed_x
            ]
        else:
            parsed_x = [
                x if len(x) == max_x_len else x + [null_idx] * (max_x_len - len(x))
                for x in parsed_x
            ]
        xs = parsed_x

        # set up the target tensors
        ys = None
        labels = None
        y_lens = None
        if some_labels_avail:
            # randomly select one of the labels to update on (if multiple)
            if labels_avail:
                labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            else:
                labels = [random.choice(ex.get('eval_labels', [''])) for ex in exs]
            # parse each label and append END
            if dq:
                parsed_y = [deque(maxlen=truncate) for _ in labels]
                for deq, y in zip(parsed_y, labels):
                    deq.extendleft(reversed(dictionary.txt2vec(y)))
            else:
                parsed_y = [dictionary.txt2vec(label) for label in labels]
            if end_idx is not None:
                for y in parsed_y:
                    y.append(end_idx)

            y_lens = [len(y) for y in parsed_y]
            max_y_len = max(y_lens)

            if dq:
                parsed_y = [
                    y
                    if len(y) == max_y_len
                    else y + deque((null_idx,)) * (max_y_len - len(y))
                    for y in parsed_y
                ]
            else:
                parsed_y = [
                    y if len(y) == max_y_len else y + [null_idx] * (max_y_len - len(y))
                    for y in parsed_y
                ]
            ys = parsed_y

        return xs, ys, labels, valid_inds, end_idxs, y_lens

    @classmethod
    def map_predictions(
        cls,
        predictions,
        valid_inds,
        batch_reply,
        observations,
        dictionary,
        end_idx,
        report_freq=0.1,
        labels=None,
        answers=None,
        ys=None,
    ):
        """
        Match predictions to original index in the batch.

        Predictions are mapped back to appropriate indices in the batch_reply
        using valid_inds.

        report_freq -- how often we report predictions

        DEPRECATED. USE PARLAI.CORE.TORCH_AGENT INSTEAD.
        """
        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a timelab
            curr = batch_reply[valid_inds[i]]
            output_tokens = []
            j = 0
            for c in predictions[i]:
                if c == end_idx and j != 0:
                    break
                else:
                    output_tokens.append(c)
                j += 1
            curr_pred = dictionary.vec2txt(output_tokens)
            curr['text'] = curr_pred

            if labels is not None and answers is not None and ys is not None:
                y = []
                for c in ys[i]:
                    if c == end_idx:
                        break
                    else:
                        y.append(c)
                answers[valid_inds[i]] = y
            elif answers is not None:
                answers[valid_inds[i]] = curr_pred

            if random.random() > (1 - report_freq):
                # log sometimes
                print('TEXT: ', observations[valid_inds[i]]['text'])
                print('PREDICTION: ', curr_pred, '\n~')
        return


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
        ellipsis = '...and {} more'.format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)


def display_messages(
    msgs: List[Dict[str, Any]],
    prettify: bool = False,
    ignore_fields: str = '',
    max_len: int = 1000,
) -> Optional[str]:
    """
    Return a string describing the set of messages provided.

    If prettify is true, candidates are displayed using prettytable. ignore_fields
    provides a list of fields in the msgs which should not be displayed.
    """

    def _token_losses_line(
        msg: Dict[str, Any], ignore_fields: List[str], space: str
    ) -> Optional[str]:
        """
        Displays the loss associated with each token. Can be used for debugging
        generative models.

        See TorchGeneratorAgent._construct_token_losses for an example implementation.
        """
        key = 'token_losses'
        token_losses = msg.get(key, None)
        if key in ignore_fields or not token_losses:
            return None
        # Reduce losses to 4 significant figures
        formatted_tl = ' | '.join(
            [f"{tl[0]} {float('{:.4g}'.format(tl[1]))}" for tl in token_losses]
        )
        return f'{space}[{key}]: {formatted_tl}'

    lines = []
    episode_done = False
    ignore_fields_ = ignore_fields.split(',')
    for index, msg in enumerate(msgs):
        if msg is None or (index == 1 and 'agent_reply' in ignore_fields_):
            # We only display the first agent (typically the teacher) if we
            # are ignoring the agent reply.
            continue
        if msg.get('episode_done'):
            episode_done = True
        # Possibly indent the text (for the second speaker, if two).
        space = ''
        if len(msgs) == 2 and index == 1:
            space = '   '
        # Only display rewards !=0 as they are confusing in non-RL tasks.
        if msg.get('reward', 0) != 0:
            lines.append(space + '[reward: {r}]'.format(r=msg['reward']))
        for key in msg:
            if key not in DISPLAY_MESSAGE_DEFAULT_FIELDS and key not in ignore_fields_:
                if type(msg[key]) is list:
                    line = '[' + key + ']:\n  ' + _ellipse(msg[key], sep='\n  ')
                else:
                    line = '[' + key + ']: ' + clip_text(str(msg.get(key)), max_len)
                lines.append(space + line)
        if type(msg.get('image')) in [str, torch.Tensor]:
            lines.append(f'[ image ]: {msg["image"]}')
        if msg.get('text', ''):
            text = clip_text(msg['text'], max_len)
            ID = '[' + msg['id'] + ']: ' if 'id' in msg else ''
            lines.append(space + ID + text)
        for field in {'labels', 'eval_labels', 'label_candidates', 'text_candidates'}:
            if msg.get(field) and field not in ignore_fields_:
                lines.append('{}[{}: {}]'.format(space, field, _ellipse(msg[field])))
        # Handling this separately since we need to clean up the raw output before displaying.
        token_loss_line = _token_losses_line(msg, ignore_fields_, space)
        if token_loss_line:
            lines.append(token_loss_line)

    if episode_done:
        lines.append('- - - - - - - - - - - - - - - - - - - - -')

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


_seen_warnings: Set[str] = set()


def warn_once(msg: str, warningtype=None) -> None:
    """
    Raise a warning, but only once.

    :param str msg: Message to display
    :param Warning warningtype: Type of warning, e.g. DeprecationWarning
    """
    global _seen_warnings
    if msg not in _seen_warnings:
        _seen_warnings.add(msg)
        warnings.warn(msg, warningtype, stacklevel=2)
