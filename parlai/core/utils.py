#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""File for miscellaneous utility functions and constants."""

from collections import deque
from functools import lru_cache
import math
import os
import random
import time
import warnings
import heapq

# some of the utility methods are helpful for Torch
try:
    import torch
    # default type in padded3d needs to be protected if torch
    # isn't installed.
    TORCH_LONG = torch.long
    __TORCH_AVAILABLE = True
except ImportError:
    TORCH_LONG = None
    __TORCH_AVAILABLE = False


"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


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
    'label_candidates_vecs'
}


def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def maintain_dialog_history(history, observation, reply='',
                            historyLength=1, useReplies='label_else_model',
                            dict=None, useStartEndIndices=True,
                            splitSentences=False):
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
        if useReplies == 'model' or (useReplies == 'label_else_model' and
                                     len(history['labels']) == 0):
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

    Every example will include these as candidates. The true labels for a
    specific example are also added to this set, so that it's possible to get
    the right answer.
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
                    line = line[space_idx + 1:]
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
        """From a ParlAI-standard message dict, get model prediction."""
        if 'episode_done' not in observation:
            observation['episode_done'] = True
        self.agent.observe(observation)
        reply = self.agent.act()
        return reply


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        """Initialize timer."""
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        """Reset timer to zero."""
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        """Resume timer."""
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        """Pause timer."""
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        """Get current timer time."""
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class TimeLogger():
    """Class for logging time progress against a goal."""

    def __init__(self):
        """Set up timer."""
        self.timer = Timer()
        self.tot_time = 0

    def total_time(self):
        """Return time elapsed at last log call."""
        return self.tot_time

    def time(self):
        """Return current timer time."""
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
            for k, v in report.items():
                if k not in log:
                    log[k] = v
        text = str(int(self.tot_time)) + "s elapsed: " + str(log).replace('\\n', '\n')
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
        """Initialize AttrDict using input dict."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def round_sigfigs(x, sigfigs=4):
    """
    Round value to specified significant figures.

    :param x: input number
    :param sigfigs: number of significant figures to return

    :returns: float number rounded to specified sigfigs
    """
    try:
        if x == 0:
            return 0
        return round(x, -math.floor(math.log10(abs(x)) - sigfigs + 1))
    except (RuntimeError, TypeError):
        # handle 1D torch tensors
        # if anything else breaks here please file an issue on Github
        if hasattr(x, 'item'):
            return round_sigfigs(x.item(), sigfigs)
        else:
            return round_sigfigs(x[0], sigfigs)
    except (ValueError, OverflowError) as ex:
        if x in [float('inf'), float('-inf')] or x != x:  # inf or nan
            return x
        else:
            raise ex


class NoLock(object):
    """Empty `lock`. Does nothing when you enter or exit."""

    def __enter__(self):
        """No-op."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """No-op."""
        pass


single_nolock = NoLock()


def no_lock():
    """Build a nolock for other classes to use for no-op locking."""
    return single_nolock


class PaddingUtils(object):
    """
    Helps with padding input and target tensors.

    DEPRECATED. USE PARLAI.CORE.TORCH_AGENT INSTEAD.
    """
    # DEPRECATIONDAY: delete!

    @classmethod
    def pad_text(cls, observations, dictionary,
                 end_idx=None, null_idx=0, dq=False, eval_labels=True,
                 truncate=None):
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
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(observations) if valid(ex)])
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
            parsed_x = [x if len(x) == max_x_len else
                        x + deque((null_idx,)) * (max_x_len - len(x))
                        for x in parsed_x]
        else:
            parsed_x = [x if len(x) == max_x_len else
                        x + [null_idx] * (max_x_len - len(x))
                        for x in parsed_x]
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
                parsed_y = [y if len(y) == max_y_len else
                            y + deque((null_idx,)) * (max_y_len - len(y))
                            for y in parsed_y]
            else:
                parsed_y = [y if len(y) == max_y_len else
                            y + [null_idx] * (max_y_len - len(y))
                            for y in parsed_y]
            ys = parsed_y

        return xs, ys, labels, valid_inds, end_idxs, y_lens

    @classmethod
    def map_predictions(cls, predictions, valid_inds, batch_reply,
                        observations, dictionary, end_idx, report_freq=0.1,
                        labels=None, answers=None, ys=None):
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


class OffensiveLanguageDetector(object):
    """
    Tries to detect offensive language in text.

    Detects offensive language using a list of offensive language and phrases
    from https://github.com/LDNOOBW.
    """

    def __init__(self):
        """Get data from external sources and build data representation."""
        import parlai.core.build_data as build_data
        from parlai.core.params import ParlaiParser
        from parlai.core.dict import DictionaryAgent
        self.tokenize = DictionaryAgent.split_tokenize

        parser = ParlaiParser(False, False)

        def _path():
            # Build the data if it doesn't exist.
            build()
            return os.path.join(
                self.datapath, 'OffensiveLanguage', 'OffensiveLanguage.txt'
            )

        def build():
            version = 'v1.0'
            dpath = os.path.join(self.datapath, 'OffensiveLanguage')
            if not build_data.built(dpath, version):
                print('[building data: ' + dpath + ']')
                if build_data.built(dpath):
                    # An older version exists, so remove these outdated files.
                    build_data.remove_dir(dpath)
                build_data.make_dir(dpath)

                # Download the data.
                fname = 'OffensiveLanguage.txt'
                url = 'http://parl.ai/downloads/offensive_language/' + fname
                build_data.download(url, dpath, fname)

                # Mark the data as built.
                build_data.mark_done(dpath, version)

        self.datapath = os.path.join(parser.parlai_home, 'data')
        self.datafile = _path()

        # store a token trie: e.g.
        # {'2': {'girls': {'1': {'cup': {'__END__': True}}}}
        self.END = '__END__'
        self.max_len = 1
        self.offensive_trie = {}
        self.word_prefixes = ['de', 'de-', 'dis', 'dis-', 'ex', 'ex-', 'mis',
                              'mis-', 'pre', 'pre-', 'non', 'non-', 'semi',
                              'semi-', 'sub', 'sub-', 'un', 'un-']
        self.word_suffixes = ['a', 'able', 'as', 'dom', 'ed', 'er', 'ers',
                              'ery', 'es', 'est', 'ful', 'fy', 'ies', 'ify',
                              'in', 'ing', 'ish', 'less', 'ly', 's', 'y']
        self.white_list = ['butter', 'buttery', 'spicy', 'spiced', 'spices',
                           'spicier', 'spicing', 'twinkies']

        with open(self.datafile, 'r') as f:
            for p in f.read().splitlines():
                mod_ps = [p]
                mod_ps += [pref + p for pref in self.word_prefixes]
                mod_ps += [p + suff for suff in self.word_suffixes]
                for mod_p in mod_ps:
                    if mod_p not in self.white_list:
                        self.add_phrase(mod_p)

    def add_phrase(self, phrase):
        """Add a single phrase to the filter."""
        toks = self.tokenize(phrase)
        curr = self.offensive_trie
        for t in toks:
            if t not in curr:
                curr[t] = {}
            curr = curr[t]
        curr[self.END] = True
        self.max_len = max(self.max_len, len(toks))

    def add_words(self, phrase_list):
        """Add list of custom phrases to the filter."""
        for phrase in phrase_list:
            self.add_phrase(phrase)

    def _check_sequence(self, toks, idx, node):
        """
        Check if words from the sequence are in the trie.

        This checks phrases made from
        toks[i], toks[i:i+2] ... toks[i:i + self.max_len]
        """
        right = min(idx + self.max_len, len(toks))
        for i in range(idx, right):
            if toks[i] in node:
                node = node[toks[i]]
                if self.END in node:
                    return ' '.join(toks[j] for j in range(idx, i + 1))
            else:
                break
        return False

    def contains_offensive_language(self, text):
        """Determine if text contains any offensive words in the filter."""
        if type(text) is str:
            toks = self.tokenize(text.lower())
        elif type(text) is list or type(text) is tuple:
            toks = text

        for i in range(len(toks)):
            res = self._check_sequence(toks, i, self.offensive_trie)
            if res:
                return res

        return None

    def __contains__(self, key):
        """Determine if text contains any offensive words in the filter."""
        return self.contains_offensive_language(key)

    def str_segment(self, text, dict_agent, k=1, max_length=None):
        """
        Function that segments a word without spaces into the most
        probable phrase with spaces

        :param string text: string to segment
        :param DictionaryAgent dict_agent: Dictionary we use
            to look at word frequencies
        :param int k: top k segmentations of string
        :param int max_length: max length of a substring
            (word) in the string. default (None) uses the
            length of the string.
        :returns: list of top k segmentations of the given string
        :rtype: list

        Example Usage:
            dict_agent = DictionaryAgent using Wiki Toxic Comments data
            old = OffensiveLanguageDector()

            split_str = old.str_segment('fucku2', dict_agent)
            split_str is 'fuck u 2'

            We can then run old.contains_offensive_language(split_str)
            which yields the offensive word 'fuck'

        """
        freqs = dict_agent.freqs()

        # Total number of word tokensd
        N = sum(freqs.values())

        # Number of distinct words in the Vocab
        V = len(freqs)

        logNV = math.log(N + V)
        max_heap = []

        if not max_length:
            max_length = len(text)

        @lru_cache(maxsize=16)
        def segment(text):
            # Return a list of words that is the best segmentation of text.
            if not text:
                return []
            candidates = [
                [first] + segment(rem)
                for first, rem in splits(text, max_length)
            ]

            nonlocal max_heap
            max_heap = []

            for c in candidates:
                cand_score = (score(c), c)  # tuple of (score, candidate)
                max_heap.append(cand_score)

            heapq._heapify_max(max_heap)

            return max_heap[0][1]

        def splits(text, max_length):
            # Returns a list of all possible first and remainder tuples where
            return [
                (text[:i+1], text[i+1:])
                for i in range(min(len(text), max_length))
            ]

        def score(words):
            # Returns probability for a sequence of words
            return sum(logprob(w) for w in words) / len(words)

        def logprob(word):
            # Utilizes laplace smoothing to get a probability of
            # unknown word
            count_w = freqs.get(word, 0)
            return math.log(count_w + 1) - logNV

        segment(text)
        res = []
        for _i in range(0, k):
            res.append(heapq._heappop_max(max_heap)[1])
        return res


def clip_text(text, max_len):
    """Clip text to max length, adding ellipses."""
    if len(text) > max_len:
        begin_text = ' '.join(
            text[:math.floor(0.8 * max_len)].split(' ')[:-1]
        )
        end_text = ' '.join(
            text[(len(text) - math.floor(0.2 * max_len)):].split(' ')[1:]
        )
        if len(end_text) > 0:
            text = begin_text + ' ...\n' + end_text
        else:
            text = begin_text + ' ...'
    return text


def _ellipse(lst, max_display=5, sep='|'):
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


def display_messages(msgs, prettify=False, ignore_fields='', max_len=1000):
    """
    Return a string describing the set of messages provided.

    If prettify is true, candidates are displayed using prettytable.
    ignore_fields provides a list of fields in the msgs which should not be
    displayed.
    """
    lines = []
    episode_done = False
    ignore_fields = ignore_fields.split(',')
    for index, msg in enumerate(msgs):
        if msg is None or (index == 1 and 'agent_reply' in ignore_fields):
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
            if key not in DISPLAY_MESSAGE_DEFAULT_FIELDS and key not in ignore_fields:
                if type(msg[key]) is list:
                    line = '[' + key + ']:\n  ' + _ellipse(msg[key], sep='\n  ')
                else:
                    line = '[' + key + ']: ' + clip_text(str(msg.get(key)), max_len)
                lines.append(space + line)
        if type(msg.get('image')) == str:
            lines.append(msg['image'])
        if msg.get('text', ''):
            text = clip_text(msg['text'], max_len)
            ID = '[' + msg['id'] + ']: ' if 'id' in msg else ''
            lines.append(space + ID + text)
        for field in {'labels', 'eval_labels', 'label_candidates', 'text_candidates'}:
            if msg.get(field) and field not in ignore_fields:
                lines.append('{}[{}: {}]'.format(space, field, _ellipse(msg[field])))

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
        elif (key == 'label_candidates' or key == 'labels' or
              key == 'eval_labels' or key == 'text_candidates'):
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
        value = t[ind + 1:]
        if key not in ignore_fields.split(','):
            msg[key] = convert(key, value)
    msg['episode_done'] = msg.get('episode_done', False)
    return msg


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

    default_fields = ['id', 'text', 'labels', 'label_candidates',
                      'episode_done', 'reward']
    txt = ""
    ignore_fields = ignore_fields.split(',')
    for f in default_fields:
        if f in msg and f not in ignore_fields:
            txt += add_field(f, msg[f])
    for f in msg.keys():
        if f not in default_fields and f not in ignore_fields:
            txt += add_field(f, msg[f])
    return txt.rstrip('\t')


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


def padded_tensor(items, pad_idx=0, use_cuda=False, left_padded=False,
                  max_len=None, fp16friendly=False):
    """
    Create a right-padded matrix from an uneven list of lists.

    Returns (padded, lengths), where padded is the padded matrix, and lengths
    is a list containing the lengths of each row.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param bool sort: If True, orders by the length
    :param int pad_idx: the value to use for padding
    :param bool use_cuda: if true, places `padded` on GPU
    :param bool left_padded:
    :param int max_len: if None, the max length is the maximum item length
    :param bool fp16friendly: if True, pads the time dimension to be a multiple of 8.

    :returns: (padded, lengths) tuple
    :rtype: (Tensor[int64], list[int])
    """
    # hard fail if we don't have torch
    if not __TORCH_AVAILABLE:
        raise ImportError(
            "Cannot use padded_tensor without torch; go to http://pytorch.org"
        )

    # number of items
    n = len(items)
    # length of each item
    lens = [len(item) for item in items]
    # max in time dimension
    t = max(lens) if max_len is None else max_len

    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)

    if fp16friendly and (t % 8 != 0):
        # pad to be a multiple of 8 to ensure we use the tensor cores
        t += 8 - (t % 8)

    if isinstance(items[0], torch.Tensor):
        # keep type of input tensors, they may already be cuda ones
        output = items[0].new(n, t)
    else:
        output = torch.LongTensor(n, t)
    output.fill_(pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            # skip empty items
            continue
        if not isinstance(item, torch.Tensor):
            # put non-tensors into a tensor
            item = torch.LongTensor(item)
        if left_padded:
            # place at end
            output[i, t - length:] = item
        else:
            # place at beginning
            output[i, :length] = item

    if use_cuda:
        output = output.cuda()
    return output, lens


def padded_3d(tensors, pad_idx=0, use_cuda=0, dtype=TORCH_LONG, fp16friendly=False):
    """
    Make 3D padded tensor for list of lists of 1D tensors or lists.

    :param tensors:
        list of lists of 1D tensors (or lists)
    :param pad_idx:
        padding to fill tensor with
    :param use_cuda:
        whether to call cuda() before returning
    :param bool fp16friendly:
        if True, pads the final dimension to be a multiple of 8.

    :returns:
        3D tensor with the maximum dimensions of the inputs
    """
    a = len(tensors)
    b = max(len(row) for row in tensors)
    c = max(len(item) for row in tensors for item in row)

    # pad empty tensors
    if fp16friendly and c % 8 != 0:
        c += 8 - (c % 8)
    c = max(c, 1)

    output = torch.full((a, b, c), pad_idx, dtype=dtype)

    for i, row in enumerate(tensors):
        for j, item in enumerate(row):
            if len(item) == 0:
                continue
            if not isinstance(item, torch.Tensor):
                item = torch.Tensor(item, dtype=dtype)
            output[i, j, :len(item)] = item

    if use_cuda:
        output = output.cuda()

    return output


def argsort(keys, *lists, descending=False):
    """
    Reorder each list in lists by the (descending) sorted order of keys.

    :param iter keys:
        Keys to order by.
    :param list[list] lists:
        Lists to reordered by keys's order.  Correctly handles lists and 1-D
        tensors.
    :param bool descending:
        Use descending order if true.

    :returns:
        The reordered items.
    """
    ind_sorted = sorted(range(len(keys)), key=lambda k: keys[k])
    if descending:
        ind_sorted = list(reversed(ind_sorted))
    output = []
    for lst in lists:
        # watch out in case we don't have torch installed
        if __TORCH_AVAILABLE and isinstance(lst, torch.Tensor):
            output.append(lst[ind_sorted])
        else:
            output.append([lst[i] for i in ind_sorted])
    return output


_seen_warnings = set()


def warn_once(msg, warningtype=None):
    """
    Raise a warning, but only once.

    :param str msg: Message to display
    :param Warning warningtype: Type of warning, e.g. DeprecationWarning
    """
    global _seen_warnings
    if msg not in _seen_warnings:
        _seen_warnings.add(msg)
        warnings.warn(msg, warningtype, stacklevel=2)


def fp16_optimizer_wrapper(
    optimizer,
    verbose=False,
    dynamic_loss_scale=True,
    loss_initial_scale=2.**17
):
    """
    Wraps the an optimizer with FP16 loss scaling protection.

    Requires apex to be installed. Will throw an ImportError if it is not.

    :param optimizer:
        Any torch optimizer
    :param bool verbose:
        Enables verbose output in the FP16 optimizer. Turning this on can help
        debug when FP16 is underperforming.
    :param bool dynamic_loss_scaling:
        FP16 requires loss scaling to avoid underflows. It is recommended this
        stays on, but advanced users may want it off.
    :param float loss_initial_scale:
        Initial loss scaling. Default chosen empirically, but models with very low
        or high loss values may need this adjusted. Stick with powers of 2.

    :returns:
        An APEX FP16 optimizer. Please note this has different requirements on
        how backward() and step() are called.
    """
    try:
        import apex.fp16_utils
    except ImportError:
        raise ImportError(
            'No fp16 support without apex. Please install it from '
            'https://github.com/NVIDIA/apex'
        )
    return apex.fp16_utils.FP16_Optimizer(
        optimizer,
        dynamic_loss_scale=dynamic_loss_scale,
        verbose=verbose,
        # TODO: We may later want to remove this flag. Right now it
        # empirically improves the first few backward passes, but future APEX
        # improvements may make this unnecessary.
        dynamic_loss_args={'init_scale': loss_initial_scale},
    )
