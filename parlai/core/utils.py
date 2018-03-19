# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from collections import deque
import math
import os
import random
import time


class Predictor(object):
    """Provides functionality for setting up a running version of a model and
    requesting predictions from that model on live data.

    Note that this maintains no World state (does not use a World), merely
    providing the observation directly to the model and getting a response.

    This is limiting when it comes to certain use cases, but is
    """

    def __init__(self, args=None, **kwargs):
        """Initializes the predictor, setting up opt automatically if necessary.

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
        parser = ParlaiParser(True, True, model_argv=args)
        self.opt = parser.parse_args(args)
        self.agent = create_agent(self.opt)

    def predict(self, observation):
        """From a ParlAI-standard observation dict, returns a prediction from
        the model.
        """
        if 'episode_done' not in observation:
            observation['episode_done'] = True
        self.agent.observe(observation)
        reply = self.agent.act()
        return reply


class Timer(object):
    """Computes elapsed time."""
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class AttrDict(dict):
    """Helper class to have a dict-like object with dot access.

    For example, instead of `d = {'key': 'value'}` use
    `d = AttrDict(key='value')`.
    To access keys, instead of doing `d['key']` use `d.key`.

    While this has some limitations on the possible keys (for example, do not
    set the key `items` or you will lose access to the `items()` method), this
    can make some code more clear.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def round_sigfigs(x, sigfigs=4):
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


def flatten(teacher, context_length=-1, include_labels=True):
    """Return a flattened version of a teacher's data where all episodes only
    have length one but contain the desired amount of context.

    If context_length is not -1, will use only that many past utterances.
    Default is -1. Setting it to one only uses the input text.

    If include_labels is True, will include a random label in past utterances.
    Default is True.
    """
    data = []
    current = []
    episode_done = False
    context_length = context_length if context_length >= 0 else None
    context = deque(maxlen=context_length)
    try:
        while not teacher.epoch_done():
            # collect examples in episode
            while not episode_done:
                action = teacher.act()
                current.append(action)
                episode_done = action['episode_done']

            # build separate episodes from each example
            for ex in current:
                context.append(ex.get('text', ''))
                if len(context) > 1:
                    ex['text'] = '\n'.join(context)
                ex['episode_done'] = True
                if include_labels:
                    # add labels to context
                    labels = ex.get('labels', ex.get('eval_labels'))
                    if labels is not None:
                        context.append(random.choice(labels))
                data.append(ex)
            # reset flags and content
            episode_done = False
            current.clear()
            context.clear()
        return data
    except MemoryError as ex:
        raise MemoryError('Ran out of memory building flattened data batches. '
                          'Try using --context-length set to a small value to '
                          'limit the length of each flattened example, '
                          'disabling batch sorting / flattening by setting '
                          '--batch-sort false, or switching to data streaming '
                          'using --datatype {type}:stream to read from disk '
                          'if it is supported for your dataset.')


def sort_data(data, key='text_label', method='spaces'):
    """Given a list of data, sort it according to the method and key.

    Currently the only supported method is counting the number of spaces.
    This appeared to be reliable enough and much faster than tokenizing.
    It performs much better than just using the length of the string.

    Currently the only supported key is sorting by first the text, then the
    label.
    See https://arxiv.org/abs/1706.05765 for an evaulation of alternative
    approaches for machine translation.
    Sorting by the source (text) gives a good improvement in speed over random
    batching and is robust to different types of optimization.
    Breaking ties by sorting by label length gives a further improvement in
    speed but can reduce robustness with some optimization schemes.
    """
    # TODO: support different keys and different methods
    tpls = []
    for ex in data:
        # first sort by input length
        fst = ex.get('text', '').count(' ')

        # then sort by target length (don't sort by eval_labels, no need)
        snd = 0
        labels = ex.get('labels', None)
        if labels is not None:
            # use average label length (probably just one answer usually)
            snd = sum(l.count(' ') for l in labels) / len(labels)

        tiebreaker = random.random()
        tpls.append((fst, snd, tiebreaker, ex))
    tpls.sort()
    return [e[-1] for e in tpls]


def make_batches(data, bsz):
    """Return a list of lists of size bsz given a list of examples."""
    return [data[i:i + bsz] for i in range(0, len(data), bsz)]


def maintain_dialog_history(history, observation, reply='',
                            historyLength=1, useReplies="labels",
                            dict=None, useStartEndIndices=True,
                            splitSentences=False):
    """Keeps track of dialog history, up to a truncation length.
    Either includes replies from the labels, model, or not all using param 'replies'."""

    def parse(txt, splitSentences=False):
        if dict is not None:
            if splitSentences:
                vec = [dict.txt2vec(t) for t in txt.split('\n')]
            else:
                vec =  dict.txt2vec(txt)
            if useStartEndIndices:
                parsed_x = deque([dict[dict.start_token]])
                parsed_x.extend(vec)
                parsed_x.append(dict[dict.end_token])
                return parsed_x
            else:
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
        if useReplies == 'model':
            if reply != '':
                history['dialog'].extend(parse(reply))
        elif len(history['labels']) > 0:
            r = history['labels'][0]
            history['dialog'].extend(parse(r, splitSentences))
    if 'text' in observation:
        history['dialog'].extend(parse(observation['text'], splitSentences))

    history['episode_done'] = observation['episode_done']
    if 'labels' in observation:
        history['labels'] = observation['labels']
    elif 'eval_labels' in observation:
        history['labels'] = observation['eval_labels']
    return history['dialog']


class NoLock(object):
    """Empty `lock`. Does nothing when you enter or exit."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


single_nolock = NoLock()
def no_lock():
    """Builds a nolock for other classes to use for no-op locking."""
    return single_nolock


class ProgressLogger(object):
    """
    Throttles and display progress in human readable form.
    Default throttle speed is 1 sec
    """
    def __init__(self, throttle=1, should_humanize=True):
        self.latest = time.time()
        self.throttle_speed = throttle
        self.should_humanize = should_humanize

    def humanize(self, num, suffix='B'):
        if num < 0:
            return num
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    def log(self, curr, total, width=40, force=False):
        """Displays a bar showing the current progress."""
        if curr == 0 and total == -1:
            print('[ no data received for this file ]', end='\r')
            return
        curr_time = time.time()
        if not force and curr_time - self.latest < self.throttle_speed:
            return
        else:
            self.latest = curr_time

        self.latest = curr_time
        done = min(curr * width // total, width)
        remain = width - done

        if self.should_humanize:
            curr = self.humanize(curr)
            total = self.humanize(total)

        progress = '[{}{}] {} / {}'.format(
            ''.join(['|'] * done),
            ''.join(['.'] * remain),
            curr,
            total
        )
        print(progress, end='\r')


class PaddingUtils(object):
    """
    Class that contains functions that help with padding input and target tensors.
    """
    @classmethod
    def pad_text(cls, observations, dictionary, end_idx=None, null_idx=0, dq=False, eval_labels=True, truncate=None):
        """We check that examples are valid, pad with zeros, and sort by length
           so that we can use the pack_padded function. The list valid_inds
           keeps track of which indices are valid and the order in which we sort
           the examples.
           dq -- whether we should use deque or list
           eval_labels -- whether or not we want to consider eval labels
           truncate -- truncate input and output lengths
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

        # set up the input tensors
        bsz = len(exs)

        # `x` text is already tokenized and truncated
        # sort by length so we can use pack_padded
        if any(['text2vec' in ex for ex in exs]):
            parsed_x = [ex['text2vec'] for ex in exs]
        else:
            parsed_x = [dictionary.txt2vec(ex['text']) for ex in exs]
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
                for dq, y in zip(parsed_y, labels):
                    dq.extendleft(reversed(dictionary.txt2vec(y)))
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
    def map_predictions(cls, predictions, valid_inds, batch_reply, observations, dictionary, end_idx, report_freq=0.1, labels=None, answers=None, ys=None):
        """Predictions are mapped back to appropriate indices in the batch_reply
           using valid_inds.
           report_freq -- how often we report predictions
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
                answers[valid_inds[i]] = output_tokens

            if random.random() > (1 - report_freq):
                # log sometimes
                print('TEXT: ', observations[valid_inds[i]]['text'].replace('__END__', ''))
                print('PREDICTION: ', curr_pred, '\n~')
        return


class OffensiveLanguageDetector(object):
    """Detects offensive language using a list of offensive language and phrases
    from https://github.com/LDNOOBW.
    """
    def __init__(self):
        import parlai.core.build_data as build_data
        from parlai.core.params import ParlaiParser
        from parlai.core.dict import DictionaryAgent
        self.tokenize = DictionaryAgent.split_tokenize

        parser = ParlaiParser(False, False)

        def _path():
            # Build the data if it doesn't exist.
            build()
            return os.path.join(self.datapath, 'OffensiveLanguage', 'OffensiveLanguage.txt')

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
                url = 'https://s3.amazonaws.com/fair-data/parlai/offensive_language/' + fname
                build_data.download(url, dpath, fname)

                # Mark the data as built.
                build_data.mark_done(dpath, version)

        self.datapath = os.path.join(parser.parlai_home, 'data')
        self.datafile = _path()

        # store a token trie: e.g.
        # {'2': {'girls': {'1': {'cup': {'__END__': True}}}}
        self.END = '__END__'
        self.offensive_trie = {}
        self.max_len = 1
        with open(self.datafile, 'r') as f:
            for p in f.read().splitlines():
                self.add_phrase(p)

    def add_phrase(self, phrase):
        """Adds a single phrase to the trie."""
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

    def check_sequence(self, toks, idx, node):
        """Check if words from the sequence are in the trie.

        This checks phrases made from
        toks[i], toks[i:i+2] ... toks[i:i + self.max_len]
        """
        right = min(idx + self.max_len, len(toks))
        for i in range(idx, right):
            if toks[i] in node:
                node = node[toks[i]]
                if self.END in node:
                    return True
            else:
                break
        return False


    def contains_offensive_language(self, text):
        """Determines if text contains any offensive words from the list."""
        if type(text) is str:
            toks = self.tokenize(text.lower())
        elif type(text) is list or type(text) is tuple:
            toks = text

        for i in range(len(toks)):
            res = self.check_sequence(toks, i, self.offensive_trie)
            if res:
                return True

        return False
