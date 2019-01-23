#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque, namedtuple
from collections.abc import MutableMapping
from multiprocessing import Lock, RawArray
from operator import attrgetter

import ctypes
import importlib
import math
import os
import random
import sys
import torch


class Beam(object):
    """Generic beam class. It keeps information about beam_size hypothesis."""

    def __init__(self, beam_size, min_length=3, padding_token=0, bos_token=1,
                 eos_token=2, min_n_best=3, cuda='cpu'):
        """Instantiate Beam object.

        :param beam_size: number of hypothesis in the beam
        :param min_length: minimum length of the predicted sequence
        :param padding_token: Set to 0 as usual in ParlAI
        :param bos_token: Set to 1 as usual in ParlAI
        :param eos_token: Set to 2 as usual in ParlAI
        :param min_n_best: Beam will not be done unless this amount of finished
                           hypothesis (with EOS) is done
        :param cuda: What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = cuda
        # recent score for each hypo in the beam
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(
            self.device)
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [torch.Tensor(self.beam_size).long()
                        .fill_(padding_token).to(self.device)]
        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.HypothesisTail = namedtuple(
            'HypothesisTail', ['timestep', 'hypid', 'score', 'tokenid'])
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best

    def get_output_from_current_step(self):
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        return self.bookkeep[-1]

    def advance(self, softmax_probs):
        voc_size = softmax_probs.size(-1)
        if len(self.bookkeep) == 0:
            # the first step we take only the first hypo into account since all
            # hypos are the same initially
            beam_scores = softmax_probs[0]
        else:
            # we need to sum up hypo scores and curr softmax scores before topk
            # [beam_size, voc_size]
            beam_scores = (softmax_probs +
                           self.scores.unsqueeze(1).expand_as(softmax_probs))
            for i in range(self.outputs[-1].size(0)):
                #  if previous output hypo token had eos
                # we penalize those word probs to never be chosen
                if self.outputs[-1][i] == self.eos:
                    # beam_scores[i] is voc_size array for i-th hypo
                    beam_scores[i] = -1e20

        flatten_beam_scores = beam_scores.view(-1)  # [beam_size * voc_size]
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(
                flatten_beam_scores, self.beam_size, dim=-1)

        self.scores = best_scores
        self.all_scores.append(self.scores)
        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs / voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                #  this is finished hypo, adding to finished
                eostail = self.HypothesisTail(timestep=len(self.outputs) - 1,
                                              hypid=hypid,
                                              score=self.scores[hypid],
                                              tokenid=self.eos)
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """Get single best hypothesis.

        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return (self.get_hyp_from_finished(top_hypothesis_tail),
                top_hypothesis_tail.score)

    def get_hyp_from_finished(self, hypothesis_tail):
        """Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep: timestep with range up to len(self.outputs)-1
        :param hyp_id: id with range up to beam_size-1
        :return: hypothesis sequence
        """
        assert (self.outputs[hypothesis_tail.timestep]
                [hypothesis_tail.hypid] == self.eos)
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(self.HypothesisTail(
                timestep=i, hypid=endback, score=self.all_scores[i][endback],
                tokenid=self.outputs[i][endback]))
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    def get_pretty_hypothesis(self, list_of_hypotails):
        hypothesis = []
        for i in list_of_hypotails:
            hypothesis.append(i.tokenid)

        hypothesis = torch.stack(list(reversed(hypothesis)))

        return hypothesis

    def get_rescored_finished(self, n_best=None):
        """

        :param n_best: how many n best hypothesis to return
        :return: list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, 0.65)
            rescored_finished.append(self.HypothesisTail(
                timestep=finished_item.timestep, hypid=finished_item.hypid,
                score=finished_item.score / length_penalty,
                tokenid=finished_item.tokenid))

        srted = sorted(rescored_finished, key=attrgetter('score'),
                       reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def check_finished(self):
        """Checks if self.finished is empty and add hyptail in that case.

        This will be suboptimal hypothesis since the model did not get any EOS

        :returns: None
        """
        if len(self.finished) == 0:
            # we change output because we want outputs to have eos
            # to pass assert in L102, it is ok since empty self.finished
            # means junk prediction anyway
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(timestep=len(self.outputs) - 1,
                                          hypid=0,
                                          score=self.all_scores[-1][0],
                                          tokenid=self.outputs[-1][0])

            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """Creates pydot graph representation of the beam.

        :param outputs: self.outputs from the beam
        :param dictionary: tok 2 word dict to save words in the tree nodes
        :returns: pydot graph
        """
        try:
            import pydot
        except ImportError:
            print("Please install pydot package to dump beam visualization")

        graph = pydot.Dot(graph_type='digraph')
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]
        if n_best is None:
            n_best = int(self.beam_size / 2)

        # get top nbest hyp
        top_hyp_idx_n_best = []
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue',
                         'green2', 'tan']
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            # do not include EOS since it has rescored score not from original
            # self.all_scores, we color EOS with black
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(
                hyptail))

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0  # collapse all __NULL__ nodes
                node_tail = self.HypothesisTail(timestep=tstep, hypid=hypid,
                                                score=all_scores[tstep][hypid],
                                                tokenid=token)
                color = 'white'
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = (
                    "<{}".format(dictionary.vec2txt([token])
                                 if dictionary is not None else token) +
                    " : " +
                    "{:.{prec}f}>".format(all_scores[tstep][hypid], prec=3))

                graph.add_node(pydot.Node(
                    node_tail.__repr__(), label=label, fillcolor=color,
                    style='filled',
                    xlabel='{}'.format(rank) if rank is not None else ''))

        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node(
                    '"{}"'.format(self.HypothesisTail(
                        timestep=revtstep, hypid=prev_id,
                        score=all_scores[revtstep][prev_id],
                        tokenid=outputs[revtstep][prev_id]).__repr__()))[0]
                to_node = graph.get_node(
                    '"{}"'.format(self.HypothesisTail(
                        timestep=revtstep + 1, hypid=i,
                        score=all_scores[revtstep + 1][i],
                        tokenid=outputs[revtstep + 1][i]).__repr__()))[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph

def maintain_dialog_history(history, observation, reply='',
                            historyLength=1, useReplies='label_else_model',
                            dict=None, useStartEndIndices=True,
                            splitSentences=False):
    """Keeps track of dialog history, up to a truncation length.
    Either includes replies from the labels, model, or not all using param 'replies'."""

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

class PaddingUtils(object):
    """
    Class that contains functions that help with padding input and target tensors.
    """
    @classmethod
    def pad_text(cls, observations, dictionary,
                 end_idx=None, null_idx=0, dq=False, eval_labels=True,
                 truncate=None):
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
                answers[valid_inds[i]] = curr_pred

            if random.random() > (1 - report_freq):
                # log sometimes
                print('TEXT: ', observations[valid_inds[i]]['text'])
                print('PREDICTION: ', curr_pred, '\n~')
        return


class SharedTable(MutableMapping):
    """Provides a simple shared-memory table of integers, floats, or strings.
    Use this class as follows:

    .. code-block:: python

        tbl = SharedTable({'cnt': 0})
        with tbl.get_lock():
            tbl['startTime'] = time.time()
        for i in range(10):
            with tbl.get_lock():
                tbl['cnt'] += 1
    """

    types = {
        int: ctypes.c_int,
        float: ctypes.c_float,
        bool: ctypes.c_bool,
    }

    def __init__(self, init_dict=None):
        """Create a shared memory version of each element of the initial
        dictionary. Creates an empty array otherwise, which will extend
        automatically when keys are added.

        Each different type (all supported types listed in the ``types`` array
        above) has its own array. For each key we store an index into the
        appropriate array as well as the type of value stored for that key.
        """
        # idx is dict of {key: (array_idx, value_type)}
        self.idx = {}
        # arrays is dict of {value_type: array_of_ctype}
        self.arrays = {}
        self.tensors = {}

        if init_dict:
            sizes = {typ: 0 for typ in self.types.keys()}
            for k, v in init_dict.items():
                if is_tensor(v):
                    # add tensor to tensor dict--don't try to put in rawarray
                    self.tensors[k] = v
                    continue
                elif type(v) not in sizes:
                    raise TypeError('SharedTable does not support values of ' +
                                    'type ' + str(type(v)))
                sizes[type(v)] += 1
            # pop tensors from init_dict
            for k in self.tensors.keys():
                init_dict.pop(k)
            # create raw arrays for each type
            for typ, sz in sizes.items():
                self.arrays[typ] = RawArray(self.types[typ], sz)
            # track indices for each key, assign them to their typed rawarray
            idxs = {typ: 0 for typ in self.types.keys()}
            for k, v in init_dict.items():
                val_type = type(v)
                self.idx[k] = (idxs[val_type], val_type)
                if val_type == str:
                    v = sys.intern(v)
                self.arrays[val_type][idxs[val_type]] = v
                idxs[val_type] += 1
        # initialize any needed empty arrays
        for typ, ctyp in self.types.items():
            if typ not in self.arrays:
                self.arrays[typ] = RawArray(ctyp, 0)
        self.lock = Lock()

    def __len__(self):
        return len(self.idx) + len(self.tensors)

    def __iter__(self):
        return iter([k for k in self.idx] + [k for k in self.tensors])

    def __contains__(self, key):
        return key in self.idx or key in self.tensors

    def __getitem__(self, key):
        """Returns shared value if key is available."""
        if key in self.tensors:
            return self.tensors[key]
        elif key in self.idx:
            idx, typ = self.idx[key]
            return self.arrays[typ][idx]
        else:
            raise KeyError('Key "{}" not found in SharedTable'.format(key))

    def __setitem__(self, key, value):
        """If key is in table, update it. Otherwise, extend the array to make
        room. This uses additive resizing not multiplicative, since the number
        of keys is not likely to change frequently during a run, so do not abuse
        it.
        Raises an error if you try to change the type of the value stored for
        that key--if you need to do this, you must delete the key first.
        """
        val_type = type(value)
        if 'Tensor' in str(val_type):
            self.tensors[key] = value
            return
        if val_type not in self.types:
            raise TypeError('SharedTable does not support type ' + str(type(value)))
        if val_type == str:
            value = sys.intern(value)
        if key in self.idx:
            idx, typ = self.idx[key]
            if typ != val_type:
                raise TypeError(('Cannot change stored type for {key} from ' +
                                 '{v1} to {v2}. You need to del the key first' +
                                 ' if you need to change value types.'
                                 ).format(key=key, v1=typ, v2=val_type))
            self.arrays[typ][idx] = value
        else:
            raise KeyError('Cannot add more keys to the shared table as '
                           'they will not be synced across processes.')

    def __delitem__(self, key):
        if key in self.tensors:
            del self.tensors[key]
        elif key in self.idx:
            del self.idx[key]
        else:
            raise KeyError('Key "{}" not found in SharedTable'.format(key))

    def __str__(self):
        """Returns simple dict representation of the mapping."""
        lhs = [
            '{k}: {v}'.format(k=key, v=self.arrays[typ][idx])
            for key, (idx, typ) in self.idx.items()
        ]
        rhs = ['{k}: {v}'.format(k=k, v=v) for k, v in self.tensors.items()]
        return '{{{}}}'.format(', '.join(lhs + rhs))

    def __repr__(self):
        """Returns the object type and memory location with the mapping."""
        representation = super().__repr__()
        return representation.replace('>', ': {}>'.format(str(self)))

    def get_lock(self):
        return self.lock


def is_tensor(v):
    if type(v).__module__.startswith('torch'):
        import torch
        return torch.is_tensor(v)
    return False


def modelzoo_path(datapath, path):
    """If path starts with 'models', then we remap it to the model zoo path
    within the data directory (default is ParlAI/data/models).
    We download models from the model zoo if they are not here yet.

    """
    if path is None:
        return None
    if not path.startswith('models:'):
        return path
    else:
        # Check if we need to download the model
        animal = path[7:path.rfind('/')].replace('/', '.')
        if '.' not in animal:
            animal += '.build'
        module_name = 'parlai.zoo.{}'.format(animal)
        try:
            my_module = importlib.import_module(module_name)
            download = getattr(my_module, 'download')
            download(datapath)
        except (ImportError, AttributeError):
            pass

        return os.path.join(datapath, 'models', path[7:])
