# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

try:
    import torch
except ImportError as e:
    raise ImportError('Need to install Pytorch: go to pytorch.org')

from collections import deque, namedtuple
import pickle
import random
import copy
import math
from operator import attrgetter

Batch = namedtuple("Batch", [
    # bsz x seqlen tensor containing the parsed text data
    "text_vec",
    # bsz x 1 tensor containing the lengths of the text in same order as
    # text_vec; necessary for pack_padded_sequence
    "text_lengths",
    # bsz x seqlen tensor containing the parsed label (one per batch row)
    "label_vec",
    # list of length bsz containing the selected label for each batch row (some
    # datasets have multiple labels per input example)
    "labels",
    # list of length bsz containing the original indices of each example in the
    # batch. we use these to map predictions back to their proper row, since
    # e.g. we may sort examples by their length or some examples may be
    # invalid.
    "valid_indices",
])


class TorchAgent(Agent):
    """A provided base agent for any model that wants to use Torch. Exists to
    make it easier to implement a new agent. Not necessary, but reduces
    duplicated code.

    This agent serves as a common framework for all ParlAI models which want
    to use PyTorch.
    """

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('TorchAgent Arguments')
        agent.add_argument('-tr', '--truncate', default=-1, type=int,
                           help='Truncate input lengths to speed up training.')
        agent.add_argument('-histd', '--history-dialog', default=-1, type=int,
                           help='Number of past dialog examples to remember.')
        agent.add_argument('-histr', '--history-replies',
                           default='label_else_model', type=str,
                           choices=['none', 'model', 'label',
                                    'label_else_model'],
                           help='Keep replies in the history, or not.')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not shared:
            # Need to set up the model from scratch
            self.dict = self.dictionary_class()(opt)
        else:
            # ... copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']

        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        if self.use_cuda:
            if not shared:
                print('[ Using CUDA ]')
            torch.cuda.device(opt['gpu'])

        self.NULL_IDX = self.dict[self.dict.null_token]
        self.END_IDX = self.dict[self.dict.end_token]
        self.START_IDX = self.dict[self.dict.start_token]

        self.history = {}
        self.truncate = opt['truncate']
        self.history_dialog = opt['history_dialog']
        self.history_replies = opt['history_replies']

    def share(self):
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict
        return shared

    def vectorize(self, obs, addStartIdx=True, addEndIdx=True):
        """
        Converts 'text' and 'label'/'eval_label' field to vectors.

        :param obs: single observation from observe function
        :param addEndIdx: default True, adds the end token to each label
        """
        if 'text' not in obs:
            return obs
        # convert 'text' field to vector using self.txt2vec and then a tensor
        obs['text_vec'] = torch.LongTensor(self.dict.txt2vec(obs['text']))
        if self.use_cuda:
            obs['text_vec'] = obs['text_vec'].cuda()

        label_type = None
        if 'labels' in obs:
            label_type = 'labels'
        elif 'eval_labels' in obs:
            label_type = 'eval_labels'

        if label_type is not None:
            new_labels = []
            for label in obs[label_type]:
                vec_label = self.dict.txt2vec(label)
                if addStartIdx:
                    vec_label.insert(0, self.START_IDX)
                if addEndIdx:
                    vec_label.append(self.END_IDX)
                new_label = torch.LongTensor(vec_label)
                if self.use_cuda:
                    new_label = new_label.cuda()
                new_labels.append(new_label)
            obs[label_type + "_vec"] = new_labels
        return obs

    def map_valid(self, obs_batch, sort=True, is_valid=lambda obs: 'text_vec' in obs):
        """Creates a batch of valid observations from an unchecked batch, where
        a valid observation is one that passes the lambda provided to the function.
        Assumes each observation has been vectorized by vectorize function.

        Returns a namedtuple Batch. See original definition for in-depth
        explanation of each field.

        :param obs_batch: list of vectorized observations
        :param sort:      default True, orders the observations by length of vector
        :param is_valid:  default function that checks if 'text_vec' is in the
                          observation, determines if an observation is valid
        """
        if len(obs_batch) == 0:
            return Batch(None, None, None, None, None)

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch(None, None, None, None, None)

        valid_inds, exs = zip(*valid_obs)

        x_text = [ex['text_vec'] for ex in exs]
        x_lens = [ex.shape[0] for ex in x_text]

        if sort:
            ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

            exs = [exs[k] for k in ind_sorted]
            valid_inds = [valid_inds[k] for k in ind_sorted]
            x_text = [x_text[k] for k in ind_sorted]
            x_lens = [x_lens[k] for k in ind_sorted]

        x_lens = torch.LongTensor(x_lens)
        padded_xs = torch.LongTensor(len(exs),
                                     max(x_lens)).fill_(self.NULL_IDX)
        if self.use_cuda:
            x_lens = x_lens.cuda()
            padded_xs = padded_xs.cuda()

        for i, ex in enumerate(x_text):
            padded_xs[i, :ex.shape[0]] = ex

        xs = padded_xs

        eval_labels_avail = any(['eval_labels_vec' in ex for ex in exs])
        labels_avail = any(['labels_vec' in ex for ex in exs])
        some_labels_avail = eval_labels_avail or labels_avail

        # set up the target tensors
        ys = None
        labels = None
        if some_labels_avail:
            # randomly select one of the labels to update on (if multiple)
            if labels_avail:
                field = 'labels'
            else:
                field = 'eval_labels'

            num_choices = [len(ex.get(field + "_vec", [])) for ex in exs]
            choices = [random.choice(range(num)) if num != 0 else -1
                       for num in num_choices]
            label_vecs = [ex[field + "_vec"][choices[i]]
                          if choices[i] != -1 else torch.LongTensor([])
                          for i, ex in enumerate(exs)]
            labels = [ex[field][choices[i]]
                      if choices[i] != -1 else ''
                      for i, ex in enumerate(exs)]
            y_lens = [y.shape[0] for y in label_vecs]
            padded_ys = torch.LongTensor(len(exs),
                                         max(y_lens)).fill_(self.NULL_IDX)
            if self.use_cuda:
                padded_ys = padded_ys.cuda()
            for i, y in enumerate(label_vecs):
                if y.shape[0] != 0:
                    padded_ys[i, :y.shape[0]] = y
            ys = padded_ys

        return Batch(xs, x_lens, ys, labels, valid_inds)

    def unmap_valid(self, predictions, valid_inds, batch_size):
        """Re-order permuted predictions to the initial ordering, includes the
        empty observations.

        :param predictions: output of module's predict function
        :param valid_inds: original indices of the predictions
        :param batch_size: overall original size of batch
        """
        unpermuted = [None]*batch_size
        for pred, idx in zip(predictions, valid_inds):
            unpermuted[idx] = pred
        return unpermuted

    def maintain_dialog_history(self, observation, reply='',
                                useStartEndIndices=False,
                                splitSentences=False):
        """Keeps track of dialog history, up to a truncation length.

        :param observation: a single observation that will be added to existing
                            dialog history
        :param reply: default empty string, allows for the addition of replies
                      into the dialog history
        :param useStartEndIndices: default False, flag to determine if START
                                   and END indices should be appended to the
                                   observation
        :param splitSentences: default False, flag to determine if the
                               observation dialog is one sentence or needs to
                               be split
        """

        def parse(txt, splitSentences):
            if splitSentences:
                vec = [self.dict.txt2vec(t) for t in txt.split('\n')]
            else:
                vec = self.dict.txt2vec(txt)
            return vec

        allow_reply = True

        if 'dialog' not in self.history:
            self.history['dialog'] = deque(
                maxlen=self.truncate if self.truncate >= 0 else None
            )
            self.history['episode_done'] = False
            self.history['labels'] = []

        if self.history['episode_done']:
            self.history['dialog'].clear()
            self.history['labels'] = []
            allow_reply = False
            self.history['episode_done'] = False

        if self.history_replies != 'none' and allow_reply:
            if self.history_replies == 'model' or \
               (self.history_replies == 'label_else_model' and len(
                                                self.history['labels']) == 0):
                if reply != '':
                    self.history['dialog'].extend(parse(reply))
            elif len(self.history['labels']) > 0:
                r = self.history['labels'][0]
                self.history['dialog'].extend(parse(r, splitSentences))

        obs = observation
        if 'text' in obs:
            if useStartEndIndices:
                obs['text'] = self.dict.end_token + ' ' + obs['text']
            self.history['dialog'].extend(parse(obs['text'], splitSentences))

        self.history['episode_done'] = obs['episode_done']
        labels = obs.get('labels', obs.get('eval_labels', None))
        if labels is not None:
            if useStartEndIndices:
                self.history['labels'] = [
                    self.dict.start_token + ' ' + l for l in labels
                ]
            else:
                self.history['labels'] = labels

        return self.history['dialog']

    def save(self, path):
        """Save model parameters if model_file is set.

        Override this method for more specific saving.
        """
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            states = {}
            if hasattr(self, 'model'):  # save model params
                states['model'] = self.model.state_dict()
            if hasattr(self, 'optimizer'):  # save optimizer params
                states['optimizer'] = self.optimizer.state_dict()

            if states:  # anything found to save?
                # also store the options with the file for good measure
                states['opt'] = self.opt
                with open(path, 'wb') as write:
                    torch.save(states, write)

                # save opt file
                with open(path + ".opt", 'wb') as handle:
                    pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Return opt and model states.

        Override this method for more specific loading.
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if 'model' in states:
            self.model.load_state_dict(states['model'])
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        return states

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        """Calls batch_act with the singleton batch"""
        return self.batch_act([self.observation])[0]

    def batch_act(self):
        raise NotImplementedError("Abstract class: user must implement batch_act")


class Beam(object):
    def __init__(self, beam_size, min_length=3, padding_token=0, bos_token=1, eos_token=2, min_n_best=3, cuda='cpu'):
        """
        Generic beam class. It keeps information about beam_size hypothesis.
        :param beam_size: number of hypothesis in the beam
        :param min_length: minimum length of the predicted sequence
        :param padding_token: Set to 0 as usual in ParlAI
        :param bos_token: Set to 1 as usual in ParlAI
        :param eos_token: Set to 2 as usual in ParlAI
        :param min_n_best: Beam will not be done unless this amount of finished hypothesis (with EOS) is done
        :param cuda: What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = cuda
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(
            self.device)  # recent score for each hypo in the beam
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]  # self.scores values per each time step
        self.bookkeep = []  # backtracking id to hypothesis at previous time step
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(padding_token).to(self.device)]  # output tokens at each time step
        self.finished = []  # keeps tuples (score, time_step, hyp_id)
        self.HypothesisTail = namedtuple('HypothesisTail', ['timestep', 'hypid', 'score', 'tokenid'])
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
            #  the first step
            beam_scores = softmax_probs[
                0]  # we take only the first hypo into account since all hypos are the same initially
        else:
            #  we need to sum up hypo scores and current softmax scores before topk
            beam_scores = softmax_probs + self.scores.unsqueeze(1).expand_as(softmax_probs)  # [beam_size, voc_size]
            for i in range(self.outputs[-1].size(0)):
                #  if previous output hypo token had eos - we penalize those word probs to never be chosen
                if self.outputs[-1][i] == self.eos:
                    beam_scores[i] = -1e20  # beam_scores[i] is voc_size array for i-th hypo

        flatten_beam_scores = beam_scores.view(-1)  # [beam_size * voc_size]
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(flatten_beam_scores, self.beam_size, dim=-1)

        self.scores = best_scores
        self.all_scores.append(self.scores)
        hyp_ids = best_idxs / voc_size  # get the backtracking hypothesis id as a multiple of full voc_sizes
        tok_ids = best_idxs % voc_size  # get the actual word id from residual of the same division

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                #  this is finished hypo, adding to finished
                eostail = self.HypothesisTail(timestep=len(self.outputs) - 1, hypid=hypid, score=self.scores[hypid],
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
        """
        Helper function to get single best hypothesis
        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return self.get_hyp_from_finished(top_hypothesis_tail), top_hypothesis_tail.score

    def get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id
        :param timestep: timestep with range up to len(self.outputs)-1
        :param hyp_id: id with range up to beam_size-1
        :return: hypothesis sequence
        """
        assert self.outputs[hypothesis_tail.timestep][hypothesis_tail.hypid] == self.eos
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(self.HypothesisTail(timestep=i, hypid=endback, score=self.all_scores[i][endback],
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
            length_penalty = math.pow((1 + current_length) / 6, 0.65)  # this is from Google NMT paper
            rescored_finished.append(self.HypothesisTail(timestep=finished_item.timestep, hypid=finished_item.hypid,
                                                         score=finished_item.score / length_penalty,
                                                         tokenid=finished_item.tokenid))

        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def check_finished(self):
        """
        this function checks if self.finished is empty and adds hyptail
        in that case (this will be suboptimal hypothesis since
        the model did not get any EOS)
        :return: None
        """
        if len(self.finished) == 0:
            # we change output because we want outputs to have this eos to pass assert in L102, it is ok since empty self.finished means junk prediction anyway
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(timestep=len(self.outputs) - 1, hypid=0, score=self.all_scores[-1][0],
                                              tokenid=self.outputs[-1][0])

            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """
        Creates pydot graph representation of the beam
        :param outputs: self.outputs from the beam
        :param dictionary: tok 2 word dict to save words in the tree nodes
        :return: pydot graph
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
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue', 'green2', 'tan']
        end_color = 'yellow'
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(
                hyptail))  # do not include EOS since it has rescored score not from original self.all_scores, we color EOS with black

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0  # collapse all __NULL__ nodes
                node_tail = self.HypothesisTail(timestep=tstep, hypid=hypid, score=all_scores[tstep][hypid],
                                                tokenid=token)
                color = 'white'
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = "<{}".format(
                    dictionary.vec2txt([token]) if dictionary is not None else token) + " : " + "{:.{prec}f}>".format(
                    all_scores[tstep][hypid], prec=3)
                graph.add_node(pydot.Node(node_tail.__repr__(), label=label, fillcolor=color, style='filled',
                                          xlabel='{}'.format(rank) if rank is not None else ''))
        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node('"{}"'.format(
                    self.HypothesisTail(timestep=revtstep, hypid=prev_id, score=all_scores[revtstep][prev_id],
                                        tokenid=outputs[revtstep][prev_id]).__repr__()))[0]
                to_node = graph.get_node('"{}"'.format(
                    self.HypothesisTail(timestep=revtstep + 1, hypid=i, score=all_scores[revtstep + 1][i],
                                        tokenid=outputs[revtstep + 1][i]).__repr__()))[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph
