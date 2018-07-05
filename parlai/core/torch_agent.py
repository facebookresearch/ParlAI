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

Batch = namedtuple("Batch", [
    # bsz x seqlen tensor containing the parsed text data
    "text_vec",
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
    # list of lists of tensors. outer list has size bsz, inner lists vary in
    # size based on the number of candidates for each row in the batch.
    "candidates",
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
        agent.add_argument(
            '-tr', '--truncate', default=-1, type=int,
            help='Truncate input lengths to increase speed / use less memory.')
        agent.add_argument(
            '-histd', '--history-dialog', default=-1, type=int,
            help='Number of past dialog utterances to remember.')
        agent.add_argument(
            '-histr', '--history-replies', default='label_else_model', type=str,
            choices=['none', 'model', 'label', 'label_else_model'],
            help='Keep replies in the history, or not.')
        agent.add_argument(
            '--no-cuda', type='bool', default=False,
            help='disable GPUs even if available. otherwise, will use GPUs if '
                 'available on the device.')
        agent.add_argument(
            '--gpu', type=int, default=-1, help='which GPU device to use')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not shared:
            # Need to set up the model from scratch
            self.dict = DictionaryAgent(opt)
        else:
            # ... copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']

        if opt.get('numthreads', 1) > 1:
            torch.set_num_threads(1)

        # check for cuda
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

    def vectorize(self, obs, add_start=True, add_end=True, truncate=None):
        """
        Converts 'text' and 'label'/'eval_label' field to vectors.

        :param obs: single observation from observe function
        :param add_start: default True, adds the end token to each label
        :param add_end: default True, adds the end token to each label
        """
        if 'text' not in obs:
            return obs
        # convert 'text' into tensor of dictionary indices
        vec_text = deque(self.dict.txt2vec(obs['text']), maxlen=truncate)
        obs['text_vec'] = torch.LongTensor(vec_text)
        if self.use_cuda:
            obs['text_vec'] = obs['text_vec'].cuda()

        label_type = None
        if 'labels' in obs:
            label_type = 'labels'
        elif 'eval_labels' in obs:
            label_type = 'eval_labels'

        if label_type is not None:
            label = random.choice(obs[label_type])
            vec_label = deque(maxlen=truncate)
            if add_start:
                vec_label.append(self.START_IDX)
            vec_label += self.dict.txt2vec(label)
            if add_end:
                vec_label.append(self.END_IDX)
            new_label = torch.LongTensor(vec_label)
            if self.use_cuda:
                new_label = new_label.cuda()
            obs[label_type + '_vec'] = new_label
            obs[label_type]
        return obs

    def batchify(self, obs_batch, sort=False, is_valid=lambda obs: 'text_vec' in obs):
        """Creates a batch of valid observations from an unchecked batch, where
        a valid observation is one that passes the lambda provided to the function.
        Assumes each observation has been vectorized by vectorize function.

        Returns a namedtuple Batch. See original definition for in-depth
        explanation of each field.

        :param obs_batch: list of vectorized observations
        :param sort:      default False, orders the observations by length of vector
        :param is_valid:  default function that checks if 'text_vec' is in the
                          observation, determines if an observation is valid
        """
        if len(obs_batch) == 0:
            return Batch(None, None, None, None, None, None)

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch(None, None, None, None, None, None)

        valid_inds, exs = zip(*valid_obs)

        x_text = [ex['text_vec'] for ex in exs]
        x_lens = [ex.shape[0] for ex in x_text]

        if sort:
            ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

            exs = [exs[k] for k in ind_sorted]
            valid_inds = [valid_inds[k] for k in ind_sorted]
            x_text = [x_text[k] for k in ind_sorted]

        padded_xs = torch.LongTensor(len(exs),
                                     max(x_lens)).fill_(self.NULL_IDX)
        if self.use_cuda:
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
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex[field + "_vec"] for i, ex in enumerate(exs)]
            labels = [ex[field] for i, ex in enumerate(exs)]
            y_lens = [y.shape[0] for y in label_vecs]
            padded_ys = torch.LongTensor(len(exs),
                                         max(y_lens)).fill_(self.NULL_IDX)
            if self.use_cuda:
                padded_ys = padded_ys.cuda()
            for i, y in enumerate(label_vecs):
                if y.shape[0] != 0:
                    padded_ys[i, :y.shape[0]] = y
            ys = padded_ys

        cands = None
        return Batch(xs, ys, labels, valid_inds, cands)

    def match_batch(self, batch_reply, valid_inds, predictions=None,
                    candidate_preds=None):
        """Match sub-batch of predictions to the original batch indices.

        Batches may be only partially filled (e.g. when completing the remainder
        at the end of the validation or test set), or we may want to sort by
        e.g. the length of the input sequences if using pack_padded_sequence.
        This matches rows back with their original row in the batch for
        calculating metrics like accuracy.

        :param batch_reply: full-batchsize list of message dictionaries to put
            responses into
        :param predictions: sub-batchsize list of text outputs from model. may
            be None (default) if model chooses to not answer.
        :param valid_inds: original indices of the predictions
        :param candidate_preds: sub-batchsize list of lists of text outputs
            ranked by the model. may be None (default) if model isn't ranking.
        """
        if predictions is not None:
            for i, response in zip(valid_inds, predictions):
                batch_reply[i]['text'] = response
        if candidate_preds is not None:
            for i, cands in zip(valid_inds, candidate_preds):
                batch_reply[i]['text_candidates'] = cands
        return batch_reply

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
        """Calls batch_act with the singleton batch."""
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        is_training = any(['labels' in obs for obs in observations])

        vec_obs = [self.vectorize(obs) for obs in observations]

        batch = self.batchify(vec_obs)

        if batch.text_vec is None:
            return batch_reply

        if is_training:
            output = self.train_step(batch.text_vec, batch.label_vec,
                                     batch.candidates)
        else:
            output = self.eval_step(batch.text_vec, batch.label_vec,
                                    batch.candidates)

        if isinstance(output, tuple):
            predictions = output[0]
            candidate_preds = output[1]
        else:
            predictions = output
            candidate_preds = None


        self.match_batch(batch_reply, batch.valid_indices,
                         predictions=predictions,
                         candidate_preds=candidate_preds)

        return batch_reply

    def train_step(self, xs, ys=None, cands=None, *args, **kwargs):
        raise NotImplementedError('Abstract class: user must implement batch_act')

    def eval_step(self, xs, ys=None, cands=None, *args, **kwargs):
        raise NotImplementedError('Abstract class: user must implement batch_act')
