# 
# 
# from parlai.core.agents import Agent
# from parlai.core.dict import DictionaryAgent
# from parlai.core.utils import PaddingUtils, round_sigfigs
# from parlai.core.thread_utils import SharedTable
# from .modules import RNNModel

from parlai.agents.language_model.language_model import LanguageModelAgent
from parlai.core.dict import DictionaryAgent

import torch
# from torch.autograd import Variable
import torch.nn as nn

# import os
# import math
# import json
import copy
from nltk.tokenize import sent_tokenize




class LanguageModelRetrieverAgent(LanguageModelAgent):
    """ Modified from LanguageModelAgent:
    Agent which trains an RNN on a language modeling task.
    It is adapted from the language model featured in Pytorch's examples repo
    here: <https://github.com/pytorch/examples/tree/master/word_language_model>.
    """

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        argparser.set_defaults(batch_sort=False)
        agent = argparser.add_argument_group('Language Model Arguments')
        agent.add_argument('--init-model', type=str, default=None,
                           help='load dict/features/weights/opts from this file')
        agent.add_argument('-hs', '--hiddensize', type=int, default=200,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=200,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-dr', '--dropout', type=float, default=0.2,
                           help='dropout rate')
        agent.add_argument('-clip', '--gradient-clip', type=float, default=0.25,
                           help='gradient clipping')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('-rnn', '--rnn-class', default='LSTM',
                           help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
        agent.add_argument('-sl', '--seq-len', type=int, default=35,
                           help='sequence length')
        agent.add_argument('-tied', '--emb-tied', action='store_true',
                           help='tie the word embedding and softmax weights')
        agent.add_argument('-seed', '--random-seed', type=int, default=1111,
                           help='random seed')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-tr', '--truncate-pred', type=int, default=50,
                           help='truncate predictions')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.1,
                           help='report frequency of prediction during eval')
        agent.add_argument('-pt', '--person-tokens', type='bool', default=True,
                           help='append person1 and person2 tokens to text')
        # learning rate parameters
        agent.add_argument('-lr', '--learningrate', type=float, default=20,
                           help='initial learning rate')
        agent.add_argument('-lrf', '--lr-factor', type=float, default=1.0,
                           help='mutliply learning rate by this factor when the \
                           validation loss does not decrease')
        agent.add_argument('-lrp', '--lr-patience', type=int, default=10,
                           help='wait before decreasing learning rate')
        agent.add_argument('-lrm', '--lr-minimum', type=float, default=0.1,
                           help='minimum learning rate')
        agent.add_argument('-sm', '--sampling-mode', type='bool', default=False,
                           help='sample when generating tokens instead of taking \
                           the max and do not produce UNK token (when bs=1)')
        LanguageModelAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        # self.episode_history = [self.END_IDX,] # OAD
        self.episode_history = [] # OAD
        self.episode_history_text = 'endofmessage endofsegment' # OAD
#         self.episode_history_text = 'endofmessage' # OAD
        
        self.id = 'LanguageModelRetriever'
        
        
    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        # self.episode_history = [self.END_IDX,] # OAD
        self.episode_history = [] # OAD
        self.episode_history_text = 'endofmessage endofsegment' # OAD
#         self.episode_history_text = 'endofmessage' # OAD
        self.reset_metrics()
        
    
    def observe(self, observation):
        
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        seq_len = self.opt['seq_len']
        is_training = True
        
        if 'labels' not in obs:
            is_training = False
        if 'is_training_lambda' in self.opt and self.opt['is_training_lambda']:
            is_training = False
        
        
        
        if is_training:
            
            if 'labels' in obs:
                
                obs['labels'][0] = obs['labels'][0]
                
                vec = self.parse(obs['labels'][0])
                vec.append(self.END_IDX)
#                 self.next_observe += vec
            
            # OAD: stop accumulating at the end of an episode.
            if obs['episode_done']:
                # self.episode_history = [self.END_IDX,]
                self.episode_history = []
                self.episode_history_text = 'endofmessage endofsegment'
#                 self.episode_history_text = 'endofmessage'
                
            else:
                # accumulate episode history.
                self.episode_history += vec
#                 self.episode_history_text = ' '.join([self.episode_history_text, 
#                                                     obs['labels'][0]]) 
                self.episode_history_text = ' '.join([self.episode_history_text, 
                                                    obs['labels'][0] + ' endofsegment'])                
                
                
            if len(self.episode_history) < (seq_len + 1):
                # not enough to return to make a batch
                # we handle this case in vectorize
                # labels indicates that we are training
                self.observation = {'labels': ''}
                return self.observation
                
            else:
                vecs_to_return = []
                overlap = 3
                
                # first obs will overlap 1 with current observation
                start = max(0, len(self.episode_history) - (len(vec) + seq_len)) 
                stop = len(self.episode_history) - (seq_len + 1)
                
                # take observations of seq-len that overlap with just observed 
                for i in range(start, stop, overlap):
                    vecs_to_return.append(self.episode_history[i:i+seq_len+1])
                    
                    
                dict_to_return = {'text': '', 'labels': '', 'text2vec': vecs_to_return}
                self.observation = dict_to_return
                
                return dict_to_return
        else:
        
            if 'text' in obs:
                obs['eval_labels'][0] = obs['eval_labels'][0]
                
                # truncate to approximate multiple of seq_len history
                obs['text'] = ' '.join(self.episode_history_text.split(' ')[-4*seq_len:])
                    
            # OAD: stop accumulating at the end of an episode.
            if obs['episode_done']:
                self.episode_history_text = 'endofmessage endofsegment'
#                 self.episode_history_text = 'endofmessage'
            else:
                # add __end__ between message moves
                response_formated = ' endofsegment '.join(sent_tokenize(obs['eval_labels'][0]))
                response_formated += ' endofsegment endofmessage endofsegment'
#                 response_formated = ' '.join(sent_tokenize(obs['eval_labels'][0]))
#                 response_formated += ' endofmessage'
                
                # accumulate episode history
                self.episode_history_text = ' '.join([self.episode_history_text, 
                                                    response_formated])
                                                    
            self.observation = obs
#             print('##### self.observation after parsing: ', obs)
            
            return obs
