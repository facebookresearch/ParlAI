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

# to accomodate functions imported from torch_agent.py
from parlai.core.distributed_utils import is_primary_worker
from parlai.core.build_data import modelzoo_path



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
                           
                           
        # my arguments
        # from torch_agent.py
        agent.add_argument(
            '-emb', '--embedding-type', default='random',
            choices=['random', 'glove', 'glove-fixed', 'glove-twitter-fixed',
                     'fasttext', 'fasttext-fixed', 'fasttext_cc',
                     'fasttext_cc-fixed'],
            help='Choose between different strategies for initializing word '
                 'embeddings. Default is random, but can also preinitialize '
                 'from Glove or Fasttext. Preinitialized embeddings can also '
                 'be fixed so they are not updated during training.')
                 
        agent.add_argument(
            '-wcidf', '--weight-criterion-idf', type='bool', default=False,
            help='Whether to weight the loss with the idf weights '
                '(must be pre-calculated)')
                 
                 
        LanguageModelAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        # self.episode_history = [self.END_IDX,] # OAD
        self.episode_history = [] # OAD
#         self.episode_history_text = 'endofmessage endofsegment' # OAD
        self.episode_history_text = 'endofsegment' # OAD
        
        
        if not self.states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(self.model.encoder.weight, opt['embedding_type'])
        
        # Weight token importance, if desired. 
        if opt['weight_criterion_idf']:
            import pickle
            
            datasetname = opt['task']
            with open('data/%s/%s/tfidf_vectorizer.pkl'% (datasetname, datasetname), 'rb') as f:
                
                vectorizer = pickle.load(f)
                word_weights = torch.zeros(len(self.dict.freq.keys()))
                
                for tok in self.dict.freq.keys(): 
                    
                    word_idf = vectorizer.idf_[vectorizer.vocabulary_[tok]]
                    word_weights[self.dict.tok2ind[tok]] = word_idf
                    
                    # word_weights[self.dict.tok2ind[tok]] = 1./(float(self.dict.freq[tok]) + 1.)**.5
            
            # set up criteria
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX,
                                                 size_average=False, 
                                                 weight=word_weights)
            if self.use_cuda:
                # push to cuda
                self.criterion.cuda()                                  
        
        self.id = 'LanguageModelRetriever'
        
        
    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        # self.episode_history = [self.END_IDX,] # OAD
        self.episode_history = [] # OAD
#         self.episode_history_text = 'endofmessage endofsegment' # OAD
        self.episode_history_text = 'endofsegment' # OAD
        self.reset_metrics()
        
    def observe(self, observation):
        return self.observe_appending_special_tokens(observation)            
            
                    
    def observe_appending_special_tokens(self, observation):
        
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
        
        
        if obs['turn_num'] == 0: 
        
            if obs['first_message'] != '':
            
                response_formated = ' endofsegment '.join(sent_tokenize(obs['first_message']))
                response_formated += ' endofsegment endofmessage'
            
                self.episode_history_text = response_formated
                self.episode_history = self.parse(obs['first_message']
                                                        + ' endofmessage endofsegment')
            
        
        if is_training:
            
            if 'labels' in obs:
                
                obs['labels'][0] = obs['labels'][0]
                
                vec = self.parse(obs['labels'][0])
                vec.append(self.END_IDX)
#                 self.next_observe += vec
            
            # OAD: stop accumulating at the end of an episode.
            if obs['episode_done']:
                self.episode_history = [self.END_IDX,]
#                 self.episode_history = []
#                 self.episode_history_text = 'endofmessage endofsegment'
                self.episode_history_text = 'endofsegment'
                
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
                
                # truncate to approximate multiple of seq_len history
                obs['text'] = ' '.join(self.episode_history_text.split(' ')[-4*seq_len:])
                    
            # OAD: stop accumulating at the end of an episode.
            if obs['episode_done']:
#                 self.episode_history_text = 'endofmessage endofsegment'
                self.episode_history_text = 'endofsegment'
            else:
                # add end tokens between message moves #TODO: replace sent_tokenize with spacy
                response_formated = ' endofsegment '.join(sent_tokenize(obs['eval_labels'][0]))
                response_formated += ' endofsegment endofmessage' # endofsegment'
#                 response_formated = ' '.join(sent_tokenize(obs['eval_labels'][0]))
#                 response_formated += ' endofmessage'
                
                # note: don't end history on EOS, as that's added to the input (history) 
                # automatically at decode time. This should probably be a TODO to change. 
                
                # accumulate episode history
                if self.episode_history_text == 'endofsegment':
                    self.episode_history_text = ' '.join([self.episode_history_text, 
                                                        response_formated])
                else: 
                    self.episode_history_text = ' '.join([self.episode_history_text, 
                                                        'endofsegment',
                                                        response_formated])
                                                    
            self.observation = obs
            
            return obs
            
            
            
            
    def _get_embtype(self, emb_type):
    
        ''' copied from torch_agent.py '''
        
        # set up preinitialized embeddings
        try:
            import torchtext.vocab as vocab
        except ImportError as ex:
            print('Please install torch text with `pip install torchtext`')
            raise ex
        pretrained_dim = 300
        if emb_type.startswith('glove'):
            if 'twitter' in emb_type:
                init = 'glove-twitter'
                name = 'twitter.27B'
                pretrained_dim = 200
            else:
                init = 'glove'
                name = '840B'
            embs = vocab.GloVe(
                name=name, dim=pretrained_dim,
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:glove_vectors'))
        elif emb_type.startswith('fasttext_cc'):
            init = 'fasttext_cc'
            from parlai.zoo.fasttext_cc_vectors.build import url as fasttext_cc_url
            embs = vocab.Vectors(
                name='crawl-300d-2M.vec',
                url=fasttext_cc_url,
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:fasttext_cc_vectors'))
        elif emb_type.startswith('fasttext'):
            init = 'fasttext'
            embs = vocab.FastText(
                language='en',
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:fasttext_vectors'))
        else:
            raise RuntimeError('embedding type {} not implemented. check arg, '
                               'submit PR to this function, or override it.'
                               ''.format(emb_type))
        return embs, init
        
        
    def _project_vec(self, vec, target_dim, method='random'):
    
        ''' copied from torch_agent.py '''
        
        """If needed, project vector to target dimensionality.
        Projection methods implemented are the following:
        random - random gaussian matrix multiplication of input vector
        :param vec:        one-dimensional vector
        :param target_dim: dimension of returned vector
        :param method:     projection method. will be used even if the dim is
                           not changing if method ends in "-force".
        """
        pre_dim = vec.size(0)
        if pre_dim != target_dim or method.endswith('force'):
            if method.startswith('random'):
                # random projection
                if not hasattr(self, 'proj_rp'):
                    self.proj_rp = torch.Tensor(pre_dim, target_dim).normal_()
                    # rescale so we're not destroying norms too much
                    # http://scikit-learn.org/stable/modules/random_projection.html#gaussian-random-projection
                    self.proj_rp /= target_dim
                return torch.mm(vec.unsqueeze(0), self.proj_rp)
            else:
                # TODO: PCA
                # TODO: PCA + RP
                # TODO: copy
                raise RuntimeError('Projection method not implemented: {}'
                                   ''.format(method))
        else:
            return vec
            
                    
    def _copy_embeddings(self, weight, emb_type, log=True):
        
        ''' copied from torch_agent.py '''
        
        """Copy embeddings from the pretrained embeddings to the lookuptable.
        :param weight:   weights of lookup table (nn.Embedding/nn.EmbeddingBag)
        :param emb_type: pretrained embedding type
        """
        if not is_primary_worker():
            # we're in distributed mode, copying embeddings in the workers
            # slows things down considerably
            return
        embs, name = self._get_embtype(emb_type)
        cnt = 0
        for w, i in self.dict.tok2ind.items():
            if w in embs.stoi:
                vec = self._project_vec(embs.vectors[embs.stoi[w]],
                                        weight.size(1))
                weight.data[i] = vec
                cnt += 1

        if log:
            print('Initialized embeddings for {} tokens ({}%) from {}.'
                  ''.format(cnt, round(cnt * 100 / len(self.dict), 1), name))
