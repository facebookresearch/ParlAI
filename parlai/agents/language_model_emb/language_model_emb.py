from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent


from parlai.agents.language_model.language_model import LanguageModelAgent
# from parlai.core.utils import PaddingUtils, round_sigfigs
# from parlai.core.thread_utils import SharedTable

# for initializing model
from parlai.agents.language_model.modules import RNNModel
from parlai.core.build_data import modelzoo_path
from parlai.core.distributed_utils import is_primary_worker

import torch
from torch.autograd import Variable
import torch.nn as nn

import os
import math
import json


class LanguageModelEmbAgent(LanguageModelAgent):
    """ Agent which trains an RNN on a language modeling task.
    It is adapted from the language model featured in Pytorch's examples repo
    here: <https://github.com/pytorch/examples/tree/master/word_language_model>.
    """

#     @staticmethod
#     def dictionary_class():
#         return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Language Model Emb Arguments')
        agent.add_argument(
            '-emb', '--embedding-type', default='random',
            choices=['random', 'glove', 'glove-fixed', 'glove-twitter-fixed',
                     'fasttext', 'fasttext-fixed', 'fasttext_cc',
                     'fasttext_cc-fixed'],
            help='Choose between different strategies for initializing word '
                 'embeddings. Default is random, but can also preinitialize '
                 'from Glove or Fasttext. Preinitialized embeddings can also '
                 'be fixed so they are not updated during training.')
        
        
        """Copied from language_model.py"""
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
        
        LanguageModelEmbAgent.dictionary_class().add_cmdline_args(argparser)
        
        return agent




#     def __init__(self, opt, shared=None):
#         """Set up model if shared params not set, otherwise no work to do."""
#         super().__init__(opt, shared)
#         
#         self.id = 'LanguageModelEmb'



    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        opt = self.opt  # there is a deepcopy in the init
        self.metrics = {
            'loss': 0,
            'num_tokens': 0,
            'lmloss': 0,
            'lm_num_tokens': 0
        }
        self.states = {}
        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        self.batchsize = opt.get('batchsize', 1)
        self.use_person_tokens = opt.get('person_tokens', True)
        self.sampling_mode = opt.get('sampling_mode', False)

        if shared:
            # set up shared properties
            self.opt = shared['opt']
            opt = self.opt
            self.dict = shared['dict']

            self.model = shared['model']
            self.metrics = shared['metrics']

            # get NULL token and END token
            self.NULL_IDX = self.dict[self.dict.null_token]
            self.END_IDX = self.dict[self.dict.end_token]

            if 'states' in shared:
                self.states = shared['states']

            if self.use_person_tokens:
                # add person1 and person2 tokens
                self.dict.add_to_dict(self.dict.tokenize("PERSON1"))
                self.dict.add_to_dict(self.dict.tokenize("PERSON2"))

        else:
            # this is not a shared instance of this class, so do full init
            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            init_model = None
            # check first for 'init_model' for loading model from file
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                init_model = opt['init_model']
            # next check for 'model_file', this would override init_model
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                init_model = opt['model_file']

            # for backwards compatibility: will only be called for older models
            # for which .opt file does not exist
            if (init_model is not None and
                    not os.path.isfile(init_model + '.opt')):
                new_opt = self.load_opt(init_model)
                # load model parameters if available
                print('[ Setting opt from {} ]'.format(
                    init_model
                ))
                # since .opt file does not exist, save one for future use
                print("Saving opt file at:", init_model + ".opt")
                with open(init_model + '.opt', 'w') as handle:
                    json.dump(new_opt, handle)
                opt = self.override_opt(new_opt)

            if ((init_model is not None and
                    os.path.isfile(init_model + '.dict')) or
                    opt['dict_file'] is None):
                opt['dict_file'] = init_model + '.dict'

            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)
            self.id = 'LanguageModelEmb'

            # get NULL token and END token
            self.NULL_IDX = self.dict[self.dict.null_token]
            self.END_IDX = self.dict[self.dict.end_token]

            if self.use_person_tokens:
                # add person1 and person2 tokens
                self.dict.add_to_dict(self.dict.tokenize("PERSON1"))
                self.dict.add_to_dict(self.dict.tokenize("PERSON2"))

            # set model
            self.model = RNNModel(opt, len(self.dict))
            

            if init_model is not None:
                self.load(init_model)
                
            else: # OAD
                # initialize embedding to chosen: OAD
                # based on initialization in build_model() 
                # from parlai/agents/seq2seq/seq2seq.py
                if opt['embedding_type'] != 'random':
                    # only set up embeddings if not loading model
                    self._copy_embeddings(self.model.encoder.weight,
                                          opt['embedding_type'])

            if self.use_cuda:
                self.model.cuda()

        self.next_observe = []
        self.next_batch = []

        self.is_training = True

        self.clip = opt.get('gradient_clip', 0.25)
        # set up criteria
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX,
                                             size_average=False)
        if self.use_cuda:
            # push to cuda
            self.criterion.cuda()
        # init hidden state
        self.hidden = self.model.init_hidden(self.batchsize)
        # init tensor of end tokens
        self.ends = torch.LongTensor([self.END_IDX for _ in range(self.batchsize)])
        if self.use_cuda:
            self.ends = self.ends.cuda()
        # set up model and learning rate scheduler parameters
        self.lr = opt['learningrate']
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.best_val_loss = self.states.get('best_val_loss', None)
        self.lr_factor = opt['lr_factor']
        if self.lr_factor < 1.0:
            self.lr_patience = opt['lr_patience']
            self.lr_min = opt['lr_minimum']
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=self.lr_factor, verbose=True,
                patience=self.lr_patience, min_lr=self.lr_min)
            # initial step for scheduler if self.best_val_loss is initialized
            if self.best_val_loss is not None:
                self.scheduler.step(self.best_val_loss)
        else:
            self.scheduler = None

        self.reset()        
        
        
        
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
                  
                  
                  
                  