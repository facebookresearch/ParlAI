import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os, copy, json
from torch import optim

from parlai.core.agents import Agent
from parlai.core.torch_agent import Output
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from parlai.agents.seq2seq.modules import Seq2seq

from torch.autograd import Variable

from parlai.agents.language_model_retriever.language_model_retriever import LanguageModelRetrieverAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent

from parlai.core.dict import DictionaryAgent
from parlai.core.utils import PaddingUtils, round_sigfigs
from parlai.core.thread_utils import SharedTable

from parlai.core.utils import (
    AttrDict, argsort, padded_tensor, warn_once, round_sigfigs
)

from parlai.core.utils import padded_3d


# from parlai.core.agents import create_agent


class LinearComboModel(nn.Module):
    
    def __init__(self):
        super(LinearComboModel, self).__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        
        # initialize to equal weight
        self.lamb = torch.nn.Parameter(data=torch.tensor(0.5, dtype=torch.double), requires_grad=True)

    def forward(self, x1, x2):
        # LM then S2S
        out = torch.logsumexp(
                    torch.cat(((1.-self.lamb) * x1.reshape(-1,1), 
                                self.lamb * x2.reshape(-1,1)),dim=1), 
                                dim=1 )

        return out
    
    def project_param(self):
#         self.lamb = torch.nn.Parameter(data=torch.clamp(self.lamb, min=0., max=1.).double(), requires_grad=True)
        w = self.lamb.data
        self.lamb.data = w.clamp(0.,1.)
                                    
    def set_lambda(self, val):
        print('#### THIS SHOULD NOT BE CALLED WHEN FITTING LAMBDA ###')
        self.lamb = torch.nn.Parameter(data=torch.DoubleTensor([val,]), requires_grad=False)
        

class SmoothedDecoderAgent(Agent):
    
    
    # from TorchAgent
    @classmethod
    def optim_opts(self):
        """Fetch optimizer selection.
        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim
        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims
    
    
    @staticmethod
    def dictionary_class():
        return DictionaryAgent
        
    @classmethod
    def add_cmdline_args(cls, argparser):
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
                           
                           
#         LanguageModelAgent.dictionary_class().add_cmdline_args(argparser)
#         return agent    
    # @staticmethod
#     def add_cmdline_args(argparser):
#         """Add command-line arguments specifically for this agent."""
#         agent = argparser.add_argument_group('SmoothedDecoder Arguments')
#         agent.add_argument('-hs', '--hiddensize', type=int, default=128,
#                            help='size of the hidden layers')
#         agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
#                            help='size of the token embeddings')
#         agent.add_argument('-nl', '--numlayers', type=int, default=2,
#                            help='number of hidden layers')
#         agent.add_argument('-dr', '--dropout', type=float, default=0.1,
#                            help='dropout rate')
#         agent.add_argument('-bi', '--bidirectional', type='bool',
#                            default=False,
#                            help='whether to encode the context with a '
#                                 'bidirectional rnn')
#         agent.add_argument('-att', '--attention', default='none',
#                            choices=['none', 'concat', 'general', 'dot',
#                                     'local'],
#                            help='Choices: none, concat, general, local. '
#                                 'If set local, also set attention-length. '
#                                 '(see arxiv.org/abs/1508.04025)')
#         agent.add_argument('-attl', '--attention-length', default=48, type=int,
#                            help='Length of local attention.')
#         agent.add_argument('--attention-time', default='post',
#                            choices=['pre', 'post'],
#                            help='Whether to apply attention before or after '
#                                 'decoding.')
#         agent.add_argument('-rnn', '--rnn-class', default='lstm',
#                            choices=Seq2seq.RNN_OPTS.keys(),
#                            help='Choose between different types of RNNs.')
#         agent.add_argument('-dec', '--decoder', default='same',
#                            choices=['same', 'shared'],
#                            help='Choose between different decoder modules. '
#                                 'Default "same" uses same class as encoder, '
#                                 'while "shared" also uses the same weights. '
#                                 'Note that shared disabled some encoder '
#                                 'options--in particular, bidirectionality.')
#         agent.add_argument('-lt', '--lookuptable', default='unique',
#                            choices=['unique', 'enc_dec', 'dec_out', 'all'],
#                            help='The encoder, decoder, and output modules can '
#                                 'share weights, or not. '
#                                 'Unique has independent embeddings for each. '
#                                 'Enc_dec shares the embedding for the encoder '
#                                 'and decoder. '
#                                 'Dec_out shares decoder embedding and output '
#                                 'weights. '
#                                 'All shares all three weights.')
#         agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
#                            help='default 1, if greater then uses mixture of '
#                                 'softmax (see arxiv.org/abs/1711.03953).')
#         agent.add_argument('-idr', '--input-dropout', type=float, default=0.0,
#                            help='Probability of replacing tokens with UNK in training.')
        
        
        # from torch generator agent. 
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
            '-embp', '--embedding-projection', default='random',
            help='If pretrained embeddings have a different dimensionality '
                 'than your embedding size, strategy for projecting to the '
                 'correct size. If the dimensions are the same, this is '
                 'ignored unless you append "-force" to your choice.')
        # optimizer arguments
        agent.add_argument(
            '-opt', '--optimizer', default='sgd', choices=cls.optim_opts(),
            help='Choose between pytorch optimizers. Any member of torch.optim'
                 ' should be valid.')
        agent.add_argument(
            '-lr', '--learningrate', type=float, default=1,
            help='learning rate')
        agent.add_argument(
            '-clip', '--gradient-clip', type=float, default=0.1,
            help='gradient clipping using l2 norm')
        agent.add_argument(
            '-mom', '--momentum', default=0, type=float,
            help='if applicable, momentum value for optimizer.')
        agent.add_argument(
            '--nesterov', default=True, type='bool',
            help='if applicable, whether to use nesterov momentum.')
        agent.add_argument(
            '-nu', '--nus', default='0.7', type='floats',
            help='if applicable, nu value(s) for optimizer. can use a single '
                 'value like 0.7 or a comma-separated tuple like 0.7,1.0')
        agent.add_argument(
            '-beta', '--betas', default='0.9,0.999', type='floats',
            help='if applicable, beta value(s) for optimizer. can use a single '
                 'value like 0.9 or a comma-separated tuple like 0.9,0.999')
        # lr scheduler
        agent.add_argument(
            '--lr-scheduler', type=str, default='reduceonplateau',
            choices=['reduceonplateau', 'none', 'fixed', 'invsqrt'],
            help='Learning rate scheduler.'
        )
        agent.add_argument(
            '--lr-scheduler-patience', type=int, default=3,
            help='LR scheduler patience. In number of validation runs. If using '
                 'fixed scheduler, LR is decayed every <patience> validations.'
        )
        agent.add_argument(
            '--lr-scheduler-decay', type=float, default=0.5,
            help='Decay factor for LR scheduler, or how much LR is multiplied by '
                 'when it is lowered.'
        )
        agent.add_argument(
            '--warmup-updates', type=int, default=-1, hidden=True,
            help='Learning rate warmup period, in number of SGD updates. '
                 'Linearly scales up LR over period. Only enabled if > 0.'
        )
        agent.add_argument(
            '--warmup-rate', type=float, default=1e-4, hidden=True,
            help='Warmup learning rate *multiplier*. Initial LR is multiplied by '
                 'this value. Linearly adjusted up to 1.0 across --warmup-updates '
                 'steps.'
        )
        agent.add_argument(
            '--update-freq', type=int, default=-1, hidden=True,
            help='Accumulate gradients N times before performing an optimizer.step().'
        )
        # preprocessing arguments
        agent.add_argument(
            '-rc', '--rank-candidates', type='bool', default=False,
            help='Whether the model should parse candidates for ranking.')
        agent.add_argument(
            '-tr', '--truncate', default=-1, type=int,
            help='Truncate input lengths to increase speed / use less memory.')
        agent.add_argument(
            '--text-truncate', type=int,
            help='Text input truncation length: if not specified, this will '
                 'default to `truncate`'
        )
        agent.add_argument(
            '--label-truncate', type=int,
            help='Label truncation length: if not specified, this will default '
                 'to `truncate`'
        )
        agent.add_argument(
            '-histsz', '--history-size', default=-1, type=int,
            help='Number of past dialog utterances to remember.')
        agent.add_argument(
            '-pt', '--person-tokens', type='bool', default=False,
            help='add person tokens to history. adds __p1__ in front of input '
                 'text and __p2__ in front of past labels when available or '
                 'past utterances generated by the model. these are added to '
                 'the dictionary during initialization.')
        agent.add_argument(
            '--use-reply', default='label', hidden=True,
            choices=['label', 'model'],
            help='Which previous replies to use as history. If label, use '
            'gold dataset replies. If model, use model\'s own replies.')
        agent.add_argument(
            '--add-p1-after-newln', type='bool', default=False, hidden=True,
            help='Add the other speaker token before the last newline in the '
                 'input instead of at the beginning of the input. this is '
                 'useful for tasks that include some kind of context before '
                 'the actual utterance (e.g. squad, babi, personachat).')
        # GPU arguments
        # these gpu options are all mutually exclusive, and should error if the
        # user tries to present multiple of them
        gpugroup = agent.add_mutually_exclusive_group()
        gpugroup.add_argument(
            '-gpu', '--gpu', type=int, default=-1, help='which GPU to use')
        gpugroup.add_argument(
            '--no-cuda', default=False, action='store_true', dest='no_cuda',
            help='disable GPUs even if available. otherwise, will use GPUs if '
                 'available on the device.')
                                   
                           
        # my arguments
        agent.add_argument('-pfile', '--num-segment-probs-file', type=str, default=None, 
            help="File that has the probabilities of number of segments to return for a "
            "given response turn")
        agent.add_argument('-lf', '--fit-lambda-file', type=str, default=None,
            help='file where fit lambda is stored')
        agent.add_argument('-lamb', '--lambda-weight', type=float, default=None,
            help='parameter that smooths between speaker/response models')
        agent.add_argument('-trlamb', '--is-training-lambda', type='bool', default=False,
            help='indicates if we should be fitting lambda.')
        agent.add_argument('-rk', '--rank-candidates', action='store_true',
            help='Rank candidates from file')
        agent.add_argument('-lmrnn', '--lm-rnn', type=str, default='GRU',
            help='RNN type for language model')
        agent.add_argument('-s2srnn', '--seq2seq-rnn', type=str, default='lstm',
            help='RNN type for seq2seq')
        agent.add_argument('-lmmf', '--lm-model-file', type=str, default=None,
            help='file for pre-trained lm')
        agent.add_argument('-s2smf', '--seq2seq-model-file', type=str, default=None,
            help='file for pre-trained seq2seq')
        
        # this should be somewhere else...
        agent.add_argument('-hs', '--history-size', type=float, default=0,
            help='???')
            
        agent.add_argument(
            '-tr', '--truncate', default=-1, type=int,
            help='Truncate input lengths to increase speed / use less memory.') # from TorchAgent
        agent.add_argument(
            '-fcp', '--fixed-candidates-path', type=str,
            help='A text file of fixed candidates to use for all examples, one '
                 'candidate per line') # from TorchRankerAgent
        agent.add_argument(
            '--fixed-candidate-vecs', type=str, default='reuse',
            help="One of 'reuse', 'replace', or a path to a file with vectors "
                 "corresponding to the candidates at --fixed-candidates-path. "
                 "The default path is a /path/to/model-file.<cands_name>, where "
                 "<cands_name> is the name of the file (not the full path) passed by "
                 "the flag --fixed-candidates-path. By default, this file is created "
                 "once and reused. To replace it, use the 'replace' option.") # from TorchRankerAgent
        agent.add_argument('-lmo', '--language-model-only', action='store_true', default=False,
                           help='Only consider language model: should reduce to language_model agent')
        agent.add_argument('-s2so', '--seq2seq-model-only', action='store_true', default=False,
                           help='Only consider seq2seq model: should reduce to seq2seq agent')

#         super(SmoothedDecoderAgent, cls).add_cmdline_args(argparser)
        SmoothedDecoderAgent.dictionary_class().add_cmdline_args(argparser)
        return agent        
        
        

    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)

        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        if opt.get('numthreads', 1) > 1:
            torch.set_num_threads(1)
        self.id = 'SmoothedDecoder'
        
        
        
        self.opt = opt
        lm_opt = json.load(open(opt['lm_model_file']+'.opt','r'))
        s2s_opt = json.load(open(opt['seq2seq_model_file']+'.opt', 'r'))
        self.replies = {}
        
        
        if not opt.get('language_model_only'):
#             s2s_opt['rnn_class'] = opt['seq2seq_rnn']
            s2s_opt['model_file'] = opt['seq2seq_model_file']
            
            self.s2s_agent = Seq2seqAgent(s2s_opt)
            self.s2s_agent.opt = s2s_opt
            
            init_model = None
            if not shared:  # only do this on first setup
                # first check load path in case we need to override paths
                if opt.get('init_model') and os.path.isfile(opt['init_model']):
                    # check first for 'init_model' for loading model from file
                    init_model = opt['init_model']
                
                if opt.get('model_file') and os.path.isfile(opt['model_file']):
                    # next check for 'model_file', this would override init_model
                    init_model = opt['model_file']

                if init_model is not None:
                    # if we are loading a model, should load its dict too
                    if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                        opt['dict_file'] = init_model + '.dict'
                    
            if init_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]'
                      ''.format(init_model))
                s2s_states = self.s2s_agent.load(init_model)
            
                        
        if not opt.get('seq2seq_model_only'):
#             lm_opt['rnn_class'] = opt['lm_rnn']
            lm_opt['model_file'] = opt['lm_model_file']
            self.lm_agent = LanguageModelRetrieverAgent(lm_opt)
            self.lm_agent.opt = lm_opt
        
        # Store a easily accessible copy of the dict, 
        # which should be shared between the two agents. 
        self.dict = self.s2s_agent.dict
        self.END_IDX = self.s2s_agent.END_IDX
        self.START_IDX = self.s2s_agent.START_IDX
        
        
        # Create model to optimize linear combination
        self.combo_model = LinearComboModel()
        self.combo_model.double()
#         self.combo_model.cuda()
        if opt['is_training_lambda']:
            self.combo_criterion = nn.CrossEntropyLoss()
            self.combo_optimizer = torch.optim.SGD(self.combo_model.parameters(), lr=0.01)
            
            #  Make sure the LM knows it should only be used in test mode
            self.lm_agent.opt['is_training_lambda'] = True
                    
        else:
            # load fit lambda
            val = json.load(open(opt['fit_lambda_file']+'.lamb','r'))
            self.lamb = self.combo_model.set_lambda(val['lambda'])
            
                    
        
        # load candidates for retrieval into self.fixed_candidates, self.fixed_candidate_vecs
        self.set_fixed_candidates(shared)
        self.cands_tensor, _ = padded_tensor(
                    self.fixed_candidate_vecs, self.s2s_agent.NULL_IDX, self.use_cuda
                )
        self.fixed_candidate_masks = [] # indicates tokens where padded.
        for i in range(self.fixed_candidate_vecs.size(0)): 
            self.fixed_candidate_masks.append(
                    self.fixed_candidate_vecs[i,:] != self.s2s_agent.NULL_IDX
                    )
        
        print('### CAND PATH ####', opt['fixed_candidates_path'], self.fixed_candidate_vecs.size())
        print('loaded fixed candidates!')
        
        if opt['num_segment_probs_file']:
            self.num_seg_probs = np.loadtxt(opt['num_segment_probs_file'])
            

            
    def observe(self, observation):
        self.observation = observation
#         print('# OBSERVATION #')
#         print(observation)
        
        if not self.opt.get('seq2seq_model_only'):
            lm_out = self.lm_agent.observe(observation)
#             print('### LM OBSERVATION ###')
#             print(lm_out)
#             print(self.lm_agent.observation)
#             print(self.lm_agent.next_observe)
            
        if not self.opt.get('language_model_only'):
            s2s_out = self.s2s_agent.observe(observation.copy())
#             print('### s2s_out ###', s2s_out)
#             print('### S2S OBSERVATION ###')
#             print(s2s_out)
#             print(self.s2s_agent.observation)
        
        return (lm_out, s2s_out)
    
            
    def act(self):
        
        return self.batch_act([self.observation])[0]
        
#         # call batch_act with this batch of one
#         if self.opt.get('seq2seq_model_only'):
#             return self.s2s_agent.batch_act([self.observation])[0]
#             
#         elif self.opt.get('language_model_only'):
#             return self.lm_agent.batch_act([self.observation])[0]
#         
#         
#         
#     def batch_act(self):
#         
#         if self.opt.get('seq2seq_model_only'):
#             return self.s2s_agent.batch_act([self.observation])
#             
#         elif self.opt.get('language_model_only'):
#             return self.lm_agent.batch_act([self.observation])
    
    
    def batch_act(self, observations):
        
        """ Modified from TorchAgent:
        Process a batch of observations (batchsize list of message dicts).
        These observations have been preprocessed by the observe method.
        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        batch_size = len(observations)
        
        # initialize a list of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        # check if there are any labels available, if so we will train on them
#         self.is_training = any('labels' in obs for obs in observations)
        self.is_training = False
        
        
        # create a batch for response model
        batch = self.s2s_agent.batchify([self.s2s_agent.observation])
        
        # create a batch for speaker model
        data_list, targets_list, labels, valid_inds, y_lens = self.lm_agent.vectorize(
                [self.lm_agent.observation], self.opt['seq_len'], self.is_training
            )
        
        
        
        if self.is_training:
            # output = self.train_step(batch)
            print('This class should only be used for evaluation')
            print('Resetting is_training variable.')
            self.is_training = False

        output = self.eval_step(batch, data_list)
        
        
        if output is None:
            self.replies['batch_reply'] = None
            return batch_reply

        self.match_batch(batch_reply, batch.valid_indices, output)
        self.replies['batch_reply'] = batch_reply
        self._save_history(observations, batch_reply)  # save model predictions
        
        return batch_reply
        
        
        

    def load(self, path):
        """Return opt and model states."""
        print("""Return opt and model states. for SMOOTHED_DECODER""")
#         states = torch.load(path, map_location=lambda cpu, _: cpu)
#         # set loaded states if applicable
#         self.model.load_state_dict(states['model'])
#         if 'longest_label' in states:
#             self.model.longest_label = states['longest_label']
#         return states
        
        s2s_states = load_seq2seq(self, path)
        
        return s2s_states

        
    
    # Load seq2seq
    def load_seq2seq(self, path):
        """Return opt and model states."""
        print("""Return opt and model states for s2s model!""")
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        # set loaded states if applicable
        self.s2s_agent.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.s2s_agent.model.longest_label = states['longest_label']
        return states
        
        
        
            
        
    # Load LM    
    def load_lm_opt(self, path):
        """Return opt, states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        return states['opt']
        

    def load_lm(self, path):
        """Load model states."""
        if os.path.isfile(path):
            # load model parameters if available
            print('[ Loading existing model params from {} ]'.format(path))
            self.states = torch.load(path, map_location=lambda cpu, _: cpu)
            self.model.load_state_dict(self.states['model'])
            
            
            
    
    ######################################################        
    ##### Retrieve candidates for evaluation/testing #####
    ######################################################
    
    def eval_step(self, batch, data_list):
        print("""Evaluate a single batch of examples.""")
        
        
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            print('batch.text_vec is None')
            return
        bsz = batch.text_vec.size(0)
        self.s2s_agent.model.eval()
        cand_scores = None
        
        text = []
        lamb = torch.tensor(self.opt['lambda_weight'], dtype=torch.double)
        
#         if batch.label_vec is not None:
#             # calculate loss on targets with teacher forcing
# #             f_scores, f_preds, _ = self.model(batch.text_vec, batch.label_vec)
# #             score_view = f_scores.view(-1, f_scores.size(-1))
# #             loss = self.criterion(score_view, batch.label_vec.view(-1))
# 
#             # save loss to metrics
#             notnull = batch.label_vec.ne(self.NULL_IDX)
#             target_tokens = notnull.long().sum().item()
#             correct = ((batch.label_vec == f_preds) * notnull).sum().item()
# 
#             self.metrics['correct_tokens'] += correct
#             self.metrics['loss'] += loss.item()
#             self.metrics['num_tokens'] += target_tokens


        cand_choices = None
        if self.opt['rank_candidates']:
            
            cand_choices = []
            num_cands = self.fixed_candidate_vecs.size(0)
            
            # initialize S2S hidden
            encoder_states = self.s2s_agent.model.encoder(batch.text_vec)
            
            # initialize LM hidden
            lm_hidden = self.lm_agent.model.init_hidden(bsz)
            
            
            for i in range(bsz):
                
                ### get likelihood from seq2seq model/agent ###
                bt_size = 100
                
                with torch.no_grad():
                    enc = self.s2s_agent.model.reorder_encoder_states(encoder_states, [i] * bt_size)
                
                with torch.no_grad(): 
                    s2s_likelihood = self.get_s2s_lik(enc, bt_size)
                        
                print('#### done s2s likelihood')
                
                                
                ### Now get likelihood from LM ###
                with torch.no_grad():
                    lm_likelihood = self.get_lm_lik(data_list[i], lm_hidden)
                
                print('#### done lm likelihood')
                
                if self.opt['is_training_lambda']: 
                
                    # Find which candidate is the same as the training example
                    # the candidates are from the training data, so the 1-hot is just 
                    # which candidate vector is the same as the example target.
                    lc = self.fixed_candidate_vecs.size(1)
                    nc = self.fixed_candidate_vecs.size(0)
                    nz = lc - self.s2s_agent.observation['eval_labels_vec'].size(0)
                    
                    
                    if nz > 0:
                        vec = torch.cat((self.s2s_agent.observation['eval_labels_vec'], 
                                            torch.tensor([self.s2s_agent.NULL_IDX]*nz)))
                        vec_rep = vec.reshape(1, -1).repeat(nc, 1)
                    else:
                        vec_rep = self.s2s_agent.observation['eval_labels_vec'][:lc].reshape(1, -1).repeat(nc, 1)
                
                    if self.use_cuda:
                        difference = torch.abs(vec_rep.cuda() - self.fixed_candidate_vecs)
                        one_hot_ind = (difference.sum(dim=1) == 0).nonzero().cpu()
                    else:
                        difference = torch.abs(vec_rep - self.fixed_candidate_vecs)
                        one_hot_ind = (difference.sum(dim=1) == 0).nonzero()
                    
                                        
                    # Take linear combination of model likelihood scores
                    self.lm_likelihood = lm_likelihood.reshape(-1,1).data#.cuda()
                    self.s2s_likelihood = s2s_likelihood.reshape(-1,1).data#.cuda()
                    
                    # SGD update after each obs.
                    self.combo_optimizer.zero_grad()
                    
                    print('#### zero-ed grads')
                                                
                    # Forward to get the outputs
                    combo_outputs = self.combo_model(self.lm_likelihood, self.s2s_likelihood)
                    
                    print('#### combo-ed model')
                    
                    combo_outputs.cpu()
                    one_hot_ind.cpu()
                    
                    if one_hot_ind.size(0) > 1: 
                        print('#### ONE HOT SHOULD BE ONE ###')
                        print(one_hot_ind)
                        print(vec_rep)
                        print('fixed_vecs: ', self.fixed_candidate_vecs[one_hot_ind.reshape(-1), :])
                        print('eval_labels: ', self.s2s_agent.observation['eval_labels'])
                        for k in range(one_hot_ind.size(0)):
                            print(self.fixed_candidates[one_hot_ind[k,:]])
                            
                        # TODO: This has to be fixed. Shouldn't one hot be size = 1?
                        one_hot_ind = one_hot_ind[0,:]
                    
                    
                    # Calcuate loss # NOTE: Need to make this scan over multiple observations, if returned.
                    self.combo_loss = self.combo_criterion(combo_outputs.reshape(1, -1), 
                                                                        one_hot_ind.reshape(-1))
                    print('#### got combo loss')
                    self.combo_model.cpu()
                 
                    # Getting gradients from parameters
                    self.combo_loss.backward()
                    print('#### went backward')
                    
                    # Updating parameters
                    self.combo_optimizer.step()
                    print('#### stepped')
                    
                    # Print lamb value
                    print('### LAMBDA: ', self.combo_model.lamb.data.cpu().item())
                    
                    # Constrain weight to [0,1]
                    self.combo_model.project_param()
                    print('### CLAMP: ', self.combo_model.lamb.data.cpu().item())
                    
                    text.append('No response -- fitting lambda' )
                    
                else:
                    self.lm_likelihood = lm_likelihood.reshape(-1,1).data
                    self.s2s_likelihood = s2s_likelihood.reshape(-1,1).data
                    
                    with torch.no_grad():  
                        likelihood = self.combo_model(self.lm_likelihood, 
                                                        self.s2s_likelihood)
                    
                    
                    # Todo: need to add a while loop in here.
                    # Options are decode until endofmessage is top of LMR or until num
                    # where num comes from 1+argmax{numpy.random.multinomial(n, pvals)}
                    # and pvals = self.num_seg_probs
                    # inside loop, need to update the LMR history 
        
                    _, ordering = likelihood.sort(descending=True)
                    text.append(self.fixed_candidates[ordering[0]])
#                     cand_choices.append([self.fixed_candidates[o] for o in ordering])
        

#         print([(likelihood[o], self.fixed_candidates[o]) for o in ordering])
#         text = [self._v2t(p) for p in preds]
#         text = [self._v2t(t) for t in self.fixed_candidate_vecs[ordering[0]]]
#         text = [cand_choices[0][0],]
#         text = [self.fixed_candidates[ordering[0]],]
            
        if bsz > 1:
            print('Batches should be size 1... If not, must handle')
            import sys; sys.exit()
            
                    
        # TODO
        # overwrite lambda file after each update until we have a better way to do this: 
        with open(self.opt['fit_lambda_file'], 'w') as f:
            # print('### LAMBDA: ', self.combo_model.lamb.data.cpu().item())
            f.write(json.dumps({'lambda': self.combo_model.lamb.data.cpu().item()}))
            
            
        return Output(text, cand_choices) #, cand_likelihood=[likelihood[o] for o in ordering])



    def get_s2s_lik(enc, bt_size):    
        
        # Initialize        
        mask = (self.fixed_candidate_vecs != self.s2s_agent.NULL_IDX)
        s2s_likelihood = torch.empty(num_cands, dtype=torch.double)
        lik_ind = 0
        
        # First batch to initialize everything. 
        scores, _ = self.s2s_agent.model.decode_forced(enc, self.cands_tensor[:bt_size,:])
        log_probs_batch = F.log_softmax(scores, dim=2).cpu()
        
        for j in range(bt_size):
            num_tok = torch.arange(mask[lik_ind,:].sum())
            s2s_likelihood[lik_ind] = log_probs_batch[
                                        j, 
                                        num_tok, 
                                        self.fixed_candidate_vecs[
                                                                lik_ind, 
                                                                mask[lik_ind,:]
                                                                ]
                                        ].sum()
            lik_ind += 1
        
        
        # need to subtract 1 from top of range to ensure that the last  
        # batch will not be empty if multiple of bt_size
        for b in range(bt_size, self.cands_tensor.size(0)-1, bt_size):
        
            cands_batch = self.cands_tensor[b:(b+bt_size),:]
                
            if cands_batch.size(0) < bt_size:
                with torch.no_grad():
                    enc = self.s2s_agent.model.reorder_encoder_states(
                                        encoder_states, [i] * cands_batch.size(0)
                                        )       
            with torch.no_grad():
                scores, _ = self.s2s_agent.model.decode_forced(enc, cands_batch)
            
                
            log_probs_batch = F.log_softmax(scores, dim=2)
            
            
            for j in range(cands_batch.size(0)):
                num_tok = torch.arange(mask[lik_ind,:].sum())
                s2s_likelihood[lik_ind] = log_probs_batch[
                                            j, 
                                            num_tok, 
                                            self.fixed_candidate_vecs[
                                                                    lik_ind, 
                                                                    mask[lik_ind,:]
                                                                    ]
                                            ].sum()
                lik_ind += 1
                
        return s2s_likelihood



    def get_lm_lik(self, data, hidden):
        
        """
        * Modified from get_target_loss in parlai/agents/language_model/language_model.py * 
           Calculates the liklihood with respect to the targets, token by token,
           where each output token is conditioned on either the input or the
           previous target token.
        """
#         loss = 0.0
        bsz = data.size(0)

        # during interactive mode, when no targets exist, we return 0
#         if targets is None:
#             return loss
            
        # feed in inputs without end token
        output, hidden = self.lm_agent.model(data.transpose(0, 1), hidden)
        self.lm_agent.hidden = self.lm_agent.repackage_hidden(hidden)
        
        # feed in end tokens
        output, hidden = self.lm_agent.model(Variable(self.lm_agent.ends[:bsz].view(1, bsz)),
                                                self.lm_agent.hidden)
        self.lm_agent.hidden = self.lm_agent.repackage_hidden(hidden)
        start_probs_vocab = F.log_softmax(output, dim=2)
        
        # store to reuse for each candidate
        self.lm_agent.input_hidden = self.lm_agent.repackage_hidden(hidden) 
        
        # used for computing loss, not likelihood
#         output_flat = output.view(-1, len(self.lm_agent.dict))
#         loss += self.criterion(output_flat, targets.select(1, 0).view(-1)).data
        
        cand_log_probs = torch.zeros([self.fixed_candidate_vecs.size(0)], dtype=torch.double)
        
        for c in range(self.fixed_candidate_vecs.size(0)):
            
            targets = self.fixed_candidate_vecs[c, self.fixed_candidate_masks[c]].reshape(1,-1)
            
            # accumulate for first token (after stop token)
            cand_log_probs[c] += start_probs_vocab[0, 0, targets[0, 0]]
            
            # reset to hidden after encoding
            self.lm_agent.hidden = self.lm_agent.repackage_hidden(self.lm_agent.input_hidden)
            
            for i in range(1, targets.size(1)):
                output, hidden = self.lm_agent.model(
                    targets.select(1, i - 1).view(1, bsz),
                    self.lm_agent.hidden,
                    no_pack=True
                )
                self.lm_agent.hidden = self.lm_agent.repackage_hidden(hidden)
                
                log_probs_vocab = F.log_softmax(output, dim=2)
                cand_log_probs[c] += log_probs_vocab[0, 0, targets[0, i]]
                
                # this was use for computing loss, not likelihood 
                # output_flat = output.view(-1, len(self.dict))
                # loss += self.criterion(output_flat, targets.select(1, i).view(-1)).data

        return cand_log_probs
        
        

    def _v2t(self, vec):
        # from TorchAgent
        
        """Convert token indices to string of tokens."""
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.s2s_agent.END_IDX:
                break
            new_vec.append(i)
        return self.s2s_agent.dict.vec2txt(new_vec)
        
        
        
    def _save_history(self, observations, replies):
    
        # from TorchAgent
        
        """Save the model replies to the history."""
        # make sure data structure is set up
        if 'predictions' not in self.replies:
            self.replies['predictions'] = {}
        if 'episode_ends' not in self.replies:
            self.replies['episode_ends'] = {}
        # shorthand
        preds = self.replies['predictions']
        ends = self.replies['episode_ends']
        for i, obs in enumerate(observations):
            # iterate through batch, saving replies
            if i not in preds:
                preds[i] = []
            if ends.get(i):
                # check whether *last* example was the end of an episode
                preds[i].clear()
            ends[i] = obs.get('episode_done', True)
            preds[i].append(replies[i].get('text'))
            
                    
    def match_batch(self, batch_reply, valid_inds, output=None):
        
        # from TorchAgent
        
        """Match sub-batch of predictions to the original batch indices.
        Batches may be only partially filled (i.e when completing the remainder
        at the end of the validation or test set), or we may want to sort by
        e.g the length of the input sequences if using pack_padded_sequence.
        This matches rows back with their original row in the batch for
        calculating metrics like accuracy.
        If output is None (model choosing not to provide any predictions), we
        will just return the batch of replies.
        Otherwise, output should be a parlai.core.torch_agent.Output object.
        This is a namedtuple, which can provide text predictions and/or
        text_candidates predictions. If you would like to map additional
        fields into the batch_reply, you can override this method as well as
        providing your own namedtuple with additional fields.
        :param batch_reply: Full-batchsize list of message dictionaries to put
            responses into.
        :param valid_inds: Original indices of the predictions.
        :param output: Output namedtuple which contains sub-batchsize list of
            text outputs from model. May be None (default) if model chooses not
            to answer. This method will check for ``text`` and
            ``text_candidates`` fields.
        """
        if output is None:
            return batch_reply
        if output.text is not None:
            for i, response in zip(valid_inds, output.text):
                batch_reply[i]['text'] = response
        if output.text_candidates is not None:
            for i, cands in zip(valid_inds, output.text_candidates):
                batch_reply[i]['text_candidates'] = cands


#     def eval_step(self, batch):
#         """Evaluate a single batch of examples."""
#         if batch.text_vec is None:
#             return
#         bsz = batch.text_vec.size(0)
#         self.model.eval()
#         cand_scores = None
# 
#         if self.skip_generation:
#             warn_once(
#                 "--skip-generation does not produce accurate metrics beyond ppl",
#                 RuntimeWarning
#             )
#             logits, preds, _ = self.model(batch.text_vec, batch.label_vec)
#         elif self.beam_size == 1:
#             # greedy decode
#             logits, preds, _ = self.model(batch.text_vec)
#         elif self.beam_size > 1:
#             out = self.beam_search(
#                 self.model,
#                 batch,
#                 self.beam_size,
#                 start=self.START_IDX,
#                 end=self.END_IDX,
#                 pad=self.NULL_IDX,
#                 min_length=self.beam_min_length,
#                 min_n_best=self.beam_min_n_best,
#                 block_ngram=self.beam_block_ngram
#             )
#             beam_preds_scores, _, beams = out
#             preds, scores = zip(*beam_preds_scores)
# 
#             if self.beam_dot_log is True:
#                 self._write_beam_dots(batch.text_vec, beams)
# 
#         if batch.label_vec is not None:
#             # calculate loss on targets with teacher forcing
#             f_scores, f_preds, _ = self.model(batch.text_vec, batch.label_vec)
#             score_view = f_scores.view(-1, f_scores.size(-1))
#             loss = self.criterion(score_view, batch.label_vec.view(-1))
#             # save loss to metrics
#             notnull = batch.label_vec.ne(self.NULL_IDX)
#             target_tokens = notnull.long().sum().item()
#             correct = ((batch.label_vec == f_preds) * notnull).sum().item()
#             self.metrics['correct_tokens'] += correct
#             self.metrics['loss'] += loss.item()
#             self.metrics['num_tokens'] += target_tokens
# 
#         cand_choices = None
#         # TODO: abstract out the scoring here
#         if self.rank_candidates:
#             # compute roughly ppl to rank candidates
#             cand_choices = []
#             encoder_states = self.model.encoder(batch.text_vec)
#             for i in range(bsz):
#                 num_cands = len(batch.candidate_vecs[i])
#                 enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
#                 cands, _ = padded_tensor(
#                     batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
#                 )
#                 scores, _ = self.model.decode_forced(enc, cands)
#                 cand_losses = F.cross_entropy(
#                     scores.view(num_cands * cands.size(1), -1),
#                     cands.view(-1),
#                     reduction='none',
#                 ).view(num_cands, cands.size(1))
#                 # now cand_losses is cands x seqlen size, but we still need to
#                 # check padding and such
#                 mask = (cands != self.NULL_IDX).float()
#                 cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
#                 _, ordering = cand_scores.sort()
#                 cand_choices.append([batch.candidates[i][o] for o in ordering])
# 
#         text = [self._v2t(p) for p in preds]
#         return Output(text, cand_choices)
        
        
        
                    
            
            
    #######################################################        
    ##### Loading fixed candidates for decode-scoring #####
    #######################################################
    
    def set_fixed_candidates(self, shared):
    
        # modified from TorchRankerAgent
        
        """Load a set of fixed candidates and their vectors (or vectorize them here)
        self.fixed_candidates will contain a [num_cands] list of strings
        self.fixed_candidate_vecs will contain a [num_cands, seq_len] LongTensor
        See the note on the --fixed-candidate-vecs flag for an explanation of the
        'reuse', 'replace', or path options.
        Note: TorchRankerAgent by default converts candidates to vectors by vectorizing
        in the common sense (i.e., replacing each token with its index in the
        dictionary). If a child model wants to additionally perform encoding, it can
        overwrite the vectorize_fixed_candidates() method to produce encoded vectors
        instead of just vectorized ones.
        """
        
        if shared:
            self.fixed_candidates = shared['fixed_candidates']
            self.fixed_candidate_vecs = shared['fixed_candidate_vecs']
        else:
            opt = self.opt
            cand_path = opt['fixed_candidates_path']
            # if ('fixed' in (opt['candidates'], opt['eval_candidates']) and
#                     cand_path):
            if cand_path:

                # Load candidates
                print("[ Loading fixed candidate set from {} ]".format(cand_path))
                with open(cand_path, 'r') as f:
                    cands = [line.strip() for line in f.readlines()]

                # Load or create candidate vectors
                if os.path.isfile(opt['fixed_candidate_vecs']):
                    vecs_path = opt['fixed_candidate_vecs']
                    vecs = self.load_candidate_vecs(vecs_path)
                else:
                    setting = opt['fixed_candidate_vecs']
                    model_dir, model_file = os.path.split(self.s2s_agent.opt['model_file'])
                    model_name = os.path.splitext(model_file)[0]
                    cands_name = os.path.splitext(os.path.basename(cand_path))[0]
                    vecs_path = os.path.join(
                        model_dir, '.'.join([model_name, cands_name]))
                    if setting == 'reuse' and os.path.isfile(vecs_path):
                        vecs = self.load_candidate_vecs(vecs_path)
                    else:  # setting == 'replace' OR generating for the first time
                        vecs = self.make_candidate_vecs(cands)
                        self.save_candidate_vecs(vecs, vecs_path)

                self.fixed_candidates = cands
                self.fixed_candidate_vecs = vecs
                
                
                if self.use_cuda:
                    self.fixed_candidate_vecs = self.fixed_candidate_vecs.cuda()
            else:
                self.fixed_candidates = None
                self.fixed_candidate_vecs = None


    def load_candidate_vecs(self, path):
        print("[ Loading fixed candidate set vectors from {} ]".format(path))
        return torch.load(path)



    def make_candidate_vecs(self, cands):
        cand_batches = [cands[i:i + 512] for i in range(0, len(cands), 512)]
        print("[ Vectorizing fixed candidates set from ({} batch(es) of up to 512) ]"
              "".format(len(cand_batches)))
        cand_vecs = []
        for batch in cand_batches:
            cand_vecs.extend(self.vectorize_fixed_candidates(batch))
        return padded_3d([cand_vecs]).squeeze(0)



    def save_candidate_vecs(self, vecs, path):
        print("[ Saving fixed candidate set vectors to {} ]".format(path))
        with open(path, 'wb') as f:
            torch.save(vecs, f)




    def vectorize_fixed_candidates(self, cands_batch):
        """Convert a batch of candidates from text to vectors
        :param cands_batch: a [batchsize] list of candidates (strings)
        :returns: a [num_cands] list of candidate vectors
        By default, candidates are simply vectorized (tokens replaced by token ids).
        A child class may choose to overwrite this method to perform vectorization as
        well as encoding if so desired.
        """
        # return [self._vectorize_text(cand, truncate=self.truncate, truncate_left=False)
#                 for cand in cands_batch]
        if self.opt['truncate'] == -1:
            return [self._vectorize_text(cand, truncate=None, 
                add_end=True, truncate_left=False) for cand in cands_batch]
        else:
            return [self._vectorize_text(cand, truncate=self.opt['truncate'], 
                add_end=True, truncate_left=False) for cand in cands_batch]
                
                
                
                
    def _vectorize_text(self, text, add_start=False, add_end=False,
                        truncate=None, truncate_left=True):
        # from TorchAgent
        
        """Return vector from text.
        :param text:          String to vectorize.
        :param add_start:     Add the start token to the front of the tensor.
        :param add_end:       Add the end token to the end of the tensor.
        :param truncate:      Truncate to this many tokens >= 0, or None.
        :param truncate_left: Truncate from the left side (keep the rightmost
                              tokens). You probably want this True for inputs,
                              False for targets.
        """
        vec = self.dict.txt2vec(text)
        if truncate is None or len(vec) + add_start + add_end < truncate:
            # simple: no truncation
            if add_start:
                vec.insert(0, self.START_IDX)
            if add_end:
                vec.append(self.END_IDX)
        elif truncate_left:
            # don't check add_start, we know are truncating it
            if add_end:
                # add the end token first
                vec.append(self.END_IDX)
            vec = vec[len(vec) - truncate:]
        else:
            # truncate from the right side
            # don't check add_end, we know we are truncating it
            vec = vec[:truncate - add_start]
            if add_start:
                # always keep the start token if it's there
                vec.insert(0, self.START_IDX)
        tensor = torch.LongTensor(vec)
        return tensor                
                
                