

from parlai.core.agents import Agent
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent

from parlai.agents.seq2seq.modules import Seq2seq, opt_to_kwargs

from parlai.misc.idf_counter import IDFScorer


import torch
import torch.nn as nn



class Seq2seqRetrieverAgent(Seq2seqAgent):
    
    
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2SeqRetriever Arguments')
        
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
        
        agent.add_argument(
            '-wcidf', '--weight-criterion-idf', type='bool', default=False,
            help='Whether to weight the loss with the idf weights '
                '(must be pre-calculated)')
                
        super(Seq2seqRetrieverAgent, cls).add_cmdline_args(argparser)
        return agent


    def __init__(self, opt, shared=None):
        
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'Seq2SeqRetriever'
        
        


    def build_criterion(self):
    
    
        if self.opt['datatype'] == 'train': 
        
            if self.opt['weight_criterion_idf']:
            
                # Weight token importance with idf, if desired.
                word_weights = torch.zeros(len(self.dict.freq.keys()))
                        
                for tok in self.dict.freq.keys(): 
                    if self.dict.freq[tok] > 0: 
                        word_idf = torch.log(
                                        torch.tensor([float(self.dict.tot_doc) 
                                                            / float(self.dict.doc_freq[tok])]
                                                    )
                                            )
                        word_weights[self.dict.tok2ind[tok]] = word_idf
                    else: 
                        print(tok, self.dict.doc_freq[str(tok)], )
                    
            
#                 idf_scorer = IDFScorer(self.opt)
#                 min_idf = min(idf_scorer.vectorizer.idf_)
#                 word_weights = min_idf * torch.ones(len(self.dict.freq.keys()))
#                         
#                 for tok in self.dict.freq.keys(): 
#                 
#                     if tok != self.dict.null_token:
#                     
#                         try:
#                             word_idf = idf_scorer.vectorizer.idf_[idf_scorer.vectorizer.vocabulary_[tok]]
#                             word_weights[self.dict.tok2ind[tok]] = word_idf
#                         except: 
#                             if tok in [self.dict.start_token, 
#                                         self.dict.end_token, 
#                                         self.dict.unk_token]:
#                                 pass # leave set to minimum idf, as initialized.
#                             
#                             else: 
#                                 print('there is no idf for token: ', tok, ' type: ', type(tok))
        
            else: 
            
                # weight with 1/sqrt(freq)
            
                word_weights = torch.zeros(len(self.dict.freq.keys()))
                for tok in self.dict.freq.keys(): 
                    word_weights[self.dict.tok2ind[tok]] = 1./(float(self.dict.freq[tok]) + 1.)**.5
                
                
            # set up criteria
            if self.opt.get('numsoftmax', 1) > 1:
                self.criterion = nn.NLLLoss(
                    ignore_index=self.NULL_IDX, size_average=False, weight=word_weights)
            else:
                self.criterion = nn.CrossEntropyLoss(
                    ignore_index=self.NULL_IDX, size_average=False, weight=word_weights)
        
        else:
        
            # don't increase weights when decoding, only for training. 
            
            # set up criteria
            if self.opt.get('numsoftmax', 1) > 1:
                self.criterion = nn.NLLLoss(
                    ignore_index=self.NULL_IDX, size_average=False)
            else:
                self.criterion = nn.CrossEntropyLoss(
                    ignore_index=self.NULL_IDX, size_average=False)
                    

        if self.use_cuda:
            self.criterion.cuda()
            
            
            
            
            
            
            
            