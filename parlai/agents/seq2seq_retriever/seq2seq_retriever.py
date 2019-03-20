

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
                
        agent.add_argument(
            '-swap', '--swap-criterion-train-eval', type='bool', default=False,
            help='Whether to swap the criterion between training and evaluation'
            'i.e., train with idf weighting, but eval without idf in criterion')
                
        super(Seq2seqRetrieverAgent, cls).add_cmdline_args(argparser)
        return agent


    def __init__(self, opt, shared=None):
        
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'Seq2SeqRetriever'
        
        


    def build_criterion(self):
    
        if self.opt['weight_criterion_idf']:
        
            # Weight token importance with idf, if desired.
            tot_doc = float(self.dict.tot_doc)
            min_idf = torch.log(torch.tensor([ tot_doc/ (tot_doc - 1.)]))
            
            word_weights = min_idf * torch.ones(len(self.dict.freq.keys()))
            
            for tok in self.dict.freq.keys(): 
                if self.dict.freq[tok] > 0: 
                    word_idf = torch.log(
                                    torch.tensor([tot_doc 
                                                    / (1. + float(self.dict.doc_freq[tok]))]
                                                )
                                        )
                    word_weights[self.dict.tok2ind[tok]] = torch.max(torch.tensor([word_idf, min_idf])) 
                else: 
                    print(tok, self.dict.doc_freq[str(tok)], )
                    
        else: 
        
            # weight with 1/sqrt(freq)
        
            word_weights = torch.zeros(len(self.dict.freq.keys()))
            for tok in self.dict.freq.keys(): 
                word_weights[self.dict.tok2ind[tok]] = 1./(float(self.dict.freq[tok]) + 1.)**.5
        
        
        if self.opt['swap_criterion_train_eval']:
            
            # set up train criteria
            if self.opt.get('numsoftmax', 1) > 1:
                
                self.train_criterion = nn.NLLLoss(
                    ignore_index=self.NULL_IDX, size_average=False, weight=word_weights)
                self.eval_criterion = nn.NLLLoss(
                    ignore_index=self.NULL_IDX, size_average=False)
                    
            else:
                self.train_criterion = nn.CrossEntropyLoss(
                    ignore_index=self.NULL_IDX, size_average=False, weight=word_weights)
                self.eval_criterion = nn.CrossEntropyLoss(
                    ignore_index=self.NULL_IDX, size_average=False)
                    
                    
        else:
        
            # set up universal criterion
            if self.opt.get('numsoftmax', 1) > 1:
                self.criterion = nn.NLLLoss(
                    ignore_index=self.NULL_IDX, size_average=False, weight=word_weights)
            
            else:
                self.criterion = nn.CrossEntropyLoss(
                    ignore_index=self.NULL_IDX, size_average=False, weight=word_weights)
            
        
           
                    
        if self.use_cuda:
            self.criterion.cuda()
            
            
            
    
    def compute_loss(self, batch, return_output=False):
        
        ''' modified from parlai/core/torch_generator_agent.py 
        This function overwrites parent class to compute idf-weighted loss 
        if training and not weight the loss if validating. This encourages
        more agressive updates to less frequent words during training, but 
        training should stop when the text looks good over all vocabulary.'''
        
        
        """
        Computes and returns the loss for the given batch. Easily overridable for
        customized loss functions.
        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        
        is_training = any('labels' in obs for obs in self.observation)  
        
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        
        if self.opt['swap_criterion_train_eval']:
        
            if is_training: 
                loss = self.train_criterion(score_view, batch.label_vec.view(-1))
            else: 
                loss = self.eval_criterion(score_view, batch.label_vec.view(-1))
                
        else: 
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum().item()
        correct = ((batch.label_vec == preds) * notnull).sum().item()
        self.metrics['correct_tokens'] += correct
        self.metrics['nll_loss'] += loss.item()
        self.metrics['num_tokens'] += target_tokens
        loss /= target_tokens  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss            
            
 






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




            
            