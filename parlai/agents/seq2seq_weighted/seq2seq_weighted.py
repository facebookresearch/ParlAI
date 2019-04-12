

from parlai.core.agents import Agent
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from parlai.core.torch_agent import Batch, Output

from parlai.agents.seq2seq.modules import Seq2seq, opt_to_kwargs



import torch
import torch.nn as nn



class Seq2seqWeightedAgent(Seq2seqAgent):
    
    
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2SeqWeighted Arguments')
        
        agent.add_argument(
            '-swap', '--swap-criterion-train-eval', type='bool', default=False,
            help='Whether to swap the criterion between training and evaluation'
            'i.e., train with idf weighting, but eval without idf in criterion')
                
        super(Seq2seqWeightedAgent, cls).add_cmdline_args(argparser)
        return agent


    def __init__(self, opt, shared=None):
#         print('#### Init-ed my method!')
            
        """Set up model."""
        super().__init__(opt, shared)
        
        self.id = 'Seq2SeqWeighted'
        
        if shared:
            # set up shared properties
            self.train_criterion = shared['train_criterion']
            
            if opt['swap_criterion_train_eval']:
                self.eval_criterion = shared['eval_criterion']
            
            
        
        
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
        
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        
        if self.opt['swap_criterion_train_eval']:
        
            if self.is_training: 
                loss = self.train_criterion(score_view, batch.label_vec.view(-1))
            else:             
                loss = self.eval_criterion(score_view, batch.label_vec.view(-1))
                
        else: 
            loss = self.train_criterion(score_view, batch.label_vec.view(-1))
            
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
            
                
    def share(self):
    
        shared = super().share()
        shared['train_criterion'] = self.train_criterion
        if self.opt['swap_criterion_train_eval']: 
            shared['eval_criterion'] = self.eval_criterion
        
        return shared


    def build_criterion(self):
        
        print('Setting up idf weights!')
        
        # Weight token importance with idf, if desired.
        tot_doc = float(self.dict.tot_doc)
        # min_idf = torch.log(torch.tensor([ tot_doc/ (tot_doc - 1.)]))
        
        special_tokens = [self.dict.null_token, self.dict.start_token,
                            self.dict.end_token, self.dict.unk_token]
                            
        max_doc_freq = sorted([self.dict.doc_freq[t] 
                                for t in self.dict.doc_freq.keys() 
                                    if t not in special_tokens])[-1]
                                    
        min_idf = torch.log(torch.tensor([ tot_doc/ (max_doc_freq)]))
        word_weights = min_idf * torch.ones(len(self.dict.freq.keys()))
        
        for tok in self.dict.doc_freq.keys(): 
            if self.dict.doc_freq[tok] > 0: 
                word_idf = torch.log(
                                torch.tensor([tot_doc 
                                                / float(self.dict.doc_freq[tok])]
                                            )
                                    )
                word_weights[self.dict.tok2ind[tok]] = torch.max(torch.tensor([word_idf, min_idf])) 
            else: 
                print(tok, self.dict.doc_freq[str(tok)], )
        
        
        if self.opt['swap_criterion_train_eval']:
            
            print('Setting up two criteria!')
            
            # set up train criterion
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
                    
                    
            if self.use_cuda:
                self.train_criterion.cuda()
                self.eval_criterion.cuda()
            
                    
        else:
        
            # set up universal criterion
            if self.opt.get('numsoftmax', 1) > 1:
                self.train_criterion = nn.NLLLoss(
                    ignore_index=self.NULL_IDX, size_average=False, weight=word_weights)
            
            else:
                self.train_criterion = nn.CrossEntropyLoss(
                    ignore_index=self.NULL_IDX, size_average=False, weight=word_weights)

            
            if self.use_cuda:
                self.train_criterion.cuda()          
        
        self.criterion = 'dummy for sharing' 
                    
        
            
      
            
 
