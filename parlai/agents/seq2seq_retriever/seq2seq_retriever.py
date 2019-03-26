

from parlai.core.agents import Agent
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from parlai.core.torch_agent import Batch, Output

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
        print('#### Init-ed my method!')
            
        """Set up model."""
        super().__init__(opt, shared)
        
        self.id = 'Seq2SeqRetriever'
        
        if shared and opt['swap_criterion_train_eval']:
            # set up shared properties
            self.train_criterion = shared['train_criterion']
            self.eval_criterion = shared['eval_criterion']
            
            
        
        
#     def compute_loss(self, batch, return_output=False):
#         print('#### Using my loss method!')
#         
#         ''' modified from parlai/core/torch_generator_agent.py 
#         This function overwrites parent class to compute idf-weighted loss 
#         if training and not weight the loss if validating. This encourages
#         more agressive updates to less frequent words during training, but 
#         training should stop when the text looks good over all vocabulary.'''
#         
#         
#         """
#         Computes and returns the loss for the given batch. Easily overridable for
#         customized loss functions.
#         If return_output is True, the full output from the call to self.model()
#         is also returned, via a (loss, model_output) pair.
#         """
#         
#         is_training = any('labels' in obs for obs in self.observation)  
#         
#         if batch.label_vec is None:
#             raise ValueError('Cannot compute loss without a label.')
#         model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
#         scores, preds, *_ = model_output
#         score_view = scores.view(-1, scores.size(-1))
#         
#         if self.opt['swap_criterion_train_eval']:
#         
#             if is_training: 
#                 print('Using training loss!')
#                 loss = self.train_criterion(score_view, batch.label_vec.view(-1))
#             else: 
#                 print('Using evaluation loss!')
#                 loss = self.eval_criterion(score_view, batch.label_vec.view(-1))
#                 
#         else: 
#             loss = self.criterion(score_view, batch.label_vec.view(-1))
#             
#         # save loss to metrics
#         notnull = batch.label_vec.ne(self.NULL_IDX)
#         target_tokens = notnull.long().sum().item()
#         correct = ((batch.label_vec == preds) * notnull).sum().item()
#         self.metrics['correct_tokens'] += correct
#         self.metrics['nll_loss'] += loss.item()
#         self.metrics['num_tokens'] += target_tokens
#         loss /= target_tokens  # average loss per token
#         if return_output:
#             return (loss, model_output)
#         else:
#             return loss      
            
                
    def share(self):
    
        shared = super().share()
        
        if self.opt['swap_criterion_train_eval']: 
            shared['train_criterion'] = self.train_criterion
            shared['eval_criterion'] = self.eval_criterion
        
        return shared

    
    def compute_criterion(self, score_view, label_view, is_training=False):
        
        if self.opt['swap_criterion_train_eval']:
            
            if is_training: 
                print('Using training loss!')
                loss = self.train_criterion(score_view, label_view)
            else: 
                print('Using evaluation loss!')
                loss = self.eval_criterion(score_view, label_view)
                
        else: 
            loss = self.criterion(score_view, label_view)
        
        return loss
            

    def build_criterion(self):
    
        if self.opt['weight_criterion_idf']:
            
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
                    
#         else: 
#         
#             # weight with 1/sqrt(freq)
#         
#             word_weights = torch.zeros(len(self.dict.freq.keys()))
#             for tok in self.dict.freq.keys(): 
#                 word_weights[self.dict.tok2ind[tok]] = 1./(float(self.dict.freq[tok]) + 1.)**.5
        
        
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
                
            self.criterion = 'dummy for sharing'
            
                    
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





    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        """Pre-initialize CUDA buffer by doing fake forward pass."""
        if self.use_cuda and (force or not hasattr(self, 'buffer_initialized')):
            try:
                dummy_xs = torch.ones(batchsize, maxlen).long().cuda()
                dummy_ys = torch.ones(batchsize, 2).long().cuda()
                scores, _, _ = self.model(dummy_xs, dummy_ys)
                # loss = self.criterion(
#                     scores.view(-1, scores.size(-1)), dummy_ys.view(-1)
#                 )
                
                # OAD: 
                score_view = scores.view(-1, scores.size(-1))
                loss = self.compute_criterion(score_view, dummy_ys.view(-1), is_training=False)
            
                loss.backward()
                self.buffer_initialized = True
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    m = ('CUDA OOM: Lower batch size (-bs) from {} or lower '
                         ' max sequence length (-tr) from {}'
                         ''.format(batchsize, maxlen))
                    raise RuntimeError(m)
                else:
                    raise e
                    
                    

    def train_step(self, batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        self._init_cuda_buffer(batchsize, self.truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            scores, preds, _ = self.model(batch.text_vec, batch.label_vec)
            score_view = scores.view(-1, scores.size(-1))
            # loss = self.criterion(score_view, batch.label_vec.view(-1))
            
            # OAD: 
            loss = self.compute_criterion(score_view, batch.label_vec.view(-1), 
                                            is_training=True)
            
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens
            loss /= target_tokens  # average loss per token
            loss.backward()
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

    def _write_beam_dots(self, text_vecs, beams):
        """Write the beam dot files to disk."""
        for i, b in enumerate(beams):
            dot_graph = b.get_beam_dot(dictionary=self.dict, n_best=3)
            image_name = self._v2t(text_vecs[i, -20:])
            image_name = image_name.replace(' ', '-').replace('__null__', '')
            dot_graph.write_png(
                os.path.join(self.beam_dot_dir, "{}.png".format(image_name))
            )

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        self.model.eval()
        cand_scores = None

        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
            logits, preds, _ = self.model(batch.text_vec, batch.label_vec)
        elif self.beam_size == 1:
            # greedy decode
            logits, preds, _ = self.model(batch.text_vec)
        elif self.beam_size > 1:
            out = self.beam_search(
                self.model,
                batch,
                self.beam_size,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=self.beam_min_n_best,
                block_ngram=self.beam_block_ngram
            )
            beam_preds_scores, _, beams = out
            preds, scores = zip(*beam_preds_scores)

            if self.beam_dot_log is True:
                self._write_beam_dots(batch.text_vec, beams)

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            f_scores, f_preds, _ = self.model(batch.text_vec, batch.label_vec)
            score_view = f_scores.view(-1, f_scores.size(-1))
#             loss = self.criterion(score_view, batch.label_vec.view(-1))
            
            # OAD: 
            loss = self.compute_criterion(score_view, batch.label_vec.view(-1), is_training=False)
            
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == f_preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens

        cand_choices = None
        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(batch.text_vec)
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = padded_tensor(
                    batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
                )
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = [self._v2t(p) for p in preds]
        return Output(text, cand_choices)      
            