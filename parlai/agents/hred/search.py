#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import codecs

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state
logger = logging.getLogger(__name__)

def sample_wrapper(sample_logic):
    def sample_apply(*args, **kwargs):
        sampler = args[0]
        contexts = args[1]

        verbose = kwargs.get('verbose', False)

        if verbose:
            logger.info("Starting {} : {} start sequences in total".format(sampler.name, len(contexts)))
         
        context_samples = []
        context_costs = []

        # Start loop for each utterance
        for context_id, context_utterances in enumerate(contexts):
            if verbose:
                logger.info("Searching for {}".format(context_utterances))

            # Convert contextes into list of ids
            joined_context = []
            if len(context_utterances) == 0:
                joined_context = [sampler.model.eos_sym]
            else:
                utterance_ids = sampler.model.words_to_indices(context_utterances.split())
                # Add eos tokens
                if len(utterance_ids) > 0:
                    if not utterance_ids[0] == sampler.model.eos_sym:
                        utterance_ids = [sampler.model.eos_sym] + utterance_ids
                    if not utterance_ids[-1] == sampler.model.eos_sym:
                        utterance_ids += [sampler.model.eos_sym]
                
                else:
                    utterance_ids = [sampler.model.eos_sym]

                joined_context += utterance_ids

            samples, costs = sample_logic(sampler, joined_context, **kwargs) 
             
            # Convert back indices to list of words
            converted_samples = map(lambda sample : sampler.model.indices_to_words(sample, exclude_end_sym=kwargs.get('n_turns', 1) == 1), samples)
            # Join the list of words
            converted_samples = map(' '.join, converted_samples)

            if verbose:
                for i in range(len(converted_samples)):
                    #print "Samples {}: {}".format(costs[i], converted_samples[i].encode('utf-8'))
                    logger.info("Samples {}: {}".format(costs[i], converted_samples[i].encode('utf-8')))

            context_samples.append(converted_samples)
            context_costs.append(costs)

        return context_samples, context_costs
    return sample_apply

class Sampler(object):
    """
    An abstract sampler class 
    """
    def __init__(self, model):
        # Compile beam search
        self.name = 'Sampler'
        self.model = model
        self.compiled = False
        self.max_len = 160

    def compile(self):
        self.next_probs_predictor = self.model.build_next_probs_function()
        self.compute_encoding = self.model.build_encoder_function()

        if not self.model.reset_utterance_decoder_at_end_of_utterance:
            self.compute_decoder_encoding = self.model.build_decoder_encoding()

        self.compiled = True
    
    def select_next_words(self, next_probs, step_num, how_many):
        pass

    def count_n_turns(self, utterance):
        return len([w for w in utterance \
                    if w == self.model.eos_sym])

    @sample_wrapper
    def sample(self, *args, **kwargs):
        context = args[0]

        max_context_length = kwargs.get('max_context_length', 400)
        if len(context) > max_context_length:
            context = context[-max_context_length:]

        n_samples = kwargs.get('n_samples', 1)
        ignore_unk = kwargs.get('ignore_unk', True)
        min_length = kwargs.get('min_length', 1)
        max_length = kwargs.get('max_length', 30)
        beam_diversity = kwargs.get('beam_diversity', 1)
        normalize_by_length = kwargs.get('normalize_by_length', True)
        verbose = kwargs.get('verbose', False)
        n_turns = kwargs.get('n_turns', 1)

        if not self.compiled:
            self.compile()
        
        # Convert to matrix, each column is a context 
        # [[1,1,1],[4,4,4],[2,2,2]]
        context = numpy.repeat(numpy.array(context, dtype='int32')[:,None], 
                               n_samples, axis=1)

        if context[-1, 0] != self.model.eos_sym:
            raise Exception('Last token of context, when present,'
                            'should be the end of utterance: %d' % self.model.eos_sym)

        # Generate the reversed context
        reversed_context = self.model.reverse_utterances(context)

        if self.model.direct_connection_between_encoders_and_decoder:
            if self.model.bidirectional_utterance_encoder:
                dialog_enc_size = self.model.sdim+self.model.qdim_encoder*2
            else:
                dialog_enc_size = self.model.sdim+self.model.qdim_encoder
        else:
            dialog_enc_size = self.model.sdim

        prev_hs = numpy.zeros((n_samples, dialog_enc_size), dtype='float32')

        prev_hd = numpy.zeros((n_samples, self.model.utterance_decoder.complete_hidden_state_size), dtype='float32')

        if not self.model.reset_utterance_decoder_at_end_of_utterance:
            assert self.model.bs >= context.shape[1]
            enlarged_context = numpy.zeros((context.shape[0], self.model.bs), dtype='int32')
            enlarged_context[:, 0:context.shape[1]] = context[:]
            enlarged_reversed_context = numpy.zeros((context.shape[0], self.model.bs), dtype='int32')
            enlarged_reversed_context[:, 0:context.shape[1]] = reversed_context[:]

            ran_gaussian_vector = self.model.rng.normal(size=(context.shape[0],n_samples,self.model.latent_gaussian_per_utterance_dim)).astype('float32')
            ran_uniform_vector = self.model.rng.uniform(low=0.0, high=1.0, size=(context.shape[0],n_samples,self.model.latent_piecewise_per_utterance_dim)).astype('float32')

            zero_mask = numpy.zeros((context.shape[0], self.model.bs), dtype='float32')
            zero_vector = numpy.zeros((self.model.bs), dtype='float32')
            ones_mask = numpy.zeros((context.shape[0], self.model.bs), dtype='float32')

            # Computes new utterance decoder hidden states (including intermediate utterance encoder and dialogue encoder hidden states)
            new_hd = self.compute_decoder_encoding(enlarged_context, enlarged_reversed_context, self.max_len, zero_mask, zero_vector, ran_gaussian_vector, ran_uniform_vector, ones_mask)


            prev_hd[:] = new_hd[0][-1][0:context.shape[1], :]

        fin_gen = []
        fin_costs = []
         
        gen = [[] for i in range(n_samples)]
        costs = [0. for i in range(n_samples)]
        beam_empty = False

        # Compute random vector as additional input
        ran_gaussian_vectors = self.model.rng.normal(size=(n_samples,self.model.latent_gaussian_per_utterance_dim)).astype('float32')
        ran_uniform_vectors = self.model.rng.uniform(low=0.0, high=1.0, size=(n_samples,self.model.latent_piecewise_per_utterance_dim)).astype('float32')

        # HACK
        #ran_uniform_vectors = numpy.greater(ran_uniform_vectors, 0.5).astype('float32')


        for k in range(max_length):
            if len(fin_gen) >= n_samples or beam_empty:
                break
             
            if verbose:
                logger.info("{} : sampling step {}, beams alive {}".format(self.name, k, len(gen)))
             
            # Here we aggregate the context and recompute the hidden state
            # at both session level and query level.
            # Stack only when we sampled something
            if k > 0:
                context = numpy.vstack([context, \
                                        numpy.array(map(lambda g: g[-1], gen))]).astype('int32')
                reversed_context = numpy.copy(context)
                for idx in range(context.shape[1]):
                    eos_indices = numpy.where(context[:, idx] == self.model.eos_sym)[0]
                    prev_eos_index = -1
                    for eos_index in eos_indices:
                        reversed_context[(prev_eos_index+2):eos_index, idx] = (reversed_context[(prev_eos_index+2):eos_index, idx])[::-1]
                        prev_eos_index = eos_index

            prev_words = context[-1, :]
           
            # Recompute encoder states, hs and random variables 
            # only for those particular utterances that meet the end-of-utterance token
            indx_update_hs = [num for num, prev_word in enumerate(prev_words)
                                if prev_word == self.model.eos_sym]

            if len(indx_update_hs):
                encoder_states = self.compute_encoding(context[:, indx_update_hs], reversed_context[:, indx_update_hs], self.max_len)
                prev_hs[indx_update_hs] = encoder_states[1][-1]
                ran_gaussian_vectors[indx_update_hs,:] = self.model.rng.normal(size=(len(indx_update_hs),self.model.latent_gaussian_per_utterance_dim)).astype('float32')
                ran_uniform_vectors[indx_update_hs,:] = self.model.rng.uniform(low=0.0, high=1.0, size=(len(indx_update_hs),self.model.latent_piecewise_per_utterance_dim)).astype('float32')


                # HACK
                #ran_uniform_vectors = numpy.greater(ran_uniform_vectors, 0.5).astype('float32')

            # ... done
            next_probs, new_hd = self.next_probs_predictor(prev_hs, prev_hd, prev_words, context, ran_gaussian_vectors, ran_uniform_vectors)



            assert next_probs.shape[1] == self.model.idim
            
            # Adjust log probs according to search restrictions
            if ignore_unk:
                next_probs[:, self.model.unk_sym] = 0
            if k <= min_length:
                next_probs[:, self.model.eos_sym] = 0
                next_probs[:, self.model.eod_sym] = 0
             
            # Update costs 
            next_costs = numpy.array(costs)[:, None] - numpy.log(next_probs)

            # Select next words here
            (beam_indx, word_indx), costs = self.select_next_words(next_costs, next_probs, k, n_samples)
            
            # Update the stacks
            new_gen = [] 
            new_costs = []
            new_sources = []

            for num, (beam_ind, word_ind, cost) in enumerate(zip(beam_indx, word_indx, costs)):
                if len(new_gen) > n_samples:
                    break

                hypothesis = gen[beam_ind] + [word_ind]
                 
                # End of utterance has been detected
                n_turns_hypothesis = self.count_n_turns(hypothesis)
                if n_turns_hypothesis == n_turns:
                    if verbose:
                        logger.debug("adding utterance {} from beam {}".format(hypothesis, beam_ind))

                    # We finished sampling
                    fin_gen.append(hypothesis)
                    fin_costs.append(cost)
                elif self.model.eod_sym in hypothesis: # End of dialogue detected
                    new_hypothesis = []
                    for wrd in hypothesis:
                        new_hypothesis += [wrd]
                        if wrd == self.model.eod_sym:
                            break
                    hypothesis = new_hypothesis

                    if verbose:
                        logger.debug("adding utterance {} from beam {}".format(hypothesis, beam_ind))

                    # We finished sampling
                    fin_gen.append(hypothesis)
                    fin_costs.append(cost)
                else:
                    # Hypothesis recombination
                    # TODO: pick the one with lowest cost 
                    has_similar = False
                    if self.hyp_rec > 0:
                        has_similar = len([g for g in new_gen if \
                            g[-self.hyp_rec:] == hypothesis[-self.hyp_rec:]]) != 0
                    
                    if not has_similar:
                        new_sources.append(beam_ind)
                        new_gen.append(hypothesis)
                        new_costs.append(cost)
            
            if verbose:
                for gen in new_gen:
                    logger.debug("partial -> {}".format(' '.join(self.model.indices_to_words(gen))))

            prev_hd = new_hd[new_sources]
            prev_hs = prev_hs[new_sources]
            ran_gaussian_vectors = ran_gaussian_vectors[new_sources,:]
            ran_uniform_vectors = ran_uniform_vectors[new_sources,:]
            context = context[:, new_sources]
            reversed_context = reversed_context[:, new_sources]
            gen = new_gen
            costs = new_costs
            beam_empty = len(gen) == 0

        # If we have not sampled anything
        # then force include stuff
        if len(fin_gen) == 0:
            fin_gen = gen 
            fin_costs = costs 
         
        # Normalize costs
        if normalize_by_length:
            fin_costs = [(fin_costs[num]/len(fin_gen[num])) \
                         for num in range(len(fin_gen))]

        fin_gen = numpy.array(fin_gen)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        return fin_gen[:n_samples], fin_costs[:n_samples] 

class RandomSampler(Sampler):
    def __init__(self, model):
        Sampler.__init__(self, model)
        self.name = 'RandomSampler'
        self.hyp_rec = 0

    def select_next_words(self, next_costs, next_probs, step_num, how_many):
        # Choice is complaining
        next_probs = next_probs.astype("float64") 
        word_indx = numpy.array([self.model.rng.choice(self.model.idim, p = x/numpy.sum(x))
                                    for x in next_probs], dtype='int32')
        beam_indx = range(next_probs.shape[0])

        args = numpy.ravel_multi_index(numpy.array([beam_indx, word_indx]), next_costs.shape)
        return (beam_indx, word_indx), next_costs.flatten()[args]

class BeamSampler(Sampler):
    def __init__(self, model):
        Sampler.__init__(self, model)
        self.name = 'BeamSampler'
        self.hyp_rec = 3

    def select_next_words(self, next_costs, next_probs, step_num, how_many):
        # Pick only on the first line (for the beginning of sampling)
        # This will avoid duplicate <q> token.
        if step_num == 0:
            flat_next_costs = next_costs[:1, :].flatten()
        else:
            # Set the next cost to infinite for finished utterances (they will be replaced)
            # by other utterances in the beam
            flat_next_costs = next_costs.flatten()
         
        voc_size = next_costs.shape[1]
         
        args = numpy.argpartition(flat_next_costs, how_many)[:how_many]
        args = args[numpy.argsort(flat_next_costs[args])]
        
        return numpy.unravel_index(args, next_costs.shape), flat_next_costs[args]
        

