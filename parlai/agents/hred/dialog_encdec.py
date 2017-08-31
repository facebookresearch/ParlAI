"""
Dialog hierarchical encoder-decoder code.
The code is inspired from nmt encdec code in groundhog
but we do not rely on groundhog infrastructure.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Iulian Vlad Serban")

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
logger = logging.getLogger(__name__)

from theano import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams
# Deprecated
#from theano.tensor.nnet.conv3d2d import *

from collections import OrderedDict

from model import *
from utils import *

import operator

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param


class EncoderDecoderBase():
    def __init__(self, state, rng, parent):
        self.rng = rng
        self.parent = parent
        
        self.state = state
        self.__dict__.update(state)
        
        self.dialogue_rec_activation = eval(self.dialogue_rec_activation)
        self.sent_rec_activation = eval(self.sent_rec_activation)
         
        self.params = []

class LinearCombination(EncoderDecoderBase):
    """
    This module computes a per-dimension weighted sum of two vectors x and y.
    The module can be extended, so that the weights of x and y depends on a conditioning vector (cond).
    """

    def init_params(self, cond_size, output_size, force_min_max_intervals, min_val, max_val):
        self.W = add_to_params(self.params, theano.shared(value=np.ones((output_size,), dtype='float32'), name='W_x'+self.name))

        self.force_min_max_intervals = force_min_max_intervals
        self.min_val = min_val
        self.max_val = max_val
        
    def build_output(self, cond, x, y):
        res = self.W*x + (np.float32(1.0) - self.W)*y

        if self.force_min_max_intervals:
            return T.clip(res, self.min_val, self.max_val)
        else:
            return res

    def __init__(self, state, cond_size, output_size, force_min_max_intervals, min_val, max_val, rng, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(cond_size, output_size, force_min_max_intervals, min_val, max_val)


class OneLayerMLP(EncoderDecoderBase):
    def init_params(self, inp_size, hidden_size, output_size):
        # First layer
        self.W1_in_act = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, inp_size, hidden_size), name='W1_in_'+self.name))
        self.b1_in_act = add_to_params(self.params, theano.shared(value=np.zeros((hidden_size,), dtype='float32'), name='b1_in_'+self.name))

        # First layer batch norm / layer norm parameters
        self.normop_in_act_h1_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((hidden_size,), dtype='float32'), name='normop_in_act_h1_gamma_'+self.name))
        self.normop_in_act_h1_mean = add_to_params(self.params, theano.shared(value=np.zeros((hidden_size,), dtype='float32'), name='normop_in_act_h1_mean_'+self.name))
        self.normop_in_act_h1_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((hidden_size,), dtype='float32'), name='normop_in_act_h1_var_'+self.name))

        # Output layer
        self.W2_in_act = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, hidden_size, output_size), name='W2_in_'+self.name))
        self.b2_in_act = add_to_params(self.params, theano.shared(value=np.zeros((output_size,), dtype='float32'), name='b2_in_'+self.name))

        # Output layer batch norm / layer norm parameters
        self.normop_in_act_h2_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((output_size,), dtype='float32'), name='normop_in_act_h2_gamma_'+self.name))
        self.normop_in_act_h2_mean = add_to_params(self.params, theano.shared(value=np.zeros((output_size,), dtype='float32'), name='normop_in_act_h2_mean_'+self.name))
        self.normop_in_act_h2_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((output_size,), dtype='float32'), name='normop_in_act_h2_var_'+self.name))

    def build_output(self, inp, bnmask):
        # Make sure bnmask is of type float32:
        if bnmask:
            bnmask = T.cast(bnmask, 'float32')

        # Execute normalization operator on inputs
        h_nonlinear_inp, h_nonlinear_inp_mean, h_nonlinear_inp_var = NormalizationOperator(self.normop_type, T.dot(inp, self.W1_in_act) + self.b1_in_act, self.normop_in_act_h1_gamma, bnmask, self.normop_in_act_h1_mean, self.normop_in_act_h1_var)

        # Compute hidden layer
        h = T.nnet.relu(h_nonlinear_inp)

        # Execute normalization operator on hidden layer
        output, output_mean, output_var = NormalizationOperator(self.normop_type, T.dot(h, self.W2_in_act) + self.b2_in_act, self.normop_in_act_h2_gamma, bnmask, self.normop_in_act_h2_mean, self.normop_in_act_h2_var)

        # Create batch norm updates
        updates = []
        if self.normop_type == 'BN':
            print(' Creating batch norm updates for OneLayerMLP (' + self.name + '):')
            vars_to_update = [self.normop_in_act_h1_mean, self.normop_in_act_h1_var]
            vars_estimates = [h_nonlinear_inp_mean, h_nonlinear_inp_var, output_mean, output_var]

            assert len(vars_estimates) == len(vars_to_update)

            for i in range(len(vars_estimates)):
                print('     ', vars_to_update[i])
                new_value = self.normop_moving_average_const*vars_to_update[i] \
                            + (1.0 - self.normop_moving_average_const)*vars_estimates[i]
                updates.append((vars_to_update[i], new_value))

        return output, updates


    def __init__(self, state, rng, inp_size, hidden_size, output_size, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(inp_size, hidden_size, output_size)


class TwoLayerMLP(EncoderDecoderBase):
    def init_params(self, inp_size, hidden_size, output_size):
        # First layer
        self.W1_in_tanh = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, inp_size, hidden_size), name='W1_in_'+self.name))
        self.b1_in_tanh = add_to_params(self.params, theano.shared(value=np.zeros((hidden_size,), dtype='float32'), name='b1_in_'+self.name))
        self.W1_in_skip = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, inp_size, hidden_size), name='W1_in_skip_'+self.name))
        self.b1_in_skip = add_to_params(self.params, theano.shared(value=np.zeros((hidden_size,), dtype='float32'), name='b1_in_skip_'+self.name))

        # First layer batch norm / layer norm parameters
        self.normop_in_tanh_h1_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((hidden_size,), dtype='float32'), name='normop_in_tanh_h1_gamma_'+self.name))
        self.normop_in_tanh_h1_mean = add_to_params(self.params, theano.shared(value=np.zeros((hidden_size,), dtype='float32'), name='normop_in_tanh_h1_mean_'+self.name))
        self.normop_in_tanh_h1_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((hidden_size,), dtype='float32'), name='normop_in_tanh_h1_var_'+self.name))

        self.normop_in_skip_h1_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((hidden_size,), dtype='float32'), name='normop_in_skip_h1_gamma_'+self.name))
        self.normop_in_skip_h1_mean = add_to_params(self.params, theano.shared(value=np.zeros((hidden_size,), dtype='float32'), name='normop_in_skip_h1_mean_'+self.name))
        self.normop_in_skip_h1_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((hidden_size,), dtype='float32'), name='normop_in_skip_h1_var_'+self.name))


        # Second layer
        self.W2_in_tanh = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, hidden_size, output_size), name='W2_in_'+self.name))
        self.b2_in_tanh = add_to_params(self.params, theano.shared(value=np.zeros((output_size,), dtype='float32'), name='b2_in_'+self.name))

        self.W2_in_skip = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, hidden_size, output_size), name='W2_in_skip_'+self.name))
        self.b2_in_skip = add_to_params(self.params, theano.shared(value=np.zeros((output_size,), dtype='float32'), name='b2_in_skip_'+self.name))

        # Second layer batch norm / layer norm parameters
        self.normop_in_tanh_h2_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((output_size,), dtype='float32'), name='normop_in_tanh_h2_gamma_'+self.name))
        self.normop_in_tanh_h2_mean = add_to_params(self.params, theano.shared(value=np.zeros((output_size,), dtype='float32'), name='normop_in_tanh_h2_mean_'+self.name))
        self.normop_in_tanh_h2_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((output_size,), dtype='float32'), name='normop_in_tanh_h2_var_'+self.name))

        self.normop_in_skip_h2_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((output_size,), dtype='float32'), name='normop_in_skip_h2_gamma_'+self.name))
        self.normop_in_skip_h2_mean = add_to_params(self.params, theano.shared(value=np.zeros((output_size,), dtype='float32'), name='normop_in_skip_h2_mean_'+self.name))
        self.normop_in_skip_h2_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((output_size,), dtype='float32'), name='normop_in_skip_h2_var_'+self.name))

    def build_output(self, inp, bnmask):
        # Make sure bnmask is of type float32:
        if bnmask:
            bnmask = T.cast(bnmask, 'float32')

        # Execute normalization operator on inputs
        h_linear_inp, h_linear_inp_mean, h_linear_inp_var = NormalizationOperator(self.normop_type, T.dot(inp, self.W1_in_skip), self.normop_in_tanh_h1_gamma, bnmask, self.normop_in_tanh_h1_mean, self.normop_in_tanh_h1_var)

        h_nonlinear_inp, h_nonlinear_inp_mean, h_nonlinear_inp_var = NormalizationOperator(self.normop_type, T.dot(inp, self.W1_in_tanh) + self.b1_in_tanh, self.normop_in_skip_h1_gamma, bnmask, self.normop_in_skip_h1_mean, self.normop_in_skip_h1_var)

        # Compute first hidden layer
        h = T.tanh(h_nonlinear_inp) + h_linear_inp + self.b1_in_skip

        # Execute normalization operator on inputs to second hidden layer
        h2_linear_inp, h2_linear_inp_mean, h2_linear_inp_var = NormalizationOperator(self.normop_type, T.dot(h, self.W2_in_skip), self.normop_in_skip_h2_gamma, bnmask, self.normop_in_skip_h2_mean, self.normop_in_skip_h2_var)
        h2_nonlinear_inp, h2_nonlinear_inp_mean, h2_nonlinear_inp_var = NormalizationOperator(self.normop_type, T.dot(h, self.W2_in_tanh) + self.b2_in_tanh, self.normop_in_tanh_h2_gamma, bnmask, self.normop_in_tanh_h2_mean, self.normop_in_tanh_h2_var)

        output = T.tanh(h2_nonlinear_inp) + h2_linear_inp + self.b2_in_skip

        # Create batch norm updates
        updates = []
        if self.normop_type == 'BN':
            print(' Creating batch norm updates for TwoLayerMLP (' + self.name + '):')
            vars_to_update = [self.normop_in_tanh_h1_mean, self.normop_in_tanh_h1_var, self.normop_in_skip_h1_mean, self.normop_in_skip_h1_var, self.normop_in_skip_h2_mean, self.normop_in_skip_h2_var, self.normop_in_tanh_h2_mean, self.normop_in_tanh_h2_var]
            vars_estimates = [h_linear_inp_mean, h_linear_inp_var, h_nonlinear_inp_mean, h_nonlinear_inp_var, h2_linear_inp_mean, h2_linear_inp_var, h2_nonlinear_inp_mean, h2_nonlinear_inp_var]

            assert len(vars_estimates) == len(vars_to_update)

            for i in range(len(vars_estimates)):
                print('     ', vars_to_update[i])
                new_value = self.normop_moving_average_const*vars_to_update[i] \
                            + (1.0 - self.normop_moving_average_const)*vars_estimates[i]
                updates.append((vars_to_update[i], new_value))

        return output, updates


    def __init__(self, state, rng, inp_size, hidden_size, output_size, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(inp_size, hidden_size, output_size)




class UtteranceEncoder(EncoderDecoderBase):
    """
    This is the GRU-gated RNN encoder class, which operates on hidden states at the word level
    (intra-utterance level). It encodes utterances into real-valued fixed-sized vectors.
    """

    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert(word_embedding_param != None)
        self.W_emb = word_embedding_param

        """ sent weights """
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder), name='W_in_'+self.name))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder), name='W_hh_'+self.name))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='b_hh_'+self.name))

        # Initialize batch norm / layer norm parameters
        self.normop_in_h_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.qdim_encoder,), dtype='float32'), name='normop_in_h_gamma_'+self.name))
        self.normop_in_h_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_in_h_mean_'+self.name))
        self.normop_in_h_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_in_h_var_'+self.name))


        self.normop_in_x_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.qdim_encoder,), dtype='float32'), name='normop_in_x_gamma_'+self.name))
        self.normop_in_x_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_in_x_mean_'+self.name))
        self.normop_in_x_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_in_x_var_'+self.name))


        
        if self.utterance_encoder_gating == "GRU":
            self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder), name='W_in_r_'+self.name))
            self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_encoder), name='W_in_z_'+self.name))
            self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder), name='W_hh_r_'+self.name))
            self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_encoder, self.qdim_encoder), name='W_hh_z_'+self.name))
            self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='b_z_'+self.name))
            self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_encoder,), dtype='float32'), name='b_r_'+self.name))


            # Initialize batch norm / layer norm parameters
            self.normop_r_h_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.qdim_encoder,), dtype='float32'), name='normop_r_h_gamma_'+self.name))
            self.normop_r_h_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_r_h_mean_'+self.name))
            self.normop_r_h_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_r_h_var_'+self.name))

            self.normop_r_x_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.qdim_encoder,), dtype='float32'), name='normop_r_x_gamma_'+self.name))
            self.normop_r_x_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_r_x_mean_'+self.name))
            self.normop_r_x_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_r_x_var_'+self.name))

            self.normop_z_h_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.qdim_encoder,), dtype='float32'), name='normop_z_h_gamma_'+self.name))
            self.normop_z_h_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_z_h_mean_'+self.name))
            self.normop_z_h_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_z_h_var_'+self.name))

            self.normop_z_x_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.qdim_encoder,), dtype='float32'), name='normop_z_x_gamma_'+self.name))
            self.normop_z_x_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_z_x_mean_'+self.name))
            self.normop_z_x_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.normop_max_enc_seq, self.qdim_encoder), dtype='float32'), name='normop_z_x_var_'+self.name))


    # This function takes as input word indices and extracts their corresponding word embeddings
    def approx_embedder(self, x):
        return self.W_emb[x]

    def plain_step(self, x_t, m_t, bnmask_t, *args):
        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        # If 'reset_utterance_encoder_at_end_of_utterance' flag is on,
        # then reset the hidden state if this is an end-of-utterance token
        # as given by m_t
        if self.reset_utterance_encoder_at_end_of_utterance:
            hr_tm1 = m_t * h_tm1
        else:
            hr_tm1 = h_tm1

        h_t = self.sent_rec_activation(T.dot(x_t, self.W_in) + T.dot(hr_tm1, self.W_hh) + self.b_hh)

        # Return hidden state only
        return [h_t]

    def GRU_step(self, x_t, m_t, bnmask_t, *args):
        args = iter(args)
        h_tm1 = next(args)
        n_t = next(args)

        if self.reset_utterance_encoder_at_end_of_utterance:
            new_n_t = T.gt(m_t, 0.5)*(n_t + 1) # n_t + T.gt(m_t, 0.5)
        else:
            new_n_t = n_t + 1

        new_n_t = T.cast(new_n_t, 'int8')

        if n_t.ndim == 2:
            n_t_truncated = T.maximum(0, T.minimum(n_t[0,:], self.normop_max_enc_seq - 1))
        else:
            n_t_truncated = T.maximum(0, T.minimum(n_t, self.normop_max_enc_seq - 1))


        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x') 

        # If 'reset_utterance_encoder_at_end_of_utterance' flag is on,
        # then reset the hidden state if this is an end-of-utterance token
        # as given by m_t
        if self.reset_utterance_encoder_at_end_of_utterance:
            hr_tm1 = m_t * h_tm1
        else:
            hr_tm1 = h_tm1

        # Compute reset gate
        r_t_normop_x_inp, r_t_normop_x_mean, r_t_normop_x_var = NormalizationOperator(self.normop_type, T.dot(x_t, self.W_in_r), self.normop_r_x_gamma, bnmask_t, self.normop_r_x_mean[n_t_truncated, :], self.normop_r_x_var[n_t_truncated, :])
        r_t_normop_h_inp, r_t_normop_h_mean, r_t_normop_h_var = NormalizationOperator(self.normop_type, T.dot(hr_tm1, self.W_hh_r), self.normop_r_h_gamma, bnmask_t, self.normop_r_h_mean[n_t_truncated, :], self.normop_r_h_var[n_t_truncated, :])
        r_t = T.nnet.sigmoid(r_t_normop_x_inp + r_t_normop_h_inp + self.b_r)



        # Compute update gate
        z_t_normop_x_inp, z_t_normop_x_mean, z_t_normop_x_var = NormalizationOperator(self.normop_type, T.dot(x_t, self.W_in_z), self.normop_z_x_gamma, bnmask_t, self.normop_z_x_mean[n_t_truncated, :], self.normop_z_x_var[n_t_truncated, :])
        z_t_normop_h_inp, z_t_normop_h_mean, z_t_normop_h_var = NormalizationOperator(self.normop_type, T.dot(hr_tm1, self.W_hh_z), self.normop_z_h_gamma, bnmask_t, self.normop_z_h_mean[n_t_truncated, :], self.normop_z_h_var[n_t_truncated, :])
        z_t = T.nnet.sigmoid(z_t_normop_x_inp + z_t_normop_h_inp + self.b_z)

        # Compute h_tilde
        h_tilde_normop_x_inp, h_tilde_normop_x_mean, h_tilde_normop_x_var = NormalizationOperator(self.normop_type, T.dot(x_t, self.W_in), self.normop_in_x_gamma, bnmask_t, self.normop_in_x_mean[n_t_truncated, :], self.normop_in_x_var[n_t_truncated, :])

        h_tilde_normop_h_inp, h_tilde_normop_h_mean, h_tilde_normop_h_var = NormalizationOperator(self.normop_type, T.dot(r_t * hr_tm1, self.W_hh), self.normop_in_h_gamma, bnmask_t, self.normop_in_h_mean[n_t_truncated, :], self.normop_in_h_var[n_t_truncated, :])

        h_tilde = self.sent_rec_activation(h_tilde_normop_x_inp + h_tilde_normop_h_inp + self.b_hh)

        # Compute h
        h_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde

        # return states, gates and batch norm parameters
        return [h_t, T.cast(new_n_t, 'int8'), r_t, z_t, h_tilde, r_t_normop_x_mean, r_t_normop_x_var, r_t_normop_h_mean, r_t_normop_h_var, z_t_normop_x_mean, z_t_normop_x_var, z_t_normop_h_mean, z_t_normop_h_var, h_tilde_normop_x_mean, h_tilde_normop_x_var, h_tilde_normop_h_mean, h_tilde_normop_h_var]

    def build_encoder(self, x, xmask=None, bnmask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1

        # if it is not one_step then we initialize everything to previous state or zero  
        if not one_step:
            if prev_state:
                h_0, n_0 = prev_state
            else:
                h_0 = T.alloc(np.float32(0), batch_size, self.qdim_encoder)
                n_0 = T.alloc(np.int8(0), batch_size)

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_h' in kwargs 
            h_0 = kwargs['prev_h']
            n_0 = T.alloc(np.int8(0), batch_size)

        # We extract the word embeddings from the word indices
        xe = self.approx_embedder(x)
        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        bnmask_given = True
        if bnmask == None:
            bnmask_given = False
            bnmask = T.zeros(xmask.shape, dtype='float32')


        # We add ones at the the beginning of the reset vector to align the resets with y_training:
        # for example for 
        # training_x =        </s> a b c </s> d
        # xmask =               0  1 1 1  0   1
        # rolled_xmask =        1  0 1 1  1   0 1
        # Thus, we ensure that the no information in the encoder is carried from input "</s>" to "a",
        # or from "</s>" to "d". 
        # Now, the state at exactly </s> always reflects the previous utterance encoding.
        # Since the dialogue encoder uses xmask, and inputs it when xmask=0, it will input the utterance encoding
        # exactly on the </s> state.

        if xmask.ndim == 2:
            ones_vector = T.ones_like(xmask[0,:]).dimshuffle('x', 0)
            rolled_xmask = T.concatenate([ones_vector, xmask], axis=0)
        else:
            ones_scalar = theano.shared(value=numpy.ones((1), dtype='float32'), name='ones_scalar')
            rolled_xmask = T.concatenate([ones_scalar, xmask])

        # GRU Encoder
        if self.utterance_encoder_gating == "GRU":
            f_enc = self.GRU_step
            o_enc_info = [h_0, n_0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        else:
            f_enc = self.plain_step
            o_enc_info = [h_0]


        # Run through all tokens (encode everything)
        if not one_step: 
            _res, _ = theano.scan(f_enc,
                              sequences=[xe, rolled_xmask, bnmask],\
                              outputs_info=o_enc_info)
        else: # Make just one step further
            _res = f_enc(xe, rolled_xmask, bnmask, [h_0, n_0])[0]

        # Get the hidden state sequence
        if self.utterance_encoder_gating == 'GRU':
            h, n = _res[0], _res[1]
            updates = []

            # Create batch norm updates
            if self.normop_type == 'BN':
                if (not one_step) and (x.ndim == 2) and (bnmask_given):
                    updates = []
                    n_max = T.maximum(0, T.minimum(h.shape[0]-1, self.normop_max_enc_seq))
                    vars_to_update = [self.normop_r_x_mean, self.normop_r_x_var, self.normop_r_h_mean, self.normop_r_h_var, self.normop_z_x_mean, self.normop_z_x_var, self.normop_z_h_mean, self.normop_z_h_var, self.normop_in_x_mean, self.normop_in_x_var, self.normop_in_h_mean, self.normop_in_h_var]

                    assert len(_res) == len(vars_to_update)+5
                    print(' Creating batch norm updates for GRU Utterance Encoder (' + self.name + '):')
                    for varidx, var in enumerate(vars_to_update):
                        sub_new_value = self.normop_moving_average_const*var[0:n_max] \
                                                + (1.0-self.normop_moving_average_const)*_res[5+varidx][0:n_max]
                        new_value = T.set_subtensor(var[0:n_max], sub_new_value)
                        updates.append((var, new_value))
                        print('     ' + str(var))

        else:
           h = _res
           n = 0
           updates = []


        return h, n, updates

    def __init__(self, state, rng, word_embedding_param, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(word_embedding_param)



class DCGMEncoder(EncoderDecoderBase):
    """
    This is the bag-of-words (DCGM) RNN encoder class, which operates on hidden states at the word level (intra-utterance level).
    It encodes utterances into real-valued fixed-sized vectors.
    """

    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert(word_embedding_param != None)
        self.W_emb = word_embedding_param
        self.Wq_in = add_to_params(self.params, \
                                   theano.shared(value=NormalInit(self.rng, self.rankdim, self.output_dim), name='dcgm_Wq_in'+self.name))
        self.bq_in = add_to_params(self.params, \
                                   theano.shared(value=np.zeros((self.output_dim,), dtype='float32'), name='dcgm_bq_in'+self.name))

    def mean_step(self, x_t, m_t, *args):
        args = iter(args)
        
        # already computed avg 
        avg_past = next(args)
        n_past = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x') 
        
        # reset avg
        avg_past_r = m_t * avg_past 
        n_past_r = m_t.T * n_past


        n = n_past_r + 1.0

        resized_n = T.repeat(n.T, avg_past_r.shape[1], axis=1)
        avg = (avg_past_r * (resized_n - 1) + x_t) / resized_n

        # Old implementation:
        #avg = (avg_past_r * (n[:, None] - 1) + x_t) / n[:, None]

        # return state and pooled state
        return avg, n

    def approx_embedder(self, x):
        return self.W_emb[x]

    def build_encoder(self, x, xmask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True

        if x.ndim == 2:
            batch_size = x.shape[1]
        else:
            batch_size = 1

        # if it is not one_step then we initialize everything to previous state or zero  
        if not one_step:
            if prev_state:
                avg_0, n_0 = prev_state
            else:
                avg_0 = T.alloc(np.float32(0), batch_size, self.rankdim)
                n_0 = T.alloc(np.float32(0), batch_size)

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_avg' in kwargs 
            avg_0 = kwargs['prev_avg']

        
        # in sampling mode (i.e. one step) we require 
        xe = self.approx_embedder(x)
        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        if xmask.ndim == 2:
            ones_vector = T.ones_like(xmask[0,:]).dimshuffle('x', 0)
            rolled_xmask = T.concatenate([ones_vector, xmask], axis=0)
        else:
            ones_scalar = theano.shared(value=numpy.ones((1), dtype='float32'), name='ones_scalar')
            rolled_xmask = T.concatenate([ones_scalar, xmask])

        f_enc = self.mean_step
        o_enc_info = [avg_0, n_0] 



        # Run through all tokens (encode everything)
        if not one_step: 
            _res, _ = theano.scan(f_enc,
                              sequences=[xe, rolled_xmask],\
                              outputs_info=o_enc_info)
        else: # Make just one step further
            _res, _ = f_enc(xe, rolled_xmask, [avg_0, n_0])
        
        avg, n = _res[0], _res[1]

        # Linear activation
        avg_q = T.dot(avg, self.Wq_in) + self.bq_in
        return avg_q, avg, n

    def __init__(self, state, rng, word_embedding_param, output_dim, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.output_dim = output_dim
        self.init_params(word_embedding_param)


class DialogEncoder(EncoderDecoderBase):
    """
    This is the context RNN encoder class, which operates on hidden states at the dialogue level
    (inter-utterance level). At the end of each utterance, it updates its hidden state using the incoming
    input from the utterance encoder(s).
    """

    def init_params(self):
        """ Context weights """

        # If the dialogue encoder is diabled, do not initialize any parameters
        if self.disable_dialogue_encoder:
            return

        if self.bidirectional_utterance_encoder:
            # With the bidirectional flag, the dialog encoder gets input 
            # from both the forward and backward utterance encoders, hence it is double qdim_encoder
            input_dim = self.qdim_encoder * 2
        else:
            # Without the bidirectional flag, the dialog encoder only gets input
            # from the forward utterance encoder, which has dim self.qdim_encoder
            input_dim = self.qdim_encoder


        transformed_input_dim = input_dim
        if self.deep_dialogue_encoder_input:
            transformed_input_dim = self.sdim

            self.input_mlp = TwoLayerMLP(self.state, self.rng, input_dim, self.sdim, self.sdim, self, '_input_mlp_'+self.name)
            self.params += self.input_mlp.params

        self.Ws_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in'+self.name))
        self.Ws_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh'+self.name))
        self.bs_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_hh'+self.name))

        if self.dialogue_encoder_gating == "GRU":
            self.Ws_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in_r'+self.name))
            self.Ws_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in_z'+self.name))
            self.Ws_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_r'+self.name))
            self.Ws_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_z'+self.name))
            self.bs_z = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_z'+self.name))
            self.bs_r = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_r'+self.name))

            # Linear skip connections, which acts as an "overwrite" mechanism.
            # It allows each GRU unit to replace its hidden state with the incoming input.
            # This is potentially useful, for example, if the dialogue changes topic.
            self.Ws_in_overwrite = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in_overwrite'+self.name))
            self.bs_overwrite = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_overwrite'+self.name))

            # Gating mechanism defining whether to overwrite or not
            self.Ws_in_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, transformed_input_dim, self.sdim), name='Ws_in_o'+self.name))    
            self.Ws_hh_o = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_o'+self.name))
            self.bs_o = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_o'+self.name))




            # Batch norm parameters
            self.normop_in_hs_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.sdim,), dtype='float32'), name='normop_in_hs_gamma'+self.name))
            self.normop_in_hs_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='normop_in_hs_mean'+self.name))
            self.normop_in_hs_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.sdim,), dtype='float32'), name='normop_in_hs_var'+self.name))

            self.normop_in_h_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.sdim,), dtype='float32'), name='normop_in_h_gamma'+self.name))
            self.normop_in_h_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='normop_in_h_mean'+self.name))
            self.normop_in_h_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.sdim,), dtype='float32'), name='normop_in_h_var'+self.name))

            self.normop_rs_hs_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.sdim,), dtype='float32'), name='normop_rs_hs_gamma'+self.name))
            self.normop_rs_hs_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='normop_rs_hs_mean'+self.name))
            self.normop_rs_hs_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.sdim,), dtype='float32'), name='normop_rs_hs_var'+self.name))

            self.normop_rs_h_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.sdim,), dtype='float32'), name='normop_rs_h_gamma'+self.name))
            self.normop_rs_h_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='normop_rs_h_mean'+self.name))
            self.normop_rs_h_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.sdim,), dtype='float32'), name='normop_rs_h_var'+self.name))

            self.normop_zs_hs_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.sdim,), dtype='float32'), name='normop_zs_hs_gamma'+self.name))
            self.normop_zs_hs_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='normop_zs_hs_mean'+self.name))
            self.normop_zs_hs_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.sdim,), dtype='float32'), name='normop_zs_hs_var'+self.name))

            self.normop_zs_h_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.sdim,), dtype='float32'), name='normop_zs_h_gamma'+self.name))
            self.normop_zs_h_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='normop_zs_h_mean'+self.name))
            self.normop_zs_h_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.sdim,), dtype='float32'), name='normop_zs_h_var'+self.name))

            self.normop_os_hs_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.sdim,), dtype='float32'), name='normop_os_hs_gamma'+self.name))
            self.normop_os_hs_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='normop_os_hs_mean'+self.name))
            self.normop_os_hs_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.sdim,), dtype='float32'), name='normop_os_hs_var'+self.name))

            self.normop_os_h_gamma = add_to_params(self.params, theano.shared(value=self.normop_gamma_init*np.ones((self.sdim,), dtype='float32'), name='normop_os_h_gamma'+self.name))
            self.normop_os_h_mean = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='normop_os_h_mean'+self.name))
            self.normop_os_h_var = add_to_params(self.params, theano.shared(value=(1e-7)*np.ones((self.sdim,), dtype='float32'), name='normop_os_h_var'+self.name))

    def plain_dialogue_step(self, h_t, m_t, bnmask_t, hs_tm1, *args):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')


        hs_tilde = self.dialogue_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(hs_tm1, self.Ws_hh) + self.bs_hh)

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_tilde

        return hs_t


    def GRU_dialogue_step(self, h_t, m_t, bnmask_t, hs_tm1, *args):

        #rs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_r) + T.dot(hs_tm1, self.Ws_hh_r) + self.bs_r)
        rs_t_normop_h_inp, rs_t_normop_h_mean, rs_t_normop_h_var = NormalizationOperator(self.normop_type, T.dot(h_t, self.Ws_in_r), self.normop_rs_h_gamma, bnmask_t, self.normop_rs_h_mean, self.normop_rs_h_var)
        rs_t_normop_hs_inp, rs_t_normop_hs_mean, rs_t_normop_hs_var = NormalizationOperator(self.normop_type, T.dot(hs_tm1, self.Ws_hh_r), self.normop_rs_hs_gamma, bnmask_t, self.normop_rs_hs_mean, self.normop_rs_hs_var)
        rs_t = T.nnet.sigmoid(rs_t_normop_h_inp + rs_t_normop_hs_inp + self.bs_r)


        #zs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_z) + T.dot(hs_tm1, self.Ws_hh_z) + self.bs_z)
        zs_t_normop_h_inp, zs_t_normop_h_mean, zs_t_normop_h_var = NormalizationOperator(self.normop_type, T.dot(h_t, self.Ws_in_z), self.normop_zs_h_gamma, bnmask_t, self.normop_zs_h_mean, self.normop_zs_h_var)
        zs_t_normop_hs_inp, zs_t_normop_hs_mean, zs_t_normop_hs_var = NormalizationOperator(self.normop_type, T.dot(hs_tm1, self.Ws_hh_z), self.normop_zs_hs_gamma, bnmask_t, self.normop_zs_hs_mean, self.normop_zs_hs_var)
        zs_t = T.nnet.sigmoid(zs_t_normop_h_inp + zs_t_normop_hs_inp + self.bs_z)

        #os_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_o) + T.dot(hs_tm1, self.Ws_hh_o) + self.bs_o)
        os_t_normop_h_inp, os_t_normop_h_mean, os_t_normop_h_var = NormalizationOperator(self.normop_type, T.dot(h_t, self.Ws_in_o), self.normop_os_h_gamma, bnmask_t, self.normop_os_h_mean, self.normop_os_h_var)
        os_t_normop_hs_inp, os_t_normop_hs_mean, os_t_normop_hs_var = NormalizationOperator(self.normop_type, T.dot(hs_tm1, self.Ws_hh_o), self.normop_os_hs_gamma, bnmask_t, self.normop_os_hs_mean, self.normop_os_hs_var)
        os_t = T.nnet.sigmoid(os_t_normop_h_inp + os_t_normop_hs_inp + self.bs_o)

        hs_overwrite = T.dot(h_t, self.Ws_in_overwrite) + self.bs_overwrite


        hs_tilde_normop_h_inp, hs_tilde_normop_h_mean, hs_tilde_normop_h_var = NormalizationOperator(self.normop_type, T.dot(h_t, self.Ws_in), self.normop_in_h_gamma, bnmask_t, self.normop_in_h_mean, self.normop_in_h_var)
        hs_tilde_normop_hs_inp, hs_tilde_normop_hs_mean, hs_tilde_normop_hs_var = NormalizationOperator(self.normop_type, T.dot(rs_t * hs_tm1, self.Ws_hh), self.normop_in_hs_gamma, bnmask_t, self.normop_in_hs_mean, self.normop_in_hs_var)
        hs_tilde = self.dialogue_rec_activation(hs_tilde_normop_h_inp + hs_tilde_normop_hs_inp + self.bs_hh)

        hs_hat = (np.float32(1.) - os_t) * hs_tilde + os_t * hs_overwrite

        hs_update = (np.float32(1.) - zs_t) * hs_tm1 + zs_t * hs_hat
         
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_update

        return hs_t, hs_hat, rs_t, zs_t, rs_t_normop_h_mean, rs_t_normop_h_var, rs_t_normop_hs_mean, rs_t_normop_hs_var, zs_t_normop_h_mean, zs_t_normop_h_var, zs_t_normop_hs_mean, zs_t_normop_hs_var, os_t_normop_h_mean, os_t_normop_h_var, os_t_normop_hs_mean, os_t_normop_hs_var, hs_tilde_normop_h_mean, hs_tilde_normop_h_var, hs_tilde_normop_hs_mean, hs_tilde_normop_hs_var

    def build_encoder(self, h, x, xmask=None, bnmask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            if prev_state:
                hs_0 = prev_state
            else:
                hs_0 = T.alloc(np.float32(0), batch_size, self.sdim)

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)       

        bnmask_given = True
        if bnmask == None:
            bnmask_given = False
            bnmask = T.zeros(xmask.shape, dtype='float32')


        # If the dialogue encoder is disabled, return zeros
        if self.disable_dialogue_encoder:
            if x.ndim == 2:
                zeros_out = T.alloc(np.float32(0), x.shape[0], x.shape[1], self.sdim)
            else:
                zeros_out = T.alloc(np.float32(0), x.shape[0], self.sdim)

            return zeros_out, []


        if self.dialogue_encoder_gating == "GRU":
            f_hier = self.GRU_dialogue_step
            o_hier_info = [hs_0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        else:
            f_hier = self.plain_dialogue_step
            o_hier_info = [hs_0]

        if self.deep_dialogue_encoder_input:        
            transformed_h, updates = self.input_mlp.build_output(h, xmask)
        else:
            transformed_h = h
            updates = []

        # The hs sequence is based on the original mask
        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[transformed_h, xmask, bnmask],\
                               outputs_info=o_hier_info)
        # Just one step further
        else:
            _res = f_hier(transformed_h, xmask, bnmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res


        # Create batch norm updates
        if self.normop_type == 'BN':
            if self.dialogue_encoder_gating == "GRU":
                if (not one_step) and (h.ndim == 3) and (bnmask_given):
                    vars_to_update = [self.normop_rs_h_mean, self.normop_rs_h_var, self.normop_rs_hs_mean, self.normop_rs_hs_var, self.normop_zs_h_mean, self.normop_zs_h_var, self.normop_zs_hs_mean, self.normop_zs_hs_var, self.normop_os_h_mean, self.normop_os_h_var, self.normop_os_hs_mean, self.normop_os_hs_var, self.normop_in_h_mean, self.normop_in_h_var, self.normop_in_hs_mean, self.normop_in_hs_var]

                    batch_examples_per_timestep = T.sum(bnmask, axis=1).dimshuffle(0, 'x')

                    assert len(_res) == len(vars_to_update)+4
                    print(' Creating batch norm updates for GRU Dialog Encoder (' + self.name + '):')
                    for varidx, var in enumerate(vars_to_update):
                        average_var = T.sum(_res[4+varidx]*batch_examples_per_timestep, axis=0) \
                                        / T.sum(batch_examples_per_timestep, axis=0)

                        new_value = self.normop_moving_average_const*var \
                                                + (1.0-self.normop_moving_average_const)*average_var

                        updates.append((var, new_value))
                        print('     ' + str(var))

        return hs, updates

    def __init__(self, state, rng, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params()




class DialogDummyEncoder(EncoderDecoderBase):
    """
    This class operates on hidden states at the dialogue level (inter-utterance level).
    At the end of each utterance, the input from the utterance encoder(s) is transferred
    to its hidden state, which can then be transfered to the decoder.
    """

    def init_params(self):
        """ Context weights """
        if self.deep_direct_connection:
            self.Ws_dummy_deep_input = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.inp_dim, self.inp_dim), name='Ws_dummy_deep_input'+self.name))
            self.bs_dummy_deep_input = add_to_params(self.params, theano.shared(value=np.zeros((self.inp_dim,), dtype='float32'), name='bs_dummy_deep_input'+self.name))


    def plain_dialogue_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        transformed_h_t = h_t
        if self.deep_direct_connection:
            transformed_h_t = self.dialogue_rec_activation(T.dot(h_t, self.Ws_dummy_deep_input) + self.bs_dummy_deep_input)

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * transformed_h_t 
        return hs_t

    def build_encoder(self, h, x, xmask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            if prev_state:
                hs_0 = prev_state
            else:
                hs_0 = T.alloc(np.float32(0), batch_size, self.inp_dim) 

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        f_hier = self.plain_dialogue_step
        o_hier_info = [hs_0]
        
        # The hs sequence is based on the original mask
        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[h, xmask],\
                               outputs_info=o_hier_info)
        # Just one step further
        else:
            _res = f_hier(h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res

        return hs 

    def __init__(self, state, rng, parent, inp_dim, name=''):
        self.inp_dim = inp_dim
        self.name = name
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.init_params()



class UtteranceDecoder(EncoderDecoderBase):
    """
    This is the decoder RNN class, which operates at the word level (intra-utterance level).
    It is an RNNLM conditioned on additional information (e.g. context level hidden state, latent variables)
    """

    NCE = 0
    EVALUATION = 1
    SAMPLING = 2
    BEAM_SEARCH = 3

    def __init__(self, state, rng, parent, dialog_encoder, word_embedding_param):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        # Take as input the encoder instance for the embeddings..
        # To modify in the future
        assert(word_embedding_param != None)
        self.word_embedding_param = word_embedding_param
        self.dialog_encoder = dialog_encoder
        self.trng = MRG_RandomStreams(self.seed)
        self.init_params()

    def init_params(self):

        assert self.utterance_decoder_gating == self.utterance_decoder_gating.upper()

        # Compute input dimensionality
        if self.direct_connection_between_encoders_and_decoder:
            # When there is a direct connection between encoder and decoder, 
            # the input has dimensionality sdim + qdim_decoder if forward encoder, and
            # sdim + 2 x qdim_decoder for bidirectional encoder
            if self.bidirectional_utterance_encoder:
                self.input_dim = self.sdim + self.qdim_encoder*2
            else:
                self.input_dim = self.sdim + self.qdim_encoder
        else:
            # When there is no connection between encoder and decoder, 
            # the input has dimensionality sdim
            self.input_dim = self.sdim

        if self.add_latent_gaussian_per_utterance and self.add_latent_piecewise_per_utterance:
            if self.condition_decoder_only_on_latent_variable:
                self.input_dim = self.latent_gaussian_per_utterance_dim + self.latent_piecewise_per_utterance_dim
            else:
                self.input_dim += self.latent_gaussian_per_utterance_dim + self.latent_piecewise_per_utterance_dim
        elif self.add_latent_gaussian_per_utterance:
            if self.condition_decoder_only_on_latent_variable:
                self.input_dim = self.latent_gaussian_per_utterance_dim
            else:
                self.input_dim += self.latent_gaussian_per_utterance_dim
        elif self.add_latent_piecewise_per_utterance:
            if self.condition_decoder_only_on_latent_variable:
                self.input_dim = self.latent_piecewise_per_utterance_dim
            else:
                self.input_dim += self.latent_piecewise_per_utterance_dim

        # Compute hidden state dimensionality
        if self.utterance_decoder_gating == "LSTM":
            # For LSTM decoder, the state hd is the concatenation of the cell state and hidden state
            self.complete_hidden_state_size = self.qdim_decoder*2
        else:
            self.complete_hidden_state_size = self.qdim_decoder

        # Compute deep input
        if self.deep_utterance_decoder_input:
            self.input_mlp = OneLayerMLP(self.state, self.rng, self.input_dim,
                                         self.input_dim, self.input_dim, self, '_input_mlp_utterance_decoder')
            self.params += self.input_mlp.params


        self.bd_out = add_to_params(self.params, theano.shared(value=np.zeros((self.idim,), dtype='float32'), name='bd_out'))
        self.Wd_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='Wd_emb'))

        """ RNN decoder weights """
        if self.utterance_decoder_gating == "" or self.utterance_decoder_gating == "NONE" \
            or self.utterance_decoder_gating == "GRU" or self.utterance_decoder_gating == "LSTM":

            self.Wd_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_hh'))
            self.bd_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_decoder,), dtype='float32'), name='bd_hh'))
            self.Wd_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_decoder), name='Wd_in')) 

            # We only include the initial hidden state if the utterance decoder is NOT reset 
            # and if its NOT a collapsed model (i.e. collapsed to standard RNN). 
            # In the collapsed model, we always initialize hidden state to zero.
            if (not self.collaps_to_standard_rnn) and (self.reset_utterance_decoder_at_end_of_utterance):
                self.Wd_s_0 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.complete_hidden_state_size), name='Wd_s_0'))
                self.bd_s_0 = add_to_params(self.params, theano.shared(value=np.zeros((self.complete_hidden_state_size,), dtype='float32'), name='bd_s_0'))

        if self.utterance_decoder_gating == "GRU":
            self.Wd_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_decoder), name='Wd_in_r'))
            self.Wd_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_decoder), name='Wd_in_z'))
            self.Wd_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_hh_r'))
            self.Wd_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_hh_z'))
            self.bd_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_decoder,), dtype='float32'), name='bd_r'))
            self.bd_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_decoder,), dtype='float32'), name='bd_z'))
        
            if self.decoder_bias_type == 'all':
                self.Wd_s_q = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_s_q'))
                self.Wd_s_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_s_z'))
                self.Wd_s_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_s_r'))

        elif self.utterance_decoder_gating == "LSTM":
            # Input gate
            self.Wd_in_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_decoder), name='Wd_in_i'))
            self.Wd_hh_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_hh_i'))
            self.Wd_c_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_c_i'))
            self.bd_i = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_decoder,), dtype='float32'), name='bd_i'))

            # Forget gate
            self.Wd_in_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_decoder), name='Wd_in_f'))
            self.Wd_hh_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_hh_f'))
            self.Wd_c_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_c_f'))
            self.bd_f = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_decoder,), dtype='float32'), name='bd_f'))

            # Output gate
            self.Wd_in_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim_decoder), name='Wd_in_o'))
            self.Wd_hh_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_hh_o'))
            self.Wd_c_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.qdim_decoder), name='Wd_c_o'))
            self.bd_o = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_decoder,), dtype='float32'), name='bd_o'))

            if self.decoder_bias_type == 'all' or self.decoder_bias_type == 'selective':
                # Input gate
                self.Wd_s_i = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_s_i'))
                # Forget gate
                self.Wd_s_f = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_s_f'))
                # Cell input
                self.Wd_s = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_s'))
                # Output gate
                self.Wd_s_o = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_s_o'))
        elif self.utterance_decoder_gating == "BOW":
            self.Wd_bow_W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_bow_W_in'))
            self.Wd_bow_b_in = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim_decoder,), dtype='float32'), name='Wd_bow_b_in'))


        # Selective gating mechanism
        if self.decoder_bias_type == 'selective':
            # Selective gating mechanism is not compatible with bag-of-words decoder
            assert not self.utterance_decoder_gating == "BOW"

            # Selective gating mechanism for LSTM
            if self.utterance_decoder_gating == "LSTM":
                self.bd_sel = add_to_params(self.params, theano.shared(value=np.zeros((self.input_dim,), dtype='float32'), name='bd_sel'))

                self.Wd_sel_s = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.input_dim, self.input_dim), \
                                                        name='Wd_sel_s'))
                # x_{n-1} -> g_r
                self.Wd_sel_e = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.rankdim, self.input_dim), \
                                                        name='Wd_sel_e'))
                # h_{n-1} -> g_r
                self.Wd_sel_h = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.input_dim), \
                                                        name='Wd_sel_h'))
                # c_{n-1} -> g_r
                self.Wd_sel_c = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.input_dim), \
                                                        name='Wd_sel_c'))
            else: # Selective gating mechanism for GRU and plain decoder
                self.bd_sel = add_to_params(self.params, theano.shared(value=np.zeros((self.input_dim,), dtype='float32'), name='bd_sel'))
                self.Wd_s_q = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.qdim_decoder), name='Wd_s_q'))
                # s -> g_r
                self.Wd_sel_s = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.input_dim, self.input_dim), \
                                                        name='Wd_sel_s'))
                # x_{n-1} -> g_r
                self.Wd_sel_e = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.rankdim, self.input_dim), \
                                                        name='Wd_sel_e'))
                # h_{n-1} -> g_r
                self.Wd_sel_h = add_to_params(self.params, \
                                          theano.shared(value=NormalInit(self.rng, self.qdim_decoder, self.input_dim), \
                                                        name='Wd_sel_h'))




        ######################   
        # Output layer weights
        ######################
        if self.maxout_out:
            if int(self.qdim_decoder) != 2*int(self.rankdim):
                raise ValueError('Error with maxout configuration in UtteranceDecoder!'
                                 + 'For maxout to work we need qdim_decoder = 2x rankdim')

        out_target_dim = self.qdim_decoder
        if not self.maxout_out:
            out_target_dim = self.rankdim

        self.Wd_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim_decoder, out_target_dim), name='Wd_out'))
         
        # Set up deep output
        if self.deep_utterance_decoder_out:

            if self.utterance_decoder_gating == "" or self.utterance_decoder_gating == "NONE" \
                or self.utterance_decoder_gating == "GRU" or self.utterance_decoder_gating == "LSTM":

                self.Wd_e_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, out_target_dim), name='Wd_e_out'))
                self.bd_e_out = add_to_params(self.params, theano.shared(value=np.zeros((out_target_dim,), dtype='float32'), name='bd_e_out'))
             
            if self.decoder_bias_type != 'first': 
                self.Wd_s_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, out_target_dim), name='Wd_s_out'))


    def build_output_layer(self, hs, xd, hd):
        if self.utterance_decoder_gating == "LSTM":
            if hd.ndim != 2:
                pre_activ = T.dot(hd[:, :, 0:self.qdim_decoder], self.Wd_out)
            else:
                pre_activ = T.dot(hd[:, 0:self.qdim_decoder], self.Wd_out)
        else:
            pre_activ = T.dot(hd, self.Wd_out)
        
        if self.deep_utterance_decoder_out:

            if self.utterance_decoder_gating == "" or self.utterance_decoder_gating == "NONE" \
                or self.utterance_decoder_gating == "GRU" or self.utterance_decoder_gating == "LSTM":

                pre_activ += T.dot(xd, self.Wd_e_out) + self.bd_e_out

            if self.decoder_bias_type != 'first':
                pre_activ += T.dot(hs, self.Wd_s_out)
                # ^ if bias all, bias the deep output

        if self.maxout_out:
            pre_activ = Maxout(2)(pre_activ)

        return pre_activ

    def build_next_probs_predictor(self, inp, x, prev_state):
        """ 
        Return output probabilities given prev_words x, hierarchical pass hs, and previous hd
        hs should always be the same (and should not be updated).
        """
        return self.build_decoder(inp, x, mode=UtteranceDecoder.BEAM_SEARCH, prev_state=prev_state)

    def approx_embedder(self, x):
        # Here we use the same embeddings learnt in the encoder.. !!!
        return self.word_embedding_param[x]
     
    def output_softmax(self, pre_activ):
        # returns a (timestep, bs, idim) matrix (huge)
        return SoftMax(T.dot(pre_activ, self.Wd_emb.T) + self.bd_out)
    
    def output_nce(self, pre_activ, y, y_hat):
        # returns a (timestep, bs, pos + neg) matrix (very small)
        target_embedding = self.Wd_emb[y]
        # ^ target embedding is (timestep x bs, rankdim)
        noise_embedding = self.Wd_emb[y_hat]
        # ^ noise embedding is (10, timestep x bs, rankdim)
        
        # pre_activ is (timestep x bs x rankdim)
        pos_scores = (target_embedding * pre_activ).sum(2)
        neg_scores = (noise_embedding * pre_activ).sum(3)
 
        pos_scores += self.bd_out[y]
        neg_scores += self.bd_out[y_hat]
         
        pos_noise = self.parent.t_noise_probs[y] * 10
        neg_noise = self.parent.t_noise_probs[y_hat] * 10
        
        pos_scores = - T.log(T.nnet.sigmoid(pos_scores - T.log(pos_noise)))
        neg_scores = - T.log(1 - T.nnet.sigmoid(neg_scores - T.log(neg_noise))).sum(0)
        return pos_scores + neg_scores

    def build_decoder(self, decoder_inp, x, xmask=None, xdropmask=None, y=None, y_neg=None, mode=EVALUATION, prev_state=None, step_num=None):

        # If model collapses to standard RNN reset all input to decoder
        if self.collaps_to_standard_rnn:
            decoder_inp = decoder_inp * 0

        # Compute deep input
        if self.deep_utterance_decoder_input:
            decoder_inp, updates = self.input_mlp.build_output(decoder_inp, xmask)
        else:
            updates = []


        # Check parameter consistency
        if mode == UtteranceDecoder.EVALUATION or mode == UtteranceDecoder.NCE:
            assert y
        else:
            assert not y
            assert prev_state
         
        # if mode == EVALUATION
        #   xd = (timesteps, batch_size, qdim_decoder)
        #
        # if mode != EVALUATION
        #   xd = (n_samples, dim)

        # If a drop mask is given, replace 'dropped' tokens with 'unk' token as input
        # to the decoder RNN.
        if self.decoder_drop_previous_input_tokens and xdropmask:
            xdropmask = xdropmask.dimshuffle(0, 1, 'x')
            xd = xdropmask*self.approx_embedder(x) + (1-xdropmask)*self.word_embedding_param[self.unk_sym].dimshuffle('x', 'x', 0)
        else:
            xd = self.approx_embedder(x)


        if not xmask:
            xmask = T.neq(x, self.eos_sym)
        
        # we must zero out the </s> embedding
        # i.e. the embedding x_{-1} is the 0 vector
        # as well as hd_{-1} which will be reseted in the scan functions
        if xd.ndim != 3:
            assert mode != UtteranceDecoder.EVALUATION
            xd = (xd.dimshuffle((1, 0)) * xmask).dimshuffle((1, 0))
        else:
            assert mode == UtteranceDecoder.EVALUATION or mode == UtteranceDecoder.NCE
            xd = (xd.dimshuffle((2,0,1)) * xmask).dimshuffle((1,2,0))

        # Run RNN decoder
        if self.utterance_decoder_gating == "" or self.utterance_decoder_gating == "NONE" \
            or self.utterance_decoder_gating == "GRU" or self.utterance_decoder_gating == "LSTM":

            if prev_state:
                hd_init = prev_state
            else:
                hd_init = T.alloc(np.float32(0), x.shape[1], self.complete_hidden_state_size)

            if self.utterance_decoder_gating == "LSTM":
                f_dec = self.LSTM_step
                o_dec_info = [hd_init]
                if self.decoder_bias_type == "selective":
                    o_dec_info += [None, None]
            elif self.utterance_decoder_gating == "GRU":
                f_dec = self.GRU_step
                o_dec_info = [hd_init, None, None, None]
                if self.decoder_bias_type == "selective":
                    o_dec_info += [None, None]
            else: # No gating
                f_dec = self.plain_step
                o_dec_info = [hd_init]
                if self.decoder_bias_type == "selective":
                    o_dec_info += [None, None] 

            # If the mode of the decoder is EVALUATION
            # then we evaluate by default all the utterances
            # xd - i.e. xd.ndim == 3, xd = (timesteps, batch_size, qdim_decoder)
            if mode == UtteranceDecoder.EVALUATION or mode == UtteranceDecoder.NCE: 
                _res, _ = theano.scan(f_dec,
                                  sequences=[xd, xmask, decoder_inp],\
                                  outputs_info=o_dec_info)
            # else we evaluate only one step of the recurrence using the
            # previous hidden states and the previous computed hierarchical 
            # states.
            else:
                _res = f_dec(xd, xmask, decoder_inp, prev_state)

            if isinstance(_res, list) or isinstance(_res, tuple):
                hd = _res[0]
            else:
                hd = _res

            # OBSOLETE:
            #   if we are using selective bias, we should update our decoder_inp
            #   to the step-selective decoder_inp
            #   if self.decoder_bias_type == "selective":
            #       decoder_inp = _res[1]

        elif self.utterance_decoder_gating == "BOW": # BOW (bag of words) decoder
            hd = T.dot(decoder_inp, self.Wd_bow_W_in) + self.Wd_bow_b_in

        pre_activ = self.build_output_layer(decoder_inp, xd, hd)

        # EVALUATION  : Return target_probs + all the predicted ranks
        # target_probs.ndim == 3
        if mode == UtteranceDecoder.EVALUATION:
            outputs = self.output_softmax(pre_activ)
            target_probs = GrabProbs(outputs, y)
            return target_probs, hd, outputs, updates

        elif mode == UtteranceDecoder.NCE:
            return self.output_nce(pre_activ, y, y_neg), hd, updates

        # BEAM_SEARCH : Return output (the softmax layer) + the new hidden states
        elif mode == UtteranceDecoder.BEAM_SEARCH:
            return self.output_softmax(pre_activ), hd

        # SAMPLING    : Return a vector of n_sample from the output layer 
        #                 + log probabilities + the new hidden states
        elif mode == UtteranceDecoder.SAMPLING:
            outputs = self.output_softmax(pre_activ)
            if outputs.ndim == 1:
                outputs = outputs.dimshuffle('x', 0) 
            sample = self.trng.multinomial(pvals=outputs, dtype='int64').argmax(axis=-1)
            if outputs.ndim == 1:
                sample = sample[0] 
            log_prob = -T.log(T.diag(outputs.T[sample])) 
            return sample, log_prob, hd

    def LSTM_step(self, xd_t, m_t, decoder_inp_t, hd_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        # If model collapses to standard RNN, or the 'reset_utterance_decoder_at_end_of_utterance' flag is off,
        # then never reset decoder. Otherwise, reset the decoder at every utterance turn.
        if (not self.collaps_to_standard_rnn) and (self.reset_utterance_decoder_at_end_of_utterance):
            hd_tm1 = (m_t) * hd_tm1 + (1 - m_t) * T.tanh(T.dot(decoder_inp_t, self.Wd_s_0) + self.bd_s_0)

        # Unlike the GRU gating function, the LSTM gating function needs to keep track of two vectors:
        # the output state and the cell state. To align the implementation with the GRU, we store 
        # both of these two states in a single vector for every time step, split them up for computation and
        # then concatenate them back together at the end.

        # Given the previous concatenated hidden states, split them up into output state and cell state.
        # By convention, we assume that the output state is always first, and the cell state second.
        hd_tm1_tilde = hd_tm1[:, 0:self.qdim_decoder]
        cd_tm1_tilde = hd_tm1[:, self.qdim_decoder:self.qdim_decoder*2]
  
        # In the 'selective' decoder bias type each hidden state of the decoder
        # RNN receives the decoder_inp_t modified by the selective bias -> decoder_inpr_t 
        if self.decoder_bias_type == 'selective':
            rd_sel_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_sel_e) + T.dot(hd_tm1_tilde, self.Wd_sel_h) + T.dot(cd_tm1_tilde, self.Wd_sel_c) + T.dot(decoder_inp_t, self.Wd_sel_s) + self.bd_sel)
            decoder_inpr_t = rd_sel_t * decoder_inp_t

            id_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_i) + T.dot(hd_tm1_tilde, self.Wd_hh_i) \
                                  + T.dot(decoder_inpr_t, self.Wd_s_i) \
                                  + T.dot(cd_tm1_tilde, self.Wd_c_i) + self.bd_i)
            fd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_f) + T.dot(hd_tm1_tilde, self.Wd_hh_f) \
                                  + T.dot(decoder_inpr_t, self.Wd_s_f) \
                                  + T.dot(cd_tm1_tilde, self.Wd_c_f) + self.bd_f)
            cd_t = fd_t*cd_tm1_tilde + id_t*self.sent_rec_activation(T.dot(xd_t, self.Wd_in)  \
                                  + T.dot(decoder_inpr_t, self.Wd_s) \
                                  + T.dot(hd_tm1_tilde, self.Wd_hh) + self.bd_hh)
            od_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_o) + T.dot(hd_tm1_tilde, self.Wd_hh_o) \
                                  + T.dot(decoder_inpr_t, self.Wd_s_o) \
                                  + T.dot(cd_t, self.Wd_c_o) + self.bd_o)

            # Concatenate output state and cell state into one vector
            hd_t = T.concatenate([od_t*self.sent_rec_activation(cd_t), cd_t], axis=1)
            output = (hd_t, decoder_inpr_t, rd_sel_t)
        
        # In the 'all' decoder bias type each hidden state of the decoder
        # RNN receives the decoder_inp_t vector as bias without modification
        elif self.decoder_bias_type == 'all':
            id_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_i) + T.dot(hd_tm1_tilde, self.Wd_hh_i) \
                                  + T.dot(decoder_inp_t, self.Wd_s_i) \
                                  + T.dot(cd_tm1_tilde, self.Wd_c_i) + self.bd_i)
            fd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_f) + T.dot(hd_tm1_tilde, self.Wd_hh_f) \
                                  + T.dot(decoder_inp_t, self.Wd_s_f) \
                                  + T.dot(cd_tm1_tilde, self.Wd_c_f) + self.bd_f)
            cd_t = fd_t*cd_tm1_tilde + id_t*self.sent_rec_activation(T.dot(xd_t, self.Wd_in)  \
                                  + T.dot(decoder_inp_t, self.Wd_s) \
                                  + T.dot(hd_tm1_tilde, self.Wd_hh) + self.bd_hh)
            od_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_o) + T.dot(hd_tm1_tilde, self.Wd_hh_o) \
                                  + T.dot(decoder_inp_t, self.Wd_s_o) \
                                  + T.dot(cd_t, self.Wd_c_o) + self.bd_o)

            # Concatenate output state and cell state into one vector
            hd_t = T.concatenate([od_t*self.sent_rec_activation(cd_t), cd_t], axis=1)
            output = (hd_t,)
        else:
            # Do not bias the decoder at every time, instead,
            # force it to store very useful information in the first state.
            id_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_i) + T.dot(hd_tm1_tilde, self.Wd_hh_i) \
                                  + T.dot(cd_tm1_tilde, self.Wd_c_i) + self.bd_i)
            fd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_f) + T.dot(hd_tm1_tilde, self.Wd_hh_f) \
                                  + T.dot(cd_tm1_tilde, self.Wd_c_f) + self.bd_f)
            cd_t = fd_t*cd_tm1_tilde + id_t*self.sent_rec_activation(T.dot(xd_t, self.Wd_in_c)  \
                                  + T.dot(hd_tm1_tilde, self.Wd_hh) + self.bd_hh)
            od_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_o) + T.dot(hd_tm1_tilde, self.Wd_hh_o) \
                                  + T.dot(cd_t, self.Wd_c_o) + self.bd_o)

            # Concatenate output state and cell state into one vector
            hd_t = T.concatenate([od_t*self.sent_rec_activation(cd_t), cd_t], axis=1)
            output = (hd_t,)

        return output

    def GRU_step(self, xd_t, m_t, decoder_inp_t, hd_tm1): 
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        # If model collapses to standard RNN, or the 'reset_utterance_decoder_at_end_of_utterance' flag is off,
        # then never reset decoder. Otherwise, reset the decoder at every utterance turn.
        if (not self.collaps_to_standard_rnn) and (self.reset_utterance_decoder_at_end_of_utterance):
            hd_tm1 = (m_t) * hd_tm1 + (1 - m_t) * T.tanh(T.dot(decoder_inp_t, self.Wd_s_0) + self.bd_s_0)
  
        # In the 'selective' decoder bias type each hidden state of the decoder
        # RNN receives the decoder_inp_t modified by the selective bias -> decoder_inpr_t 
        if self.decoder_bias_type == 'selective':
            rd_sel_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_sel_e) + T.dot(hd_tm1, self.Wd_sel_h) + T.dot(decoder_inp_t, self.Wd_sel_s) + self.bd_sel)
            decoder_inpr_t = rd_sel_t * decoder_inp_t
             
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + self.bd_r)
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + self.bd_z)
            hd_tilde = self.sent_rec_activation(T.dot(xd_t, self.Wd_in) \
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh) \
                                        + T.dot(decoder_inpr_t, self.Wd_s_q) \
                                        + self.bd_hh)


            hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde 
            output = (hd_t, decoder_inpr_t, rd_sel_t, rd_t, zd_t, hd_tilde)
        
        # In the 'all' decoder bias type each hidden state of the decoder
        # RNN receives the decoder_inp_t vector as bias without modification
        elif self.decoder_bias_type == 'all':
        
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + T.dot(decoder_inp_t, self.Wd_s_r) + self.bd_r)
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + T.dot(decoder_inp_t, self.Wd_s_z) + self.bd_z)
            hd_tilde = self.sent_rec_activation(T.dot(xd_t, self.Wd_in) \
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh) \
                                        + T.dot(decoder_inp_t, self.Wd_s_q) \
                                        + self.bd_hh)
            hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde 
            output = (hd_t, rd_t, zd_t, hd_tilde)

        else:
            # Do not bias the decoder at every time, instead,
            # force it to store very useful information in the first state.
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + self.bd_r)
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + self.bd_z)
            hd_tilde = self.sent_rec_activation(T.dot(xd_t, self.Wd_in) \
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh) \
                                        + self.bd_hh) 
            hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde
            output = (hd_t, rd_t, zd_t, hd_tilde)
        return output
    
    def plain_step(self, xd_t, m_t, decoder_inp_t, hd_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        # If model collapses to standard RNN, or the 'reset_utterance_decoder_at_end_of_utterance' flag is off,
        # then never reset decoder. Otherwise, reset the decoder at every utterance turn.
        if (not self.collaps_to_standard_rnn) and (self.reset_utterance_decoder_at_end_of_utterance):
            # We already assume that xd are zeroed out
            hd_tm1 = (m_t) * hd_tm1 + (1-m_t) * T.tanh(T.dot(decoder_inp_t, self.Wd_s_0) + self.bd_s_0)

        if self.decoder_bias_type == 'first':
            # Do not bias the decoder at every time, instead,
            # force it to store very useful information in the first state.
            hd_t = self.sent_rec_activation( T.dot(xd_t, self.Wd_in) \
                                             + T.dot(hd_tm1, self.Wd_hh) \
                                             + self.bd_hh )
            output = (hd_t,)
        elif self.decoder_bias_type == 'all':
            hd_t = self.sent_rec_activation( T.dot(xd_t, self.Wd_in) \
                                             + T.dot(hd_tm1, self.Wd_hh) \
                                             + T.dot(decoder_inp_t, self.Wd_s_q) \
                                             + self.bd_hh )
            output = (hd_t,)
        elif self.decoder_bias_type == 'selective':
            rd_sel_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_sel_e) + T.dot(hd_tm1, self.Wd_sel_h) + T.dot(decoder_inp_t, self.Wd_sel_s) + self.bd_sel)
            decoder_inpr_t = rd_sel_t * decoder_inp_t
             
            hd_t = self.sent_rec_activation( T.dot(xd_t, self.Wd_in) \
                                        + T.dot(hd_tm1, self.Wd_hh) \
                                        + T.dot(decoder_inpr_t, self.Wd_s_q) \
                                        + self.bd_hh )
            output = (hd_t, decoder_inpr_t, rd_sel_t)

        return output


class DialogLevelLatentGaussianEncoder(EncoderDecoderBase):
    """
    This class operates on hidden states at the dialogue level (inter-utterance level).
    At the end of each utterance, the input from the utterance encoder(s) is transferred
    to its hidden state. This hidden state is then transformed to output a mean and a (diagonal) 
    covariance matrix, which parametrizes a latent Gaussian variable.
    """

    def init_params(self):
        """ Encoder weights """

        # Initialize input MLP
        self.input_mlp = TwoLayerMLP(self.state, self.rng, self.input_dim, self.latent_dim*2, self.latent_dim, self, '_input_mlp_'+self.name)
        self.params += self.input_mlp.params

        # Initialize mean and diagonal covariance matrix
        self.Wl_mean_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.latent_dim, self.latent_dim), name='Wl_mean_out'+self.name))
        self.bl_mean_out = add_to_params(self.params, theano.shared(value=np.zeros((self.latent_dim,), dtype='float32'), name='bl_mean_out'+self.name))

        self.Wl_std_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.latent_dim, self.latent_dim), name='Wl_std_out'+self.name))
        self.bl_std_out = add_to_params(self.params, theano.shared(value=np.zeros((self.latent_dim,), dtype='float32'), name='bl_std_out'+self.name))

    def plain_dialogue_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * h_t

        return hs_t

    def build_encoder(self, h, x, xmask=None, latent_variable_mask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True

        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            if prev_state:
                hs_0 = prev_state
            else:
                hs_0 = T.alloc(np.float32(0), batch_size, self.latent_dim)

        # sampling mode (i.e. one step)
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        if xmask.ndim == 1:
            xmask = xmask.dimshuffle(0, 'x')

        if latent_variable_mask == None:
            latent_variable_mask = T.eq(x, self.eos_sym)

        if latent_variable_mask.ndim == 1:
            latent_variable_mask = latent_variable_mask.dimshuffle(0, 'x')


        f_hier = self.plain_dialogue_step
        o_hier_info = [hs_0]

        transformed_h, updates = self.input_mlp.build_output(h, latent_variable_mask)
       

        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[transformed_h, xmask],\
                               outputs_info=o_hier_info)

        # Just one step further
        else:
            _res = f_hier(transformed_h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res

        hs_mean = T.dot(hs, self.Wl_mean_out) + self.bl_mean_out
        hs_var = T.nnet.softplus((T.dot(hs, self.Wl_std_out) + self.bl_std_out)) * self.scale_latent_gaussian_variable_variances

        hs_var = T.clip(hs_var, self.min_latent_gaussian_variable_variances, self.max_latent_gaussian_variable_variances)

        return [hs, hs_mean, hs_var], updates

    def __init__(self, state, input_dim, latent_dim, rng, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.name = name
        self.init_params()


class DialogLevelLatentPiecewiseEncoder(EncoderDecoderBase):
    """
    This class operates on hidden states at the dialogue level (inter-utterance level).
    At the end of each utterance, the input from the utterance encoder(s) is transferred
    to its hidden state. This hidden state is then transformed to output alpha vectors, which parametrize the vector of latent piecewise variables.
    """

    def init_params(self):
        """ Encoder weights """
        # Initialize input MLP
        self.input_mlp = TwoLayerMLP(self.state, self.rng, self.input_dim, self.latent_dim*2, self.latent_dim, self, '_input_mlp_'+self.name)
        self.params += self.input_mlp.params

        # Alpha output parameters
        self.Wl_alpha_out = add_to_params(self.params, theano.shared(value=NormalInit3D(self.rng, self.latent_dim, self.latent_dim, self.pieces_alpha), name='Wl_alpha_out'+self.name))
        self.bl_alpha_out = add_to_params(self.params, theano.shared(value=np.zeros((self.latent_dim, self.pieces_alpha), dtype='float32'), name='bl_alpha_out'+self.name))



    def plain_dialogue_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * h_t

        return hs_t

    def build_encoder(self, h, x, xmask=None, latent_variable_mask=None, prev_state=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True

        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            if prev_state:
                hs_0 = prev_state
            else:
                hs_0 = T.alloc(np.float32(0), batch_size, self.latent_dim)

        # sampling mode (i.e. one step)
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        if xmask.ndim == 1:
            xmask = xmask.dimshuffle(0, 'x')

        if latent_variable_mask == None:
            latent_variable_mask = T.eq(x, self.eos_sym)

        if latent_variable_mask.ndim == 1:
            latent_variable_mask = latent_variable_mask.dimshuffle(0, 'x')

        f_hier = self.plain_dialogue_step
        o_hier_info = [hs_0]

        transformed_h, updates = self.input_mlp.build_output(h, latent_variable_mask)



        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[transformed_h, xmask],\
                               outputs_info=o_hier_info)

        # Just one step further
        else:
            _res = f_hier(transformed_h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res

        hs_reshaped = hs.reshape((1,hs.shape[0],hs.shape[1],hs.shape[2]))

        hs_repeated = T.repeat(hs_reshaped, self.pieces_alpha, axis=0).reshape((self.pieces_alpha, hs.shape[0], hs.shape[1], hs.shape[2])).dimshuffle(1,2,3,0)

        hs_alpha = BatchedDot(hs_repeated, self.Wl_alpha_out, True) + self.bl_alpha_out

        # hs: time steps x batch size x hidden dim
        # hs_reshaped: time steps x batch size x hidden dim x pieces
        # Wl_alpha_out: hidden dim x latent dim x pieces
        # hs_alpha: time steps x batch size x latent dim x pieces

        if self.scale_latent_piecewise_variable_alpha_use_softplus:
            hs_alpha = T.nnet.softplus(hs_alpha)*self.scale_alpha
        else:
            hs_alpha = T.exp(hs_alpha)*self.scale_alpha

        return [hs, hs_alpha], updates

    def __init__(self, state, input_dim, latent_dim, pieces_alpha, scale_alpha, rng, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pieces_alpha = pieces_alpha
        self.scale_alpha = scale_alpha
        self.name = name
        self.init_params()



class DialogLevelRollLeft(EncoderDecoderBase):
    """
    This class operates on hidden states at the dialogue level (inter-utterance level).
    It rolls the hidden states at utterance t to be at position t-1.
    It is used for the latent variable approximate posterior, which needs to use the future h variable.
    """
    def plain_dialogue_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * h_t
        return hs_t

    def build_encoder(self, h, x, xmask=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True

        assert not one_step

        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            hs_0 = h[-1]

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_hs' in kwargs
            hs_0 = kwargs['prev_hs']

        if xmask == None:
            xmask = T.neq(x, self.eos_sym)

        f_hier = self.plain_dialogue_step
        o_hier_info = [hs_0]

        h_reversed = h[::-1]
        xmask_reversed = xmask[::-1]
        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[h_reversed, xmask_reversed],\
                               outputs_info=o_hier_info)




        # Just one step further
        else:
            _res = f_hier(h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0][::-1]
        else:
            hs = _res[::-1]

        final_hs = hs[1:(self.parent.x_max_length-1)]
        final_hs = T.concatenate([final_hs, h[-1].dimshuffle('x', 0, 1)], axis=0)

        return final_hs


    def __init__(self, state, input_dim, rng, parent):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.input_dim = input_dim

class DialogEncoderDecoder(Model):
    """
    Main model class, which links together all other sub-components
    and provides functions for training and sampling from the model.
    """

    def indices_to_words(self, seq, exclude_end_sym=True):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        def convert():
            for word_index in seq:
                if word_index > len(self.idx_to_str):
                    raise ValueError('Word index is too large for the model vocabulary!')
                if not exclude_end_sym or (word_index != self.eos_sym):
                    yield self.idx_to_str[word_index]
        return list(convert())

    def words_to_indices(self, seq):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        return [self.str_to_idx.get(word, self.unk_sym) for word in seq]

    def reverse_utterances(self, seq):
        """
        Reverses the words in each utterance inside a sequence of utterance (e.g. a dialogue)
        This is used for the bidirectional encoder RNN.
        """
        reversed_seq = numpy.copy(seq)
        for idx in range(seq.shape[1]):
            eos_indices = numpy.where(seq[:, idx] == self.eos_sym)[0]
            prev_eos_index = -1
            for eos_index in eos_indices:
                reversed_seq[(prev_eos_index+1):eos_index, idx] = (reversed_seq[(prev_eos_index+1):eos_index, idx])[::-1]
                prev_eos_index = eos_index

        return reversed_seq

    def compute_updates(self, training_cost, params):
        updates = []
         
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))

        # Gradient clipping
        c = numpy.float32(self.cutoff)
        clip_grads = []
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
         
        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))
        
        grads = OrderedDict(clip_grads)

        if self.W_emb in grads:
            if self.initialize_from_pretrained_word_embeddings and self.fix_pretrained_word_embeddings:
                assert not self.fix_encoder_parameters
                # Keep pretrained word embeddings fixed
                logger.debug("Will use mask to fix pretrained word embeddings")
                grads[self.W_emb] = grads[self.W_emb] * self.W_emb_pretrained_mask
            elif self.fix_encoder_parameters:
                # If 'fix_encoder_parameters' is on, the word embeddings will be excluded from parameter training set
                logger.debug("Will fix word embeddings to initial embeddings or embeddings from resumed model")
            else:
                logger.debug("Will train all word embeddings")

        optimizer_variables = []
        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates, optimizer_variables = Adam(grads, self.lr)
        else:
            raise Exception("Updater not understood!") 

        return updates, optimizer_variables
  
    # Batch training function.
    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building train function")

            if self.add_latent_gaussian_per_utterance and self.add_latent_piecewise_per_utterance:

                self.train_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                             self.x_max_length,
                                                             self.x_cost_mask,
                                                             self.x_reset_mask, 
                                                             self.ran_gaussian_cost_utterance,
                                                             self.ran_uniform_cost_utterance,
                                                             self.x_dropmask],
                                                outputs=[self.training_cost, self.kl_divergence_cost_acc, self.latent_gaussian_utterance_variable_approx_posterior_mean_var, self.latent_piecewise_utterance_variable_approx_posterior_alpha[-1], self.latent_piecewise_utterance_variable_prior_alpha[-1], self.kl_divergences_between_piecewise_prior_and_posterior, self.kl_divergences_between_gaussian_prior_and_posterior, self.latent_piecewise_posterior_sample],
                                                updates=self.updates + self.state_updates, 
                                                on_unused_input='warn', 
                                                name="train_fn")

            elif self.add_latent_gaussian_per_utterance:
                self.train_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                             self.x_max_length,
                                                             self.x_cost_mask,
                                                             self.x_reset_mask, 
                                                             self.ran_gaussian_cost_utterance,
                                                             self.ran_uniform_cost_utterance,
                                                             self.x_dropmask],
                                                outputs=[self.training_cost, self.kl_divergence_cost_acc, self.latent_gaussian_utterance_variable_approx_posterior_mean_var, self.kl_divergences_between_gaussian_prior_and_posterior],
                                                updates=self.updates + self.state_updates, 
                                                on_unused_input='warn', 
                                                name="train_fn")

            elif self.add_latent_piecewise_per_utterance:
                self.train_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                             self.x_max_length,
                                                             self.x_cost_mask,
                                                             self.x_reset_mask, 
                                                             self.ran_gaussian_cost_utterance,
                                                             self.ran_uniform_cost_utterance,
                                                             self.x_dropmask],
                                                outputs=[self.training_cost, self.kl_divergence_cost_acc, self.kl_divergences_between_piecewise_prior_and_posterior],
                                                updates=self.updates + self.state_updates, 
                                                on_unused_input='warn', 
                                                name="train_fn")

            else:
                self.train_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                             self.x_max_length,
                                                             self.x_cost_mask,
                                                             self.x_reset_mask, 
                                                             self.ran_gaussian_cost_utterance,
                                                             self.ran_uniform_cost_utterance,
                                                             self.x_dropmask],
                                                outputs=self.training_cost,
                                                updates=self.updates + self.state_updates, 
                                                on_unused_input='warn', 
                                                name="train_fn")                

        return self.train_fn

    def build_gamma_bounding_function(self):
        if not hasattr(self, 'gamma_bounding_fn'):
            # Compile functions
            logger.debug("Building gamma bounding function")
                
            self.gamma_bounding_fn = theano.function(inputs=[],
                                            outputs=[],
                                            updates=self.gamma_bounding_updates, 
                                            on_unused_input='warn', 
                                            name="gamma_bounding_fn")

        return self.gamma_bounding_fn

    # Helper function used for computing the initial decoder hidden states before sampling starts.
    def build_decoder_encoding(self):
        if not hasattr(self, 'decoder_encoding_fn'):
            # Compile functions
            logger.debug("Building decoder encoding function")
                
            self.decoder_encoding_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                         self.x_max_length, self.x_cost_mask,
                                                         self.x_reset_mask, 
                                                         self.ran_gaussian_cost_utterance,
                                                         self.ran_uniform_cost_utterance,
                                                         self.x_dropmask],
                                            outputs=[self.hd],
                                            on_unused_input='warn', 
                                            name="decoder_encoding_fn")

        return self.decoder_encoding_fn

    # Helper function used for the training with noise contrastive estimation (NCE).
    # This function is currently not supported.
    def build_nce_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building NCE train function")

            self.nce_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                  self.y_neg, self.x_max_length, 
                                                  self.x_cost_mask,
                                                  self.x_reset_mask,
                                                  self.ran_gaussian_cost_utterance,
                                                  self.ran_uniform_cost_utterance,
                                                  self.x_dropmask],
                                            outputs=[self.training_cost, self.kl_divergence_cost_acc, self.latent_gaussian_utterance_variable_approx_posterior_mean_var],
                                            updates=self.updates + self.state_updates, 
                                            on_unused_input='warn', 
                                            name="train_fn")

        return self.nce_fn

    # Batch evaluation function.
    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            # Compile functions
            logger.debug("Building evaluation function")

            self.eval_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_cost_mask, self.x_reset_mask, self.ran_gaussian_cost_utterance, self.ran_uniform_cost_utterance, self.x_dropmask], 
                                            outputs=[self.evaluation_cost, self.softmax_cost, self.kl_divergence_cost_acc], 
                                            updates=self.state_updates,
                                            on_unused_input='warn', name="eval_fn")



        return self.eval_fn

    # Batch mean field update function.
    def build_mf_update_function(self):
        if not hasattr(self, 'mf_update_fn'):
            # Compile functions
            logger.debug("Building mean field update function")

            mf_params = []

            if self.add_latent_gaussian_per_utterance:
                mf_params.append(self.latent_gaussian_utterance_variable_approx_posterior_mean_mfbias)
                mf_params.append(self.latent_gaussian_utterance_variable_approx_posterior_var_mfbias)

            if self.add_latent_piecewise_per_utterance:
                mf_params.append(self.latent_piecewise_utterance_variable_approx_posterior_alpha_mfbias)

            mf_updates, _ = self.compute_updates(self.training_cost, mf_params)

            if self.add_latent_gaussian_per_utterance and self.add_latent_piecewise_per_utterance:

                self.mf_update_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                             self.x_max_length,
                                                             self.x_cost_mask,
                                                             self.x_reset_mask,
                                                             self.ran_gaussian_cost_utterance,
                                                             self.ran_uniform_cost_utterance,
                                                             self.x_dropmask],
                                                outputs=[self.training_cost, self.kl_divergence_cost_acc,
                                                         self.kl_divergences_between_piecewise_prior_and_posterior,
                                                         self.kl_divergences_between_gaussian_prior_and_posterior],
                                                updates=mf_updates, 
                                                on_unused_input='warn', 
                                                name="mf_update_fn")

            elif self.add_latent_gaussian_per_utterance:
                self.mf_update_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                             self.x_max_length,
                                                             self.x_cost_mask,
                                                             self.x_reset_mask, 
                                                             self.ran_gaussian_cost_utterance,
                                                             self.ran_uniform_cost_utterance,
                                                             self.x_dropmask],
                                                outputs=[self.training_cost, self.kl_divergence_cost_acc,
                                                         self.kl_divergences_between_gaussian_prior_and_posterior],
                                                updates=mf_updates, 
                                                on_unused_input='warn', 
                                                name="mf_update_fn")

            elif self.add_latent_piecewise_per_utterance:
                self.mf_update_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, 
                                                             self.x_max_length,
                                                             self.x_cost_mask,
                                                             self.x_reset_mask, 
                                                             self.ran_gaussian_cost_utterance,
                                                             self.ran_uniform_cost_utterance,
                                                             self.x_dropmask],
                                                outputs=[self.training_cost, self.kl_divergence_cost_acc,\
                                                        self.kl_divergences_between_piecewise_prior_and_posterior],
                                                updates=mf_updates, 
                                                on_unused_input='warn', 
                                                name="mf_update_fn")
          

        return self.mf_update_fn

    def build_mf_reset_function(self):
        if not hasattr(self, 'mf_reset_fn'):
            # Compile functions
            logger.debug("Building mean field reset function")

            mf_reset_update = []

            if self.add_latent_gaussian_per_utterance:
                mf_reset_update.append((self.latent_gaussian_utterance_variable_approx_posterior_mean_mfbias, T.zeros_like(self.latent_gaussian_utterance_variable_approx_posterior_mean_mfbias)))
                mf_reset_update.append((self.latent_gaussian_utterance_variable_approx_posterior_var_mfbias, T.zeros_like(self.latent_gaussian_utterance_variable_approx_posterior_var_mfbias)))

            if self.add_latent_piecewise_per_utterance:
                mf_reset_update.append((self.latent_piecewise_utterance_variable_approx_posterior_alpha_mfbias, T.zeros_like(self.latent_piecewise_utterance_variable_approx_posterior_alpha_mfbias)))
            


            self.mf_reset_fn = theano.function(inputs=[],
                                                outputs=[],
                                                updates=mf_reset_update, 
                                                on_unused_input='warn', 
                                                name="mf_reset_fn")

        return self.mf_reset_fn

    # Batch saliency evaluation function.
    def build_saliency_eval_function(self):
        if not hasattr(self, 'saliency_eval_fn'):
            # Compile functions
            logger.debug("Building saliency evaluation function")

            training_x = self.x_data[:(self.x_max_length-1)]
            training_x_cost_mask = self.x_cost_mask[1:self.x_max_length]
            latent_variable_mask = T.eq(training_x, self.eos_sym) * training_x_cost_mask

            # Compute Gaussian KL divergence saliency:
            if self.add_latent_gaussian_per_utterance:
                kl_saliency_gaussian = \
                    T.grad(T.sum(self.kl_divergences_between_gaussian_prior_and_posterior*latent_variable_mask), self.W_emb)**2
                kl_saliency_gaussian = T.sum(kl_saliency_gaussian, axis=-1)
            else:
                kl_saliency_gaussian = T.sum(T.zeros_like(self.W_emb), axis=-1)


            # Compute Piecewise KL divergence saliency:
            if self.add_latent_piecewise_per_utterance:
                kl_saliency_piecewise = \
                    T.grad(T.sum(self.kl_divergences_between_piecewise_prior_and_posterior*latent_variable_mask), self.W_emb)**2
                kl_saliency_piecewise = T.sum(kl_saliency_piecewise, axis=-1)
            else:
                kl_saliency_piecewise = T.sum(T.zeros_like(self.W_emb), axis=-1)

            self.saliency_eval_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, self.x_max_length, self.x_cost_mask, self.x_reset_mask, self.ran_gaussian_cost_utterance, self.ran_uniform_cost_utterance, self.x_dropmask], 
                                            outputs=[kl_saliency_gaussian, kl_saliency_piecewise], 
                                            updates=self.state_updates,
                                            on_unused_input='warn', name="saliency_eval_fn")



        return self.saliency_eval_fn

    # Helper function used to compute decoder hidden states and token probabilities.
    # Currently this function does not supported truncated computations.
    def build_next_probs_function(self):
        if not hasattr(self, 'next_probs_fn'):

            if self.add_latent_gaussian_per_utterance or self.add_latent_piecewise_per_utterance:

                if self.condition_latent_variable_on_dialogue_encoder:
                    if self.direct_connection_between_encoders_and_decoder:
                        hs_to_condition_latent_variable_on = self.beam_hs.dimshuffle((0, 'x', 1))
                    else:
                        hs_to_condition_latent_variable_on = self.beam_hs.dimshuffle((0, 'x', 1))[:, :, 0:self.sdim]
                else:
                    hs_to_condition_latent_variable_on = T.alloc(np.float32(0), self.beam_hs.shape[0], 1, self.beam_hs.shape[1])[:, :, 0:self.sdim]

                if self.add_latent_gaussian_per_utterance:
                    _gaussian_prior_out, _ = self.latent_gaussian_utterance_variable_prior_encoder.build_encoder(hs_to_condition_latent_variable_on, self.beam_x_data[-1])

                    latent_gaussian_utterance_variable_prior_mean = _gaussian_prior_out[1][-1]
                    latent_gaussian_utterance_variable_prior_var = _gaussian_prior_out[2][-1]

                    prior_gaussian_sample = self.beam_ran_gaussian_cost_utterance * T.sqrt(latent_gaussian_utterance_variable_prior_var) + latent_gaussian_utterance_variable_prior_mean

                if self.add_latent_piecewise_per_utterance:
                    _piecewise_prior_out, _ = self.latent_piecewise_utterance_variable_prior_encoder.build_encoder(hs_to_condition_latent_variable_on, self.beam_x_data[-1])

                    latent_piecewise_utterance_variable_prior_alpha_hat = _piecewise_prior_out[1][-1]

                    # Apply alpha parameter trying / convolution
                    if self.latent_piecewise_variable_alpha_parameter_tying:
                        latent_piecewise_utterance_variable_prior_alpha = \
                            T.zeros_like(latent_piecewise_utterance_variable_prior_alpha_hat)

                        for i in range(1, self.latent_piecewise_alpha_variables+1):
                            normalization_constant = 0.0
                            for j in range(1, self.latent_piecewise_alpha_variables+1):
                                # Compute current alpha_hat weight
                                w = numpy.exp(-self.latent_piecewise_variable_alpha_parameter_tying_beta*(i-j)**2)

                                # Add weight to normalization constant
                                normalization_constant += w

                            normalization_constant = normalization_constant.astype('float32')

                            for j in range(1, self.latent_piecewise_alpha_variables+1):
                                # Compute normalized alpha_hat weight
                                wn = numpy.exp(-self.latent_piecewise_variable_alpha_parameter_tying_beta*(i-j)**2)\
                                    /normalization_constant
                                wn = wn.astype('float32')

                                # Add weight to alpha prior
                                latent_piecewise_utterance_variable_prior_alpha =                               \
                                 T.inc_subtensor(latent_piecewise_utterance_variable_prior_alpha[:,:,i-1],\
                                  wn*latent_piecewise_utterance_variable_prior_alpha_hat[:,:,j-1])

                    else:
                        latent_piecewise_utterance_variable_prior_alpha = \
                            latent_piecewise_utterance_variable_prior_alpha_hat




                    latent_piecewise_utterance_prior_ki = latent_piecewise_utterance_variable_prior_alpha / self.latent_piecewise_alpha_variables
                    latent_piecewise_utterance_prior_k = T.sum(latent_piecewise_utterance_prior_ki, axis=2)

                    # Sample from prior using inverse transform sampling:
                    epsilon = self.beam_ran_uniform_cost_utterance
                    prior_piecewise_sample = T.zeros_like(epsilon)
                    for i in range(1, self.latent_piecewise_alpha_variables+1):
                        lowerbound = T.zeros_like(epsilon)
                        for j in range(1, i):
                            lowerbound += (1.0/latent_piecewise_utterance_prior_k)*latent_piecewise_utterance_prior_ki[:, :,j-1]
                        upperbound = lowerbound + (1.0/latent_piecewise_utterance_prior_k)*latent_piecewise_utterance_prior_ki[:, :,i-1]
                        indicator = T.ge(epsilon, lowerbound)*T.lt(epsilon, upperbound)

                        prior_piecewise_sample += \
                              indicator*((i - 1.0)/(self.latent_piecewise_alpha_variables) \
                              + (latent_piecewise_utterance_prior_k/latent_piecewise_utterance_variable_prior_alpha[:,:,i-1])*(epsilon - lowerbound))


                    # Transform sample to be in the range [-1, 1] with initial mean at zero.
                    prior_piecewise_sample = 2.0*prior_piecewise_sample - 1.0


                if self.add_latent_gaussian_per_utterance and self.add_latent_piecewise_per_utterance:
                    if self.condition_decoder_only_on_latent_variable:
                        decoder_inp = T.concatenate([prior_gaussian_sample, prior_piecewise_sample], axis=1)
                    else:
                        decoder_inp = T.concatenate([self.beam_hs, prior_gaussian_sample, prior_piecewise_sample], axis=1)
                elif self.add_latent_gaussian_per_utterance:
                    if self.condition_decoder_only_on_latent_variable:
                        decoder_inp = prior_gaussian_sample
                    else:
                        decoder_inp = T.concatenate([self.beam_hs, prior_gaussian_sample], axis=1)
                else:
                    if self.condition_decoder_only_on_latent_variable:
                        decoder_inp = prior_piecewise_sample
                    else:
                        decoder_inp = T.concatenate([self.beam_hs, prior_piecewise_sample], axis=1)



            else:
                decoder_inp = self.beam_hs

            outputs, hd = self.utterance_decoder.build_next_probs_predictor(decoder_inp, self.beam_source, prev_state=self.beam_hd)
            self.next_probs_fn = theano.function(inputs=[self.beam_hs, self.beam_hd, self.beam_source, self.beam_x_data, self.beam_ran_gaussian_cost_utterance, self.beam_ran_uniform_cost_utterance],
                outputs=[outputs, hd],
                on_unused_input='warn',
                name="next_probs_fn")
        return self.next_probs_fn

    # Currently this function does not support truncated computations.
    # NOTE: If batch is given as input with padded endings,
    # e.g. last 'n' tokens are all zero and not part of the real sequence, 
    # then the encoding must be extracted at index of the last non-padded (non-zero) token.
    def build_encoder_function(self):
        if not hasattr(self, 'encoder_fn'):

            if self.bidirectional_utterance_encoder:
                res_forward, _, _ = self.utterance_encoder_forward.build_encoder(self.x_data)
                res_backward, _, _ = self.utterance_encoder_backward.build_encoder(self.x_data_reversed)

                # Each encoder gives a single output vector
                h = T.concatenate([res_forward, res_backward], axis=2)
            else:
                h, _, _ = self.utterance_encoder.build_encoder(self.x_data)

            hs, _ = self.dialog_encoder.build_encoder(h, self.x_data)

            if self.direct_connection_between_encoders_and_decoder:
                hs_dummy = self.dialog_dummy_encoder.build_encoder(h, self.x_data)
                hs_complete = T.concatenate([hs, hs_dummy], axis=2)
            else:
                hs_complete = hs


            if self.add_latent_gaussian_per_utterance:

                # Initialize hidden states to zero
                platent_gaussian_utterance_variable_approx_posterior = theano.shared(value=numpy.zeros((self.bs, self.latent_gaussian_per_utterance_dim), dtype='float32'), name='encoder_fn_platent_gaussian_utterance_variable_approx_posterior')

                if self.condition_posterior_latent_variable_on_dcgm_encoder:
                    platent_dcgm_avg = theano.shared(value=numpy.zeros((self.bs, self.rankdim), dtype='float32'), name='encoder_fn_platent_dcgm_avg')
                    platent_dcgm_n = theano.shared(value=numpy.zeros((1, self.bs), dtype='float32'), name='encoder_fn_platent_dcgm_n')

                # Create computational graph for latent variables
                latent_variable_mask = T.eq(self.x_data, self.eos_sym)

                if self.condition_latent_variable_on_dialogue_encoder:
                    hs_to_condition_latent_variable_on = hs_complete
                else:
                    hs_to_condition_latent_variable_on = T.alloc(np.float32(0), hs.shape[0], hs.shape[1], hs.shape[2])

                logger.debug("Initializing approximate posterior encoder for utterance-level latent variable")
                if self.bidirectional_utterance_encoder and not self.condition_posterior_latent_variable_on_dcgm_encoder:
                    posterior_latent_input_size = self.sdim + self.qdim_encoder*2
                    if self.direct_connection_between_encoders_and_decoder:
                        posterior_latent_input_size += self.qdim_encoder*2
                else:
                    posterior_latent_input_size = self.sdim + self.qdim_encoder
                    if self.direct_connection_between_encoders_and_decoder:
                        posterior_latent_input_size += self.qdim_encoder

                if self.condition_posterior_latent_variable_on_dcgm_encoder:
                    logger.debug("Build dcgm encoder")
                    latent_dcgm_res, latent_dcgm_avg, latent_dcgm_n = self.dcgm_encoder.build_encoder(self.x_data, prev_state=[platent_dcgm_avg, platent_dcgm_n])
                    h_future = self.utterance_encoder_rolledleft.build_encoder( \
                                         latent_dcgm_res, \
                                         self.x_data)

                else:
                    h_future = self.utterance_encoder_rolledleft.build_encoder( \
                                         h, \
                                         self.x_data)


                # Compute prior
                _prior_out, _ = self.latent_gaussian_utterance_variable_prior_encoder.build_encoder(hs_to_condition_latent_variable_on, self.x_data, latent_variable_mask=latent_variable_mask)

                latent_utterance_variable_prior_mean = _prior_out[1]
                latent_utterance_variable_prior_variance = _prior_out[2]

                if self.direct_connection_between_encoders_and_decoder:
                    if self.condition_decoder_only_on_latent_variable:
                        hd_input = latent_utterance_variable_prior_mean
                        hd_input_variance = latent_utterance_variable_prior_variance
                    else:
                        hd_input = T.concatenate([hs, hs_dummy, latent_utterance_variable_prior_mean], axis=2)
                        hd_input_variance = T.concatenate([T.zeros_like(hs), T.zeros_like(hs_dummy), latent_utterance_variable_prior_variance], axis=2)                       
                else:
                    if self.condition_decoder_only_on_latent_variable:
                        hd_input = latent_utterance_variable_prior_mean
                        hd_input_variance = latent_utterance_variable_prior_variance
                    else:
                        hd_input = T.concatenate([hs, latent_utterance_variable_prior_mean], axis=2)
                        hd_input_variance = T.concatenate([T.zeros_like(hs), latent_utterance_variable_prior_variance], axis=2)


                ## Compute candidate posterior
                #hs_and_h_future = T.concatenate([hs_to_condition_latent_variable_on, h_future], axis=2)

                #logger.debug("Build approximate posterior encoder for utterance-level latent variable")
                #_posterior_out, _ = self.latent_gaussian_utterance_variable_approx_posterior_encoder.build_encoder( \
                                         #hs_and_h_future, \
                                         #self.x_data, \
                                         #latent_variable_mask=latent_variable_mask)


                ## Use an MLP to interpolate between prior mean and candidate posterior mean and variance.
                #latent_utterance_variable_approx_posterior_mean = self.gaussian_posterior_mean_combination.build_output(self.hs_and_h_future, _prior_out[1], _posterior_out[1])
                #latent_utterance_variable_approx_posterior_var = self.posterior_variance_combination.build_output(self.hs_and_h_future, _prior_out[2], _posterior_out[2])

            else:
                hd_input = hs_complete
                hd_input_variance = T.zeros_like(hs_complete)

            #decoder_inp = hd_input
            #if self.deep_utterance_decoder_input:
            #    decoder_inp, _ = self.utterance_decoder.input_mlp.build_output(hd_input, T.neq(self.x_data[1:self.x_data.shape[0]], self.eos_sym))

            # TODO: Implement posterior distribution encoding of piecewise latent variables here!

            if self.add_latent_gaussian_per_utterance:
                self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, \
                             self.x_max_length], \
                             outputs=[h, hs_complete, hd_input, hd_input_variance], on_unused_input='warn', name="encoder_fn")
                #self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, \
                #             self.x_max_length], \
                #             outputs=[h, hs_complete, hs_and_h_future, latent_utterance_variable_approx_posterior_mean], on_unused_input='warn', name="encoder_fn")
            else:
                self.encoder_fn = theano.function(inputs=[self.x_data, self.x_data_reversed, \
                             self.x_max_length], \
                             outputs=[h, hs_complete, hd_input, hd_input_variance], on_unused_input='warn', name="encoder_fn")


        return self.encoder_fn


    def compute_utterance_embeddings(self, utterances):
        # Build encoder function if it doesn't already exist
        if not hasattr(self, 'encoder_fn'):
            self.build_encoder_function()

        maxlen = 1
        for utterance_id in range(len(utterances)):
            words = utterances[utterance_id].split()
            words_count = len(words)
            if len(words) > 0:
                if not words[0] == self.end_sym_utterance:
                    utterances[utterance_id] = (self.end_sym_utterance + ' ' + utterances[utterance_id]).replace('  ', ' ')
                    words_count += 1
                if not words[-1] == self.end_sym_utterance:
                    utterances[utterance_id] = (utterances[utterance_id] + ' ' + self.end_sym_utterance).replace('  ', ' ')
                    words_count += 1

            maxlen = max(maxlen, words_count)

        maxlen = min(maxlen, self.max_len)
        dialogue = numpy.zeros((maxlen, len(utterances)), dtype='int32')
        dialogue_eos_indices = []
        for utterance_id in range(len(utterances)):
            word_ids = self.words_to_indices(utterances[utterance_id].split())
            if word_ids > maxlen:
                word_ids = word_ids[-maxlen:]

            dialogue[0:len(word_ids), utterance_id] = word_ids
            dialogue_eos_indices.append(len(word_ids)-1)

        dialogue_reversed = self.reverse_utterances(dialogue)

        full_embeddings = self.encoder_fn(dialogue, dialogue_reversed, dialogue.shape[0])

        # Use utterance encoder
        full_embeddings = full_embeddings[0]

        # Use transformed input to decoder
        #full_embeddings = full_embeddings[2]       
       
        embeddings = numpy.zeros((full_embeddings.shape[1], full_embeddings.shape[2]), dtype='float32')
        for utterance_id in range(len(utterances)):
            embeddings[utterance_id, :] = full_embeddings[dialogue_eos_indices[utterance_id], utterance_id, :]
        
        normalized_embeddings = (embeddings.T / numpy.linalg.norm(embeddings, axis=1)).T

        return normalized_embeddings

    def compute_utterance_embeddings_from_list(self, utterances):
        # Compute embedding size embeddings
        # Use utterance encoder
        if True:
            if self.bidirectional_utterance_encoder:
                embedding_dim = self.qdim_encoder*2
            else:
                embedding_dim = self.qdim_encoder
                
        # Use transformed input to decoder
        if False:
            embedding_dim = self.utterance_decoder.input_dim

        # Compute utterance embeddings
        utterance_embeddings = numpy.zeros((len(utterances), embedding_dim), dtype='float32')
        last_utterance_id_computed = 0
        utterances_to_compute = []
        for utterance_id in range(len(utterances)):
            utterances_to_compute.append(utterances[utterance_id])

            if (len(utterances_to_compute) == self.bs) or (utterance_id+1 == len(utterances)):
                print('utterance_id', utterance_id)

                computed_emb = self.compute_utterance_embeddings(utterances_to_compute)
                utterance_embeddings[last_utterance_id_computed:last_utterance_id_computed+computed_emb.shape[0], :] = computed_emb[:, :]
                last_utterance_id_computed = utterance_id+1
                utterances_to_compute = []

        return utterance_embeddings

    def compute_utterance_embeddings_with_variance(self, utterances):
        # Build encoder function if it doesn't already exist
        if not hasattr(self, 'encoder_fn'):
            self.build_encoder_function()

        maxlen = 1
        for utterance_id in range(len(utterances)):
            words = utterances[utterance_id].split()
            words_count = len(words)
            if len(words) > 0:
                if not words[0] == self.end_sym_utterance:
                    utterances[utterance_id] = (self.end_sym_utterance + ' ' + utterances[utterance_id]).replace('  ', ' ')
                    words_count += 1
                if not words[-1] == self.end_sym_utterance:
                    utterances[utterance_id] = (utterances[utterance_id] + ' ' + self.end_sym_utterance).replace('  ', ' ')
                    words_count += 1

            maxlen = max(maxlen, words_count)

        maxlen = min(maxlen, self.max_len)
        dialogue = numpy.zeros((maxlen, len(utterances)), dtype='int32')
        dialogue_eos_indices = []
        for utterance_id in range(len(utterances)):
            word_ids = self.words_to_indices(utterances[utterance_id].split())
            if word_ids > maxlen:
                word_ids = word_ids[-maxlen:]

            dialogue[0:len(word_ids), utterance_id] = word_ids
            dialogue_eos_indices.append(len(word_ids)-1)

        dialogue_reversed = self.reverse_utterances(dialogue)

        full_embeddings = self.encoder_fn(dialogue, dialogue_reversed, dialogue.shape[0])

        # Use transformed input to decoder
        full_embeddings_mean = full_embeddings[2]       
        full_embeddings_var = full_embeddings[3]       

        embeddings = numpy.zeros((full_embeddings_mean.shape[1], full_embeddings_mean.shape[2]), dtype='float32')
        embeddings_var = numpy.zeros((full_embeddings_mean.shape[1], full_embeddings_mean.shape[2]), dtype='float32')
        for utterance_id in range(len(utterances)):
            embeddings[utterance_id, :] = full_embeddings_mean[dialogue_eos_indices[utterance_id], utterance_id, :]
            embeddings_var[utterance_id, :] = full_embeddings_var[dialogue_eos_indices[utterance_id], utterance_id, :]

        return embeddings, embeddings_var

    def compute_utterance_embeddings_with_variance_from_list(self, utterances):
        # Compute embedding size embeddings
        # Use utterance encoder
        if self.bidirectional_utterance_encoder:
            embedding_dim = self.qdim_encoder*2 + self.sdim
        else:
            embedding_dim = self.qdim_encoder + self.sdim

        if self.add_latent_gaussian_per_utterance:
            embedding_dim += self.latent_gaussian_per_utterance_dim

        # Compute utterance embeddings
        utterance_embeddings = numpy.zeros((len(utterances), embedding_dim), dtype='float32')
        utterance_variance_embeddings = numpy.zeros((len(utterances), embedding_dim), dtype='float32')
        last_utterance_id_computed = 0
        utterances_to_compute = []
        for utterance_id in range(len(utterances)):
            utterances_to_compute.append(utterances[utterance_id])

            if (len(utterances_to_compute) == self.bs) or (utterance_id+1 == len(utterances)):
                print('utterance_id', utterance_id)

                computed_emb, computed_emb_variance = self.compute_utterance_embeddings_with_variance(utterances_to_compute)
                utterance_embeddings[last_utterance_id_computed:last_utterance_id_computed+computed_emb.shape[0], :] = computed_emb[:, :]
                utterance_variance_embeddings[last_utterance_id_computed:last_utterance_id_computed+computed_emb_variance.shape[0], :] = computed_emb_variance[:, :]

                last_utterance_id_computed = utterance_id+1
                utterances_to_compute = []

        # Remove useless sdim values
        utterance_embeddings = utterance_embeddings[:, self.sdim:]
        utterance_variance_embeddings = utterance_variance_embeddings[:, self.sdim:]

        return utterance_embeddings, utterance_variance_embeddings

    def __init__(self, state):
        Model.__init__(self)

        # Make sure eos_sym is never zero, otherwise generate_encodings script would fail
        assert state['eos_sym'] > 0

        if not 'bidirectional_utterance_encoder' in state:
            state['bidirectional_utterance_encoder'] = False

        if 'encode_with_l2_pooling' in state:
            assert state['encode_with_l2_pooling'] == False # We don't support L2 pooling right now...

        if not 'direct_connection_between_encoders_and_decoder' in state:
            state['direct_connection_between_encoders_and_decoder'] = False

        if not 'deep_direct_connection' in state:
            state['deep_direct_connection'] = False

        if not 'disable_dialogue_encoder' in state:
            state['disable_dialogue_encoder'] = False

        if state['disable_dialogue_encoder']:
            # We can only disable the dialoge encoder, if the utterance encoder hidden state
            # is given as input to the decoder directly.
            assert state['direct_connection_between_encoders_and_decoder']

        if not state['direct_connection_between_encoders_and_decoder']:
            assert(state['deep_direct_connection'] == False)

        if not 'collaps_to_standard_rnn' in state:
            state['collaps_to_standard_rnn'] = False

        if not 'reset_utterance_decoder_at_end_of_utterance' in state:
            state['reset_utterance_decoder_at_end_of_utterance'] = True

        if not 'reset_utterance_encoder_at_end_of_utterance' in state:
            state['reset_utterance_encoder_at_end_of_utterance'] = False
        else:
            assert state['reset_utterance_encoder_at_end_of_utterance'] == False

        if not 'deep_dialogue_encoder_input' in state:
            state['deep_dialogue_encoder_input'] = True

        if not 'deep_utterance_decoder_input' in state:
            state['deep_utterance_decoder_input'] = False

        if not 'reset_hidden_states_between_subsequences' in state:
            state['reset_hidden_states_between_subsequences'] = False

        if not 'fix_encoder_parameters' in state:
            state['fix_encoder_parameters'] = False

        if not 'decoder_drop_previous_input_tokens' in state:
            state['decoder_drop_previous_input_tokens'] = False
        else:
            if state['decoder_drop_previous_input_tokens']:
                assert state['decoder_drop_previous_input_tokens_rate']

        if not 'add_latent_gaussian_per_utterance' in state:
           state['add_latent_gaussian_per_utterance'] = False
        if not 'latent_gaussian_per_utterance_dim' in state:
           state['latent_gaussian_per_utterance_dim'] = 1
        if not 'condition_latent_variable_on_dialogue_encoder' in state:
           state['condition_latent_variable_on_dialogue_encoder'] = True
        if not 'condition_posterior_latent_variable_on_dcgm_encoder' in state:
           state['condition_posterior_latent_variable_on_dcgm_encoder'] = False
        if not 'scale_latent_gaussian_variable_variances' in state:
           state['scale_latent_gaussian_variable_variances'] = 0.01
        if not 'condition_decoder_only_on_latent_variable' in state:
           state['condition_decoder_only_on_latent_variable'] = False

        if not 'train_latent_variables_with_kl_divergence_annealing' in state:
           state['train_latent_variables_with_kl_divergence_annealing'] = False
        if state['train_latent_variables_with_kl_divergence_annealing']:
            assert 'kl_divergence_annealing_rate' in state

        if not 'kl_divergence_max_weight' in state:
            state['kl_divergence_max_weight'] = 1.0


        if not 'add_latent_piecewise_per_utterance' in state:
            state['add_latent_piecewise_per_utterance'] = False
        if not 'gate_latent_piecewise_per_utterance' in state:
            state['gate_latent_piecewise_per_utterance'] = True

        if not 'constraint_latent_piecewise_variable_posterior' in state:
            state['constraint_latent_piecewise_variable_posterior'] = True
        if not 'scale_latent_piecewise_variable_prior_alpha' in state:
            state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
        if not 'scale_latent_piecewise_variable_posterior_alpha' in state:
            state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0
        if not 'scale_latent_piecewise_variable_alpha_use_softplus' in state:
            state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
        if not 'latent_piecewise_variable_alpha_parameter_tying' in state:
            state['latent_piecewise_variable_alpha_parameter_tying'] = False

        if not 'apply_meanfield_inference' in state:
            state['apply_meanfield_inference'] = False

        if state['collaps_to_standard_rnn']:
            # If we collapse to standard RNN (e.g. LSTM language model) then we should not reset.
            # If we did reset, we'd have a language model over individual utterances, which is what we want!
            assert not state['reset_utterance_decoder_at_end_of_utterance']

        if not 'compute_training_updates' in state:
            state['compute_training_updates'] = True

        self.state = state
        self.global_params = []

        self.__dict__.update(state)
        self.rng = numpy.random.RandomState(state['seed']) 

        # Load dictionary
        raw_dict = cPickle.load(open(self.dictionary, 'r'))

        # Probabilities for each term in the corpus used for noise contrastive estimation (NCE)
        self.noise_probs = [x[2] for x in sorted(raw_dict, key=operator.itemgetter(1))]
        self.noise_probs = numpy.array(self.noise_probs, dtype='float64')
        self.noise_probs /= numpy.sum(self.noise_probs)
        self.noise_probs = self.noise_probs ** 0.75
        self.noise_probs /= numpy.sum(self.noise_probs)
        
        self.t_noise_probs = theano.shared(self.noise_probs.astype('float32'), 't_noise_probs')

        # Dictionaries to convert str to idx and vice-versa
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict])
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in raw_dict])

        # Extract document (dialogue) frequency for each word
        self.word_freq = dict([(tok_id, freq) for _, tok_id, freq, _ in raw_dict])
        self.document_freq = dict([(tok_id, df) for _, tok_id, _, df in raw_dict])

        if self.end_sym_utterance not in self.str_to_idx:
           raise Exception("Error, malformed dictionary!")
         
        # Number of words in the dictionary 
        self.idim = len(self.str_to_idx)
        self.state['idim'] = self.idim
        logger.debug("idim: " + str(self.idim))

        logger.debug("Initializing Theano variables")
        self.y_neg = T.itensor3('y_neg')
        self.x_data = T.imatrix('x_data')
        self.x_data_reversed = T.imatrix('x_data_reversed')
        self.x_cost_mask = T.matrix('cost_mask')
        self.x_reset_mask = T.vector('reset_mask')
        self.x_max_length = T.iscalar('x_max_length')
        self.ran_gaussian_cost_utterance = T.tensor3('ran_gaussian_cost_utterance')
        self.ran_uniform_cost_utterance = T.tensor3('ran_uniform_cost_utterance')
        self.x_dropmask = T.matrix('x_dropmask')


        
        # The 'x' data (input) is defined as all symbols except the last, and
        # the 'y' data (output) is defined as all symbols except the first.
        training_x = self.x_data[:(self.x_max_length-1)]
        training_x_reversed = self.x_data_reversed[:(self.x_max_length-1)]
        training_y = self.x_data[1:self.x_max_length]
        training_x_dropmask = self.x_dropmask[:(self.x_max_length-1)]

        # Here we find the end-of-utterance tokens in the minibatch.
        training_hs_mask = T.neq(training_x, self.eos_sym)
        training_x_cost_mask = self.x_cost_mask[1:self.x_max_length]
        training_x_cost_mask_flat = training_x_cost_mask.flatten()
        
        # Backward compatibility
        if 'decoder_bias_type' in self.state:
            logger.debug("Decoder bias type {}".format(self.decoder_bias_type))


        # Build word embeddings, which are shared throughout the model
        if self.initialize_from_pretrained_word_embeddings == True:
            # Load pretrained word embeddings from pickled file
            logger.debug("Loading pretrained word embeddings")
            pretrained_embeddings = cPickle.load(open(self.pretrained_word_embeddings_file, 'r'))

            # Check all dimensions match from the pretrained embeddings
            assert(self.idim == pretrained_embeddings[0].shape[0])
            assert(self.rankdim == pretrained_embeddings[0].shape[1])
            assert(self.idim == pretrained_embeddings[1].shape[0])
            assert(self.rankdim == pretrained_embeddings[1].shape[1])

            self.W_emb_pretrained_mask = theano.shared(pretrained_embeddings[1].astype(numpy.float32), name='W_emb_mask')
            self.W_emb = add_to_params(self.global_params, theano.shared(value=pretrained_embeddings[0].astype(numpy.float32), name='W_emb'))
        else:
            # Initialize word embeddings randomly
            self.W_emb = add_to_params(self.global_params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))

        # Variables to store encoder and decoder states
        if self.bidirectional_utterance_encoder:
            # Previous states variables
            self.ph_fwd = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'), name='ph_fwd')
            self.ph_fwd_n = theano.shared(value=numpy.zeros((1, self.bs), dtype='int8'), name='ph_fwd_n')

            self.ph_bck = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'), name='ph_bck')
            self.ph_bck_n = theano.shared(value=numpy.zeros((1, self.bs), dtype='int8'), name='ph_bck_n')

            self.phs = theano.shared(value=numpy.zeros((self.bs, self.sdim), dtype='float32'), name='phs')

            if self.direct_connection_between_encoders_and_decoder:
                self.phs_dummy = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder*2), dtype='float32'), name='phs_dummy')

        else:
            # Previous states variables
            self.ph = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'), name='ph')
            self.ph_n = theano.shared(value=numpy.zeros((1, self.bs), dtype='int8'), name='ph_n')

            self.phs = theano.shared(value=numpy.zeros((self.bs, self.sdim), dtype='float32'), name='phs')

            if self.direct_connection_between_encoders_and_decoder:
                self.phs_dummy = theano.shared(value=numpy.zeros((self.bs, self.qdim_encoder), dtype='float32'), name='phs_dummy')

        if self.utterance_decoder_gating == 'LSTM':
            self.phd = theano.shared(value=numpy.zeros((self.bs, self.qdim_decoder*2), dtype='float32'), name='phd')
        else:
            self.phd = theano.shared(value=numpy.zeros((self.bs, self.qdim_decoder), dtype='float32'), name='phd')

        if self.add_latent_gaussian_per_utterance:
            self.platent_gaussian_utterance_variable_prior = theano.shared(value=numpy.zeros((self.bs, self.latent_gaussian_per_utterance_dim), dtype='float32'), name='platent_gaussian_utterance_variable_prior')
            self.platent_gaussian_utterance_variable_approx_posterior = theano.shared(value=numpy.zeros((self.bs, self.latent_gaussian_per_utterance_dim), dtype='float32'), name='platent_gaussian_utterance_variable_approx_posterior')

        if self.add_latent_piecewise_per_utterance:
            self.platent_piecewise_utterance_variable_prior = theano.shared(value=numpy.zeros((self.bs, self.latent_piecewise_per_utterance_dim), dtype='float32'), name='platent_piecewise_utterance_variable_prior')
            self.platent_piecewise_utterance_variable_approx_posterior = theano.shared(value=numpy.zeros((self.bs, self.latent_piecewise_per_utterance_dim), dtype='float32'), name='platent_piecewise_utterance_variable_approx_posterior')

        if self.add_latent_gaussian_per_utterance or self.add_latent_piecewise_per_utterance:
            if self.condition_posterior_latent_variable_on_dcgm_encoder:
                self.platent_dcgm_avg = theano.shared(value=numpy.zeros((self.bs, self.rankdim), dtype='float32'), name='platent_dcgm_avg')
                self.platent_dcgm_n = theano.shared(value=numpy.zeros((1, self.bs), dtype='float32'), name='platent_dcgm_n')


        # Build utterance encoders
        if self.bidirectional_utterance_encoder:
            logger.debug("Initializing forward utterance encoder")
            self.utterance_encoder_forward = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')
            logger.debug("Build forward utterance encoder")
            res_forward, res_forward_n, res_forward_updates = self.utterance_encoder_forward.build_encoder(training_x, xmask=training_hs_mask, prev_state=[self.ph_fwd, self.ph_fwd_n])

            logger.debug("Initializing backward utterance encoder")
            self.utterance_encoder_backward = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'bck')
            logger.debug("Build backward utterance encoder")
            res_backward, res_backward_n, res_backward_updates = self.utterance_encoder_backward.build_encoder(training_x_reversed, xmask=training_hs_mask, prev_state=[self.ph_bck, self.ph_bck_n])

            # The encoder h embedding is a concatenation of final states of the forward and backward encoder RNNs
            self.h = T.concatenate([res_forward, res_backward], axis=2)

        else:
            logger.debug("Initializing utterance encoder")
            self.utterance_encoder = UtteranceEncoder(self.state, self.rng, self.W_emb, self, 'fwd')

            logger.debug("Build utterance encoder")

            # The encoder h embedding is the final hidden state of the forward encoder RNN
            res_forward, res_forward_n, res_forward_updates = self.utterance_encoder.build_encoder(training_x, xmask=training_hs_mask, prev_state=[self.ph, self.ph_n])

            self.h = res_forward


        logger.debug("Initializing dialog encoder")
        self.dialog_encoder = DialogEncoder(self.state, self.rng, self, '_dialogue_encoder')

        logger.debug("Build dialog encoder")
        self.hs, self.dialogue_encoder_updates = self.dialog_encoder.build_encoder(self.h, training_x, xmask=training_hs_mask, prev_state=self.phs)

        # Define input vector for decoder
        if self.direct_connection_between_encoders_and_decoder:
            logger.debug("Initializing dialog dummy encoder")
            if self.bidirectional_utterance_encoder:
                self.dialog_dummy_encoder = DialogDummyEncoder(self.state, self.rng, self, self.qdim_encoder*2)
            else:
                self.dialog_dummy_encoder = DialogDummyEncoder(self.state, self.rng, self, self.qdim_encoder)

            logger.debug("Build dialog dummy encoder")
            self.hs_dummy = self.dialog_dummy_encoder.build_encoder(self.h, training_x, xmask=training_hs_mask, prev_state=self.phs_dummy)



        # Compute quantities necessary for handling latent variables
        if self.add_latent_gaussian_per_utterance or self.add_latent_piecewise_per_utterance:
            # Define list storing variable updates related to latent modules
            self.latent_variable_updates = []

            # Define KL divergence cost
            self.kl_divergence_cost = training_x_cost_mask*0

            # Compute mask over latent variables. 
            # One means that a variable is part of the computational graph and zero that it's not.
            latent_variable_mask = T.eq(training_x, self.eos_sym) * training_x_cost_mask

            # We consider two kinds of prior: one case where the latent variables are 
            # conditioned on the dialogue encoder, and one case where they are not conditioned on anything.
            if self.condition_latent_variable_on_dialogue_encoder:
                if self.direct_connection_between_encoders_and_decoder:
                    self.hs_to_condition_latent_variable_on = T.concatenate([self.hs, self.hs_dummy], axis=2)
                    if self.bidirectional_utterance_encoder:
                        prior_latent_input_size = self.sdim + self.qdim_encoder*2
                    else:
                        prior_latent_input_size = self.sdim + self.qdim_encoder
                else:
                    self.hs_to_condition_latent_variable_on = self.hs
                    prior_latent_input_size = self.sdim
            else:
                self.hs_to_condition_latent_variable_on = T.alloc(np.float32(0), self.hs.shape[0], self.hs.shape[1], self.hs.shape[2])
                prior_latent_input_size = self.sdim


            if self.bidirectional_utterance_encoder and not self.condition_posterior_latent_variable_on_dcgm_encoder:
                posterior_latent_input_size = prior_latent_input_size + self.qdim_encoder*2
            else:
                posterior_latent_input_size = prior_latent_input_size + self.qdim_encoder

            # Retrieve hidden state at the end of next utterance from the utterance encoders
            # (or at the end of the batch, if there are no end-of-token symbols at the end of the batch)
            if self.bidirectional_utterance_encoder:
                self.utterance_encoder_rolledleft = DialogLevelRollLeft(self.state, self.qdim_encoder, self.rng, self)
            else:
                self.utterance_encoder_rolledleft = DialogLevelRollLeft(self.state, self.qdim_encoder*2, self.rng, self)

            if self.condition_posterior_latent_variable_on_dcgm_encoder:
                logger.debug("Initializing DCGM encoder for conditioning input to the utterance-level latent variable")

                self.dcgm_encoder = DCGMEncoder(self.state, self.rng, self.W_emb, self.qdim_encoder, self, 'latent_dcgm_encoder')
                logger.debug("Build DCGM encoder")
                latent_dcgm_res, self.latent_dcgm_avg, self.latent_dcgm_n = self.dcgm_encoder.build_encoder(training_x, xmask=training_hs_mask, prev_state=[self.platent_dcgm_avg, self.platent_dcgm_n])

                self.h_future = self.utterance_encoder_rolledleft.build_encoder( \
                                     latent_dcgm_res, \
                                     training_x, \
                                     xmask=training_hs_mask)
            else:
                self.h_future = self.utterance_encoder_rolledleft.build_encoder( \
                                     self.h, \
                                     training_x, \
                                     xmask=training_hs_mask)

            self.hs_and_h_future = T.concatenate([self.hs_to_condition_latent_variable_on, self.h_future], axis=2)









        # We initialize the multivariate Gaussian latent variables
        if self.add_latent_gaussian_per_utterance:
            logger.debug("Initializing prior encoder for utterance-level latent multivariate Gaussian variables")

            self.latent_gaussian_utterance_variable_prior_encoder = DialogLevelLatentGaussianEncoder(self.state, prior_latent_input_size, self.latent_gaussian_per_utterance_dim, self.rng, self, 'latent_gaussian_utterance_prior')

            logger.debug("Build prior encoder for utterance-level latent multivariate Gaussian variables")
            _prior_gaussian_out, _prior_gaussian_updates = self.latent_gaussian_utterance_variable_prior_encoder.build_encoder(self.hs_to_condition_latent_variable_on, training_x, xmask=training_hs_mask, latent_variable_mask=latent_variable_mask, prev_state=self.platent_gaussian_utterance_variable_prior)
            self.latent_variable_updates += _prior_gaussian_updates

            self.latent_gaussian_utterance_variable_prior = _prior_gaussian_out[0]
            self.latent_gaussian_utterance_variable_prior_mean = _prior_gaussian_out[1]
            self.latent_gaussian_utterance_variable_prior_var = _prior_gaussian_out[2]

            self.latent_gaussian_utterance_variable_approx_posterior_encoder = DialogLevelLatentGaussianEncoder(self.state, posterior_latent_input_size, self.latent_gaussian_per_utterance_dim, self.rng, self, 'latent_gaussian_utterance_approx_posterior')

            logger.debug("Build approximate posterior encoder for utterance-level latent multivariate Gaussian variables")
            _posterior_gaussian_out, _posterior_gaussian_updates =                                  \
                    self.latent_gaussian_utterance_variable_approx_posterior_encoder.build_encoder( \
                                     self.hs_and_h_future,                                          \
                                     training_x,                                                    \
                                     xmask=training_hs_mask,                                        \
                                     latent_variable_mask=latent_variable_mask,                     \
                                     prev_state=self.platent_gaussian_utterance_variable_approx_posterior)
            self.latent_variable_updates += _posterior_gaussian_updates

            self.latent_gaussian_utterance_variable_approx_posterior = _posterior_gaussian_out[0]

            # Use an MLP to interpolate between prior mean and candidate posterior mean.
            # This allows model to revert back to prior easily for dimensions, where it is uncertain.
            self.gaussian_posterior_mean_combination = LinearCombination(self.state, posterior_latent_input_size, self.latent_gaussian_per_utterance_dim, False, 0.0, 0.0, self.rng, self, 'latent_gaussian_utterance_approx_posterior_mean_combination')
            self.latent_gaussian_utterance_variable_approx_posterior_mean = self.gaussian_posterior_mean_combination.build_output(self.hs_and_h_future, self.latent_gaussian_utterance_variable_prior_mean, _posterior_gaussian_out[1])


            # Use an MLP to interpolate between prior variance and candidate posterior variance.
            # This allows model to revert back to prior easily for dimensions, where it is uncertain.
            self.posterior_variance_combination = LinearCombination(self.state, posterior_latent_input_size, self.latent_gaussian_per_utterance_dim, True, self.min_latent_gaussian_variable_variances, self.max_latent_gaussian_variable_variances, self.rng, self, 'latent_gaussian_utterance_approx_posterior_variance_combination')
            self.latent_gaussian_utterance_variable_approx_posterior_var = self.posterior_variance_combination.build_output(self.hs_and_h_future, self.latent_gaussian_utterance_variable_prior_var, _posterior_gaussian_out[2])


            # Apply mean-field inference?
            if self.apply_meanfield_inference:
                self.latent_gaussian_utterance_variable_approx_posterior_mean_mfbias = \
                    theano.shared(value=numpy.zeros((self.bs, self.latent_gaussian_per_utterance_dim), dtype='float32'), name='latent_gaussian_utterance_variable_approx_posterior_mean_mfbias')
                self.latent_gaussian_utterance_variable_approx_posterior_var_mfbias = \
                    theano.shared(value=numpy.zeros((self.bs, self.latent_gaussian_per_utterance_dim), dtype='float32'), name='latent_gaussian_utterance_variable_approx_posterior_var_mfbias')

                self.latent_gaussian_utterance_variable_approx_posterior_mean += \
                    self.latent_gaussian_utterance_variable_approx_posterior_mean_mfbias.dimshuffle('x', 0, 1)

                self.latent_gaussian_utterance_variable_approx_posterior_var += \
                    T.maximum(self.latent_gaussian_utterance_variable_approx_posterior_var_mfbias.dimshuffle('x', 0, 1), - self.latent_gaussian_utterance_variable_approx_posterior_var + 0.000001)




            self.latent_gaussian_utterance_variable_approx_posterior_mean_var = T.sum(T.mean(self.latent_gaussian_utterance_variable_approx_posterior_var,axis=2)*latent_variable_mask) / (T.sum(latent_variable_mask) + 0.0000001)

            # Sample utterance latent variable from posterior
            self.latent_gaussian_posterior_sample = self.ran_gaussian_cost_utterance[:(self.x_max_length-1)] * T.sqrt(self.latent_gaussian_utterance_variable_approx_posterior_var) + self.latent_gaussian_utterance_variable_approx_posterior_mean

            # Compute KL divergence cost
            mean_diff_squared = (self.latent_gaussian_utterance_variable_prior_mean \
                                 - self.latent_gaussian_utterance_variable_approx_posterior_mean)**2

            logger.debug("Build KL divergence cost for latent multivariate Gaussian variables")
            #self.kl_divergences_between_gaussian_prior_and_posterior                                      \
            #    = T.maximum(0.0, (T.sum(self.latent_gaussian_utterance_variable_approx_posterior_var      \
            #                            /self.latent_gaussian_utterance_variable_prior_var, axis=2)       \
            #       + T.sum(mean_diff_squared/self.latent_gaussian_utterance_variable_prior_var, axis=2)   \
            #       - state['latent_gaussian_per_utterance_dim']                                           \
            #       + T.sum(T.log(self.latent_gaussian_utterance_variable_prior_var), axis=2)              \
            #       - T.sum(T.log(self.latent_gaussian_utterance_variable_approx_posterior_var), axis=2)   \
            #      ) / 2)

            # Numerically stable without truncation at zero
            self.kl_divergences_between_gaussian_prior_and_posterior                                      \
                = (T.sum(self.latent_gaussian_utterance_variable_approx_posterior_var      \
                                        /self.latent_gaussian_utterance_variable_prior_var, axis=2)       \
                   + T.sum(mean_diff_squared/self.latent_gaussian_utterance_variable_prior_var, axis=2)   \
                   - state['latent_gaussian_per_utterance_dim']                                           \
                   + T.sum(T.log(self.latent_gaussian_utterance_variable_prior_var), axis=2)              \
                   - T.sum(T.log(self.latent_gaussian_utterance_variable_approx_posterior_var), axis=2))/2



            self.kl_divergence_cost += self.kl_divergences_between_gaussian_prior_and_posterior*latent_variable_mask

        else:
            self.latent_gaussian_utterance_variable_approx_posterior_mean_var = theano.shared(value=numpy.float(0))











        # We initialize the stochastic latent variables
        # platent_piecewise_utterance_variable_prior
        if self.add_latent_piecewise_per_utterance:
            # Compute prior
            logger.debug("Initializing prior encoder for utterance-level latent piecewise variables")
            self.latent_piecewise_utterance_variable_prior_encoder = DialogLevelLatentPiecewiseEncoder(self.state, prior_latent_input_size, self.latent_piecewise_per_utterance_dim, self.latent_piecewise_alpha_variables, self.scale_latent_piecewise_variable_prior_alpha, self.rng, self, 'latent_piecewise_utterance_prior')

            logger.debug("Build prior encoder for utterance-level latent piecewise variables")
            _prior_piecewise_out, _prior_piecewise_updates = self.latent_piecewise_utterance_variable_prior_encoder.build_encoder(self.hs_to_condition_latent_variable_on, training_x, xmask=training_hs_mask, latent_variable_mask=latent_variable_mask, prev_state=self.platent_piecewise_utterance_variable_prior)
            self.latent_variable_updates += _prior_piecewise_updates

            self.latent_piecewise_utterance_variable_prior = _prior_piecewise_out[0]
            self.latent_piecewise_utterance_variable_prior_alpha_hat = _prior_piecewise_out[1]


            # Compute posterior using prior
            logger.debug("Initializing approximate posterior encoder for utterance-level latent piecewise variables")
            self.latent_piecewise_utterance_variable_approx_posterior_encoder = DialogLevelLatentPiecewiseEncoder(self.state, posterior_latent_input_size, self.latent_piecewise_per_utterance_dim, self.latent_piecewise_alpha_variables, self.scale_latent_piecewise_variable_posterior_alpha, self.rng, self, 'latent_piecewise_utterance_approx_posterior')

            logger.debug("Build approximate posterior encoder for utterance-level latent piecewise variables")
            _posterior_piecewise_out, _posterior_piecewise_updates =                                    \
                     self.latent_piecewise_utterance_variable_approx_posterior_encoder.build_encoder( \
                                     self.hs_and_h_future, \
                                     training_x, \
                                     xmask=training_hs_mask, \
                                     latent_variable_mask=latent_variable_mask, \
                                     prev_state=self.platent_piecewise_utterance_variable_approx_posterior)
            self.latent_variable_updates += _posterior_piecewise_updates

            self.latent_piecewise_utterance_variable_approx_posterior = _posterior_piecewise_out[0]

            # Apply gating mechanism for linear interpolation
            if self.gate_latent_piecewise_per_utterance:
                self.piecewise_posterior_mean_combination = LinearCombination(self.state, posterior_latent_input_size, self.latent_piecewise_per_utterance_dim, False, 0.0, 0.0, self.rng, self, 'latent_piecewise_utterance_approx_posterior_alpha_combination')
                self.latent_piecewise_utterance_variable_approx_posterior_alpha_hat = self.piecewise_posterior_mean_combination.build_output(self.hs_and_h_future, self.latent_piecewise_utterance_variable_prior_alpha_hat.dimshuffle(0, 1, 3, 2), _posterior_piecewise_out[1].dimshuffle(0, 1, 3, 2)).dimshuffle(0, 1, 3, 2)
            else:
                self.latent_piecewise_utterance_variable_approx_posterior_alpha_hat = _posterior_piecewise_out[1]


            # Apply alpha parameter trying / convolution
            if self.latent_piecewise_variable_alpha_parameter_tying:
                self.latent_piecewise_utterance_variable_prior_alpha = \
                    T.zeros_like(self.latent_piecewise_utterance_variable_prior_alpha_hat)
                self.latent_piecewise_utterance_variable_approx_posterior_alpha = \
                    T.zeros_like(self.latent_piecewise_utterance_variable_approx_posterior_alpha_hat)

                for i in range(1, self.latent_piecewise_alpha_variables+1):
                    normalization_constant = 0.0
                    for j in range(1, self.latent_piecewise_alpha_variables+1):
                        # Compute current alpha_hat weight
                        w = numpy.exp(-self.latent_piecewise_variable_alpha_parameter_tying_beta*(i-j)**2)

                        # Add weight to normalization constant
                        normalization_constant += w

                    normalization_constant = normalization_constant.astype('float32')

                    for j in range(1, self.latent_piecewise_alpha_variables+1):
                        # Compute normalized alpha_hat weight
                        wn = numpy.exp(-self.latent_piecewise_variable_alpha_parameter_tying_beta*(i-j)**2)\
                            /normalization_constant
                        wn = wn.astype('float32')

                        # Add weight to alpha prior
                        self.latent_piecewise_utterance_variable_prior_alpha =                               \
                         T.inc_subtensor(self.latent_piecewise_utterance_variable_prior_alpha[:,:,:,i-1],\
                          wn*self.latent_piecewise_utterance_variable_prior_alpha_hat[:,:, :,j-1])

                        # Add weight to alpha posterior
                        self.latent_piecewise_utterance_variable_approx_posterior_alpha =                   \
                         T.inc_subtensor(self.latent_piecewise_utterance_variable_approx_posterior_alpha[:,:,:,i-1],\
                         wn*self.latent_piecewise_utterance_variable_approx_posterior_alpha_hat[:,:, :,j-1])


            else:
                self.latent_piecewise_utterance_variable_prior_alpha = \
                    self.latent_piecewise_utterance_variable_prior_alpha_hat
                self.latent_piecewise_utterance_variable_approx_posterior_alpha = \
                    self.latent_piecewise_utterance_variable_approx_posterior_alpha_hat


            if self.apply_meanfield_inference:
                self.latent_piecewise_utterance_variable_approx_posterior_alpha_mfbias = \
                    theano.shared(value=numpy.zeros((self.bs, self.latent_piecewise_per_utterance_dim,\
                                  self.latent_piecewise_alpha_variables), dtype='float32'),\
                                  name='latent_piecewise_utterance_variable_approx_posterior_alpha_mfbias')

                self.latent_piecewise_utterance_variable_approx_posterior_alpha += \
                    T.exp(self.latent_piecewise_utterance_variable_approx_posterior_alpha_mfbias.dimshuffle('x', 0, 1, 2))


            # Compute prior normalization constants:
            latent_piecewise_utterance_prior_ki = self.latent_piecewise_utterance_variable_prior_alpha / self.latent_piecewise_alpha_variables
            latent_piecewise_utterance_prior_k = T.sum(latent_piecewise_utterance_prior_ki, axis=3)



            # epsilon: a standard uniform sample in range [0, 1];
            #          shape: (time steps x batch size x number of piecewise latent variables)
            # latent_piecewise_posterior_sample: initialized to zeros;
            #          shape: (time steps x batch size x number of piecewise latent variables)
            # latent_piecewise_alpha_variables: integer representing number of pieces (I set this to 3)
            # latent_piecewise_utterance_variable_approx_posterior_alpha:
            #      un-normalized a values, i.e. the height of each rectangle;
            #      shape: (time steps x batch size x number of piecewise latent variables x latent_piecewise_alpha_variables)


            # Compute posterior normalization constants:
            # latent_piecewise_utterance_variable_prior_alpha: time steps x batch sizes x latent dim x pieces
            latent_piecewise_utterance_posterior_ki = self.latent_piecewise_utterance_variable_approx_posterior_alpha / self.latent_piecewise_alpha_variables
            latent_piecewise_utterance_posterior_k = T.sum(latent_piecewise_utterance_posterior_ki, axis=3)

            epsilon = self.ran_uniform_cost_utterance[:(self.x_max_length-1)]

            # Sample from posterior using inverse transform sampling:
            self.latent_piecewise_posterior_sample = T.zeros_like(epsilon)
            for i in range(1, self.latent_piecewise_alpha_variables+1):
                lowerbound = T.zeros_like(epsilon)
                for j in range(1, i):
                    lowerbound += (1.0/latent_piecewise_utterance_posterior_k)*latent_piecewise_utterance_posterior_ki[:,:, :,j-1]
                upperbound = lowerbound + (1.0/latent_piecewise_utterance_posterior_k)*latent_piecewise_utterance_posterior_ki[:,:, :,i-1]
                indicator = T.ge(epsilon, lowerbound)*T.lt(epsilon, upperbound)

                self.latent_piecewise_posterior_sample += \
                      indicator*((i - 1.0)/(self.latent_piecewise_alpha_variables) \
                      + (latent_piecewise_utterance_posterior_k/self.latent_piecewise_utterance_variable_approx_posterior_alpha[:,:,:,i-1])*(epsilon - lowerbound))

            # Transform sample to be in the range [-1, 1] with initial mean at zero.
            # This is considered as part of the decoder and does not affect KL divergence computations.
            self.latent_piecewise_posterior_sample = 2.0*self.latent_piecewise_posterior_sample - 1.0

            # Next, compute KL divergence cost
            self.kl_divergences_between_piecewise_prior_and_posterior = T.zeros_like(latent_variable_mask)
            for i in range(1, self.latent_piecewise_alpha_variables+1):
                self.kl_divergences_between_piecewise_prior_and_posterior += T.sum((1.0/self.latent_piecewise_alpha_variables)*(self.latent_piecewise_utterance_variable_approx_posterior_alpha[:,:,:,i-1]/latent_piecewise_utterance_posterior_k)*(T.log(self.latent_piecewise_utterance_variable_approx_posterior_alpha[:,:,:,i-1]/latent_piecewise_utterance_posterior_k)-T.log(self.latent_piecewise_utterance_variable_prior_alpha[:,:,:,i-1]/latent_piecewise_utterance_prior_k)), axis=2)

            self.kl_divergence_cost += self.kl_divergences_between_piecewise_prior_and_posterior*latent_variable_mask

        else:
            self.latent_piecewise_utterance_variable_approx_posterior_alpha = theano.shared(value=numpy.float(0))
            self.latent_piecewise_utterance_variable_prior_alpha = theano.shared(value=numpy.float(0))


        # We initialize the decoder, and fix its word embeddings to that of the encoder(s)
        logger.debug("Initializing decoder")
        self.utterance_decoder = UtteranceDecoder(self.state, self.rng, self, self.dialog_encoder, self.W_emb)

        # Define input vector for decoder
        if self.direct_connection_between_encoders_and_decoder:
            logger.debug("Build decoder (NCE) with direct connection from encoder(s)")
            if self.add_latent_gaussian_per_utterance and self.add_latent_piecewise_per_utterance:
                if self.condition_decoder_only_on_latent_variable:
                    self.hd_input = T.concatenate([self.latent_gaussian_posterior_sample, self.latent_piecewise_posterior_sample], axis=2)
                else:
                    self.hd_input = T.concatenate([self.hs, self.hs_dummy, self.latent_gaussian_posterior_sample, self.latent_piecewise_posterior_sample], axis=2)

            elif self.add_latent_gaussian_per_utterance:
                if self.condition_decoder_only_on_latent_variable:
                    self.hd_input = self.latent_gaussian_posterior_sample
                else:
                    self.hd_input = T.concatenate([self.hs, self.hs_dummy, self.latent_gaussian_posterior_sample], axis=2)
            elif self.add_latent_piecewise_per_utterance:
                if self.condition_decoder_only_on_latent_variable:
                    self.hd_input = self.latent_piecewise_posterior_sample
                else:
                    self.hd_input = T.concatenate([self.hs, self.hs_dummy, self.latent_piecewise_posterior_sample], axis=2)          
            else:
                self.hd_input = T.concatenate([self.hs, self.hs_dummy], axis=2)

        else:
            if self.add_latent_gaussian_per_utterance and self.add_latent_piecewise_per_utterance:
                if self.condition_decoder_only_on_latent_variable:
                    self.hd_input =  T.concatenate([self.latent_gaussian_posterior_sample, self.latent_piecewise_posterior_sample], axis=2)
                else:
                    self.hd_input = T.concatenate([self.hs, self.latent_gaussian_posterior_sample, self.latent_piecewise_posterior_sample], axis=2)
            elif self.add_latent_gaussian_per_utterance:
                if self.condition_decoder_only_on_latent_variable:
                    self.hd_input = self.latent_gaussian_posterior_sample
                else:
                    self.hd_input = T.concatenate([self.hs, self.latent_gaussian_posterior_sample], axis=2)
            elif self.add_latent_piecewise_per_utterance:
                if self.condition_decoder_only_on_latent_variable:
                    self.hd_input = self.latent_piecewise_posterior_sample
                else:
                    self.hd_input = T.concatenate([self.hs, self.latent_piecewise_posterior_sample], axis=2)
            else:
                self.hd_input = self.hs

        # Build decoder
        logger.debug("Build decoder (NCE)")
        contrastive_cost, self.hd_nce, self.utterance_decoder_nce_updates = self.utterance_decoder.build_decoder(self.hd_input, training_x, y_neg=self.y_neg, y=training_y, xmask=training_hs_mask, xdropmask=training_x_dropmask, mode=UtteranceDecoder.NCE, prev_state=self.phd)

        logger.debug("Build decoder (EVAL)")
        target_probs, self.hd, target_probs_full_matrix, self.utterance_decoder_updates = self.utterance_decoder.build_decoder(self.hd_input, training_x, xmask=training_hs_mask, xdropmask=training_x_dropmask, y=training_y, mode=UtteranceDecoder.EVALUATION, prev_state=self.phd)

        # Prediction cost and rank cost
        self.contrastive_cost = T.sum(contrastive_cost.flatten() * training_x_cost_mask_flat)
        self.softmax_cost = -T.log(target_probs) * training_x_cost_mask_flat
        self.softmax_cost_acc = T.sum(self.softmax_cost)

        # Prediction accuracy
        self.training_misclassification = T.neq(T.argmax(target_probs_full_matrix, axis=2), training_y).flatten() * training_x_cost_mask_flat

        self.training_misclassification_acc = T.sum(self.training_misclassification)

        # Compute training cost, which equals standard cross-entropy error
        self.training_cost = self.softmax_cost_acc
        if self.use_nce:
            self.training_cost = self.contrastive_cost

        # Compute training cost as variational lower bound with possible annealing of KL-divergence term
        if self.add_latent_gaussian_per_utterance or self.add_latent_piecewise_per_utterance:
            self.kl_divergence_cost_acc = T.sum(self.kl_divergence_cost)
            if self.train_latent_variables_with_kl_divergence_annealing:
                assert hasattr(self, 'max_kl_percentage') == False

                self.evaluation_cost = self.training_cost + T.minimum(self.kl_divergence_max_weight, 1.0)*self.kl_divergence_cost_acc

                self.kl_divergence_cost_weight = add_to_params(self.global_params, theano.shared(value=numpy.float32(0), name='kl_divergence_cost_weight'))
                self.training_cost = self.training_cost + T.minimum(self.kl_divergence_max_weight, self.kl_divergence_cost_weight)*self.kl_divergence_cost_acc

            else:
                if hasattr(self, 'max_kl_percentage'):
                    self.evaluation_cost = self.training_cost + self.kl_divergence_cost_acc

                    if self.add_latent_gaussian_per_utterance:
                        self.training_cost += T.maximum(self.max_kl_percentage*self.training_cost, T.sum(self.kl_divergences_between_gaussian_prior_and_posterior*latent_variable_mask))

                    if self.add_latent_piecewise_per_utterance:
                        self.training_cost += T.maximum(self.max_kl_percentage*self.training_cost, T.sum(self.kl_divergences_between_piecewise_prior_and_posterior*latent_variable_mask))

                else:
                    self.evaluation_cost = self.training_cost + self.kl_divergence_cost_acc
                    self.training_cost += self.kl_divergence_cost_acc

        else:
            self.evaluation_cost = self.training_cost
            self.kl_divergence_cost_acc = theano.shared(value=numpy.float(0))



        # Init params
        if self.collaps_to_standard_rnn:
                self.params = self.global_params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_decoder.params))
        else:
            if self.bidirectional_utterance_encoder:
                self.params = self.global_params + self.utterance_encoder_forward.params + self.utterance_encoder_backward.params + self.dialog_encoder.params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_encoder_forward.params) + len(self.utterance_encoder_backward.params) + len(self.dialog_encoder.params) + len(self.utterance_decoder.params))
            else:
                self.params = self.global_params + self.utterance_encoder.params + self.dialog_encoder.params + self.utterance_decoder.params
                assert len(set(self.params)) == (len(self.global_params) + len(self.utterance_encoder.params) + len(self.dialog_encoder.params) + len(self.utterance_decoder.params))

        if self.add_latent_gaussian_per_utterance:
            assert len(set(self.params)) + len(set(self.latent_gaussian_utterance_variable_prior_encoder.params)) \
                == len(set(self.params+self.latent_gaussian_utterance_variable_prior_encoder.params))
            self.params += self.latent_gaussian_utterance_variable_prior_encoder.params
            assert len(set(self.params)) + len(set(self.latent_gaussian_utterance_variable_approx_posterior_encoder.params)) \
                == len(set(self.params+self.latent_gaussian_utterance_variable_approx_posterior_encoder.params))
            self.params += self.latent_gaussian_utterance_variable_approx_posterior_encoder.params

            assert len(set(self.params)) + len(set(self.gaussian_posterior_mean_combination.params)) \
                == len(set(self.params+self.gaussian_posterior_mean_combination.params))
            self.params += self.gaussian_posterior_mean_combination.params

            assert len(set(self.params)) + len(set(self.posterior_variance_combination.params)) \
                == len(set(self.params+self.posterior_variance_combination.params))
            self.params += self.posterior_variance_combination.params

            if self.condition_posterior_latent_variable_on_dcgm_encoder:
                assert len(set(self.params)) + len(set(self.dcgm_encoder.params)) \
                    == len(set(self.params+self.dcgm_encoder.params))
                self.params += self.dcgm_encoder.params

        if self.add_latent_piecewise_per_utterance:
            assert len(set(self.params)) + len(set(self.latent_piecewise_utterance_variable_prior_encoder.params)) \
                == len(set(self.params+self.latent_piecewise_utterance_variable_prior_encoder.params))
            self.params += self.latent_piecewise_utterance_variable_prior_encoder.params
            assert len(set(self.params)) + len(set(self.latent_piecewise_utterance_variable_approx_posterior_encoder.params)) \
                == len(set(self.params+self.latent_piecewise_utterance_variable_approx_posterior_encoder.params))
            self.params += self.latent_piecewise_utterance_variable_approx_posterior_encoder.params

            if self.gate_latent_piecewise_per_utterance:
                assert len(set(self.params)) + len(set(self.piecewise_posterior_mean_combination.params)) \
                    == len(set(self.params+self.piecewise_posterior_mean_combination.params))
                self.params += self.piecewise_posterior_mean_combination.params


        # Create set of parameters to train
        self.params_to_train = []
        self.params_to_exclude = []
        if self.fix_encoder_parameters:
            # If the option fix_encoder_parameters is on, then we exclude all parameters 
            # related to the utterance encoder(s) and dialogue encoder, including the word embeddings,
            # from the parameter training set.
            if self.bidirectional_utterance_encoder:
                self.params_to_exclude = self.global_params + self.utterance_encoder_forward.params + self.utterance_encoder_backward.params + self.dialog_encoder.params
            else:
                self.params_to_exclude = self.global_params + self.utterance_encoder.params + self.dialog_encoder.params

        if self.add_latent_gaussian_per_utterance or self.add_latent_piecewise_per_utterance:
            # We always need to exclude the KL-divergence term weight from training,
            # since this value is being annealed (and should therefore not be optimized with SGD).
            if self.train_latent_variables_with_kl_divergence_annealing:
                self.params_to_exclude += [self.kl_divergence_cost_weight]

        # Add appropriate normalization operator parameters to list of parameters to exclude from training.
        # These parameters will be updated elsewhere.
        for param in self.params:
            if len(param.name) > 3:
                if param.name[0:7] == 'normop_':
                    if ('_mean_' in param.name) or ('_var_' in param.name):
                       self.params_to_exclude += [param]


        for param in self.params:
            if not param in self.params_to_exclude:
                self.params_to_train += [param]

        if self.compute_training_updates:
            self.updates, self.optimizer_variables = self.compute_updates(self.training_cost / training_x.shape[1], self.params_to_train)

            # Add additional updates, i.e. updates not related to SGD (e.g. batch norm updates)
            self.updates += res_forward_updates
            if self.bidirectional_utterance_encoder:
                self.updates += res_backward_updates

            self.updates += self.dialogue_encoder_updates
            self.updates += self.utterance_decoder_updates

            if self.add_latent_gaussian_per_utterance or self.add_latent_piecewise_per_utterance:
                self.updates += self.latent_variable_updates

            # Add optimizer parameters to parameter set. This will ensure that they are saved and loaded correctly.
            assert len(set(self.params)) + len(set(self.optimizer_variables)) \
                == len(set(self.params+self.optimizer_variables))
            self.params += self.optimizer_variables

            # Truncate gradients properly by bringing forward previous states
            # First, create reset mask
            x_reset = self.x_reset_mask.dimshuffle(0, 'x')
            # if flag 'reset_hidden_states_between_subsequences' is on, then always reset
            if self.reset_hidden_states_between_subsequences:
                x_reset = 0

            # Next, compute updates using reset mask (this depends on the number of RNNs in the model)
            self.state_updates = []
            if self.bidirectional_utterance_encoder:
                self.state_updates.append((self.ph_fwd, x_reset * res_forward[-1]))
                self.state_updates.append((self.ph_fwd_n, T.gt(x_reset.T, 0.0) * res_forward_n[-1]))

                self.state_updates.append((self.ph_bck, x_reset * res_backward[-1]))
                self.state_updates.append((self.ph_bck_n, T.gt(x_reset.T, 0.0) * res_backward_n[-1]))

                self.state_updates.append((self.phs, x_reset * self.hs[-1]))

                self.state_updates.append((self.phd, x_reset * self.hd[-1]))
            else:
                self.state_updates.append((self.ph, x_reset * res_forward[-1]))
                self.state_updates.append((self.ph_n, T.gt(x_reset.T, 0.0) * res_forward_n[-1]))

                self.state_updates.append((self.phs, x_reset * self.hs[-1]))

                self.state_updates.append((self.phd, x_reset * self.hd[-1]))

            if self.direct_connection_between_encoders_and_decoder:
                self.state_updates.append((self.phs_dummy, x_reset * self.hs_dummy[-1]))

            if self.add_latent_gaussian_per_utterance:
                self.state_updates.append((self.platent_gaussian_utterance_variable_prior, x_reset * self.latent_gaussian_utterance_variable_prior[-1]))
                self.state_updates.append((self.platent_gaussian_utterance_variable_approx_posterior, x_reset * self.latent_gaussian_utterance_variable_approx_posterior[-1]))

            if self.add_latent_piecewise_per_utterance:
                self.state_updates.append((self.platent_piecewise_utterance_variable_prior, x_reset * self.latent_piecewise_utterance_variable_prior[-1]))
                self.state_updates.append((self.platent_piecewise_utterance_variable_approx_posterior, x_reset * self.latent_piecewise_utterance_variable_approx_posterior[-1]))

            if self.add_latent_gaussian_per_utterance or self.add_latent_piecewise_per_utterance:
                if self.condition_posterior_latent_variable_on_dcgm_encoder:
                    self.state_updates.append((self.platent_dcgm_avg, x_reset * self.latent_dcgm_avg[-1]))
                    self.state_updates.append((self.platent_dcgm_n, x_reset.T * self.latent_dcgm_n[-1]))

                if self.train_latent_variables_with_kl_divergence_annealing:
                    self.state_updates.append((self.kl_divergence_cost_weight, T.minimum(1.0, self.kl_divergence_cost_weight + self.kl_divergence_annealing_rate)))



            # Add normalization operator updates,
            # which projects gamma parameters back to their constrained intervals:
            self.normop_gamma_params = []
            if not self.normop_type.upper() == 'NONE':
                print(' Searching for gamma parameters which must have bounded interval:')
                for param in self.params:
                    if len(param.name) > 9:
                        if param.name[0:3] == 'normop_':
                            if '_gamma_' in param.name:
                                if not '_optimizer_' in param.name:
                                    self.normop_gamma_params += [param]
                                    print('     ', param.name)

            self.gamma_bounding_updates = []
            for param in self.normop_gamma_params:
                new_gamma = T.minimum(T.maximum(param, self.normop_gamma_min), self.normop_gamma_max)
                self.gamma_bounding_updates.append((param, new_gamma))

        else:
            self.state_updates = []


        # Beam-search variables
        self.beam_x_data = T.imatrix('beam_x_data')
        self.beam_source = T.lvector("beam_source")
        self.beam_hs = T.matrix("beam_hs")
        self.beam_step_num = T.lscalar("beam_step_num")
        self.beam_hd = T.matrix("beam_hd")
        self.beam_ran_gaussian_cost_utterance = T.matrix('beam_ran_gaussian_cost_utterance')
        self.beam_ran_uniform_cost_utterance = T.matrix('beam_ran_uniform_cost_utterance')
