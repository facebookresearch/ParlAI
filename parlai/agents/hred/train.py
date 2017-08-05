# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from dialog_encdec import *
from utils import *

import time
import traceback
import sys
import argparse
import cPickle
import logging
import search
import pprint
import numpy
import collections
import signal
import math
import gc

import os
import os.path

# For certain clusters (e.g. Guillumin) we use flag 'DUMP_EXPERIMENT_LOGS_TO_DISC'
# to force dumping log outputs to file.
if 'DUMP_EXPERIMENT_LOGS_TO_DISC' in os.environ:
    if os.environ['DUMP_EXPERIMENT_LOGS_TO_DISC'] == '1':
        sys.stdout = open('Exp_Out.txt', 'a')
        sys.stderr = open('Exp_Err.txt', 'a')

from os import listdir
from os.path import isfile, join

import matplotlib
matplotlib.use('Agg')
import pylab


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
logger = logging.getLogger(__name__)

### Unique RUN_ID for this execution
RUN_ID = str(time.time())

### Additional measures can be set here
measures = ["train_cost", "train_misclass", "train_kl_divergence_cost", "train_posterior_gaussian_mean_variance", "valid_cost", "valid_misclass", "valid_posterior_gaussian_mean_variance", "valid_kl_divergence_cost", "valid_emi"]


def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save(model, timings, train_iterator, post_fix = ''):
    print("Saving the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)

    model.state['train_iterator_offset'] = train_iterator.get_offset() + 1
    model.state['train_iterator_reshuffle_count'] = train_iterator.get_reshuffle_count()

    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'state.pkl', 'w'))
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'timing.npz', **timings)
    signal.signal(signal.SIGINT, s)
    
    print("Model saved, took {}".format(time.time() - start))

def load(model, filename, parameter_strings_to_ignore):
    print("Loading the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename, parameter_strings_to_ignore)
    signal.signal(signal.SIGINT, s)

    print("Model loaded, took {}".format(time.time() - start))

def main(args):     
    logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
   
    state = eval(args.prototype)() 
    timings = init_timings() 

    auto_restarting = False
    if args.auto_restart:
        assert not args.save_every_valid_iteration
        assert len(args.resume) == 0

        directory = state['save_dir']
        if not directory[-1] == '/':
            directory = directory + '/' 

        auto_resume_postfix = state['prefix'] + '_auto_model.npz'

        if os.path.exists(directory):
            directory_files = [f for f in listdir(directory) if isfile(join(directory, f))]
            resume_filename = ''
            for f in directory_files:
                if len(f) > len(auto_resume_postfix):
                    if f[len(f) - len(auto_resume_postfix):len(f)] == auto_resume_postfix:
                        if len(resume_filename) > 0:
                            print('ERROR: FOUND MULTIPLE MODELS IN DIRECTORY:', directory)
                            assert False
                        else:
                            resume_filename = directory + f[0:len(f)-len('__auto_model.npz')]

            if len(resume_filename) > 0:
                logger.debug("Found model to automatically resume: %s" % resume_filename)
                auto_restarting = True
                # Setup training to automatically resume training with the model found
                args.resume = resume_filename + '__auto'
                # Disable training from reinitialization any parameters
                args.reinitialize_decoder_parameters = False
                args.reinitialize_latent_variable_parameters = False
            else:
                logger.debug("Could not find any model to automatically resume...")



    if args.resume != "":
        logger.debug("Resuming %s" % args.resume)
        
        state_file = args.resume + '_state.pkl'
        timings_file = args.resume + '_timing.npz'
        
        if os.path.isfile(state_file) and os.path.isfile(timings_file):
            logger.debug("Loading previous state")
            
            state = cPickle.load(open(state_file, 'r'))
            timings = dict(numpy.load(open(timings_file, 'r')))
            for x, y in timings.items():
                timings[x] = list(y)

            # Increment seed to make sure we get newly shuffled batches when training on large datasets
            state['seed'] = state['seed']

        else:
            raise Exception("Cannot resume, cannot find files!")



    logger.debug("State:\n{}".format(pprint.pformat(state)))
    logger.debug("Timings:\n{}".format(pprint.pformat(timings)))
 
    if args.force_train_all_wordemb == True:
        state['fix_pretrained_word_embeddings'] = False

    model = DialogEncoderDecoder(state)
    rng = model.rng 

    valid_rounds = 0
    save_model_on_first_valid = False

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")

            parameter_strings_to_ignore = []
            if args.reinitialize_decoder_parameters:
                parameter_strings_to_ignore += ['Wd_']
                parameter_strings_to_ignore += ['bd_']

                save_model_on_first_valid = True
            if args.reinitialize_latent_variable_parameters:
                parameter_strings_to_ignore += ['latent_utterance_prior']
                parameter_strings_to_ignore += ['latent_utterance_approx_posterior']
                parameter_strings_to_ignore += ['kl_divergence_cost_weight']
                parameter_strings_to_ignore += ['latent_dcgm_encoder']

                save_model_on_first_valid = True

            load(model, filename, parameter_strings_to_ignore)
        else:
            raise Exception("Cannot resume, cannot find model file!")
        
        if 'run_id' not in model.state:
            raise Exception('Backward compatibility not ensured! (need run_id in state)')           

    else:
        # assign new run_id key
        model.state['run_id'] = RUN_ID

    logger.debug("Compile trainer")
    if not state["use_nce"]:
        if ('add_latent_gaussian_per_utterance' in state) and (state["add_latent_gaussian_per_utterance"]):
            logger.debug("Training using variational lower bound on log-likelihood")
        else:
            logger.debug("Training using exact log-likelihood")

        train_batch = model.build_train_function()
    else:
        logger.debug("Training with noise contrastive estimation")
        train_batch = model.build_nce_function()

    eval_batch = model.build_eval_function()

    gamma_bounding = model.build_gamma_bounding_function()

    random_sampler = search.RandomSampler(model)
    beam_sampler = search.BeamSampler(model) 

    logger.debug("Load data")
    train_data, \
    valid_data, = get_train_iterator(state)
    train_data.start()

    # Start looping through the dataset
    step = 0
    patience = state['patience'] 
    start_time = time.time()
     
    train_cost = 0
    train_kl_divergence_cost = 0
    train_posterior_gaussian_mean_variance = 0
    train_misclass = 0
    train_done = 0
    train_dialogues_done = 0.0

    prev_train_cost = 0
    prev_train_done = 0

    ex_done = 0
    is_end_of_batch = True
    start_validation = False

    batch = None

    while (step < state['loop_iters'] and
            (time.time() - start_time)/60. < state['time_stop'] and
            patience >= 0):

        # Flush to log files
        sys.stderr.flush()
        sys.stdout.flush()

        ### Sampling phase
        if step % 200 == 0:
            # First generate stochastic samples
            for param in model.params:
                print("%s = %.4f" % (param.name, numpy.sum(param.get_value() ** 2) ** 0.5))

            samples, costs = random_sampler.sample([[]], n_samples=1, n_turns=3)
            print("Sampled : {}".format(samples[0]))


        ### Training phase
        batch = train_data.next()

        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break

        logger.debug("[TRAIN] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
        x_data = batch['x']
        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
        x_reset = batch['x_reset']
        ran_gaussian_const_utterance = batch['ran_var_gaussian_constutterance']
        ran_uniform_const_utterance = batch['ran_var_uniform_constutterance']

        ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

        is_end_of_batch = False
        if numpy.sum(numpy.abs(x_reset)) < 1:
            # Print when we reach the end of an example (e.g. the end of a dialogue or a document)
            # Knowing when the training procedure reaches the end is useful for diagnosing training problems
            # print('END-OF-BATCH EXAMPLE!')
            is_end_of_batch = True

        if state['use_nce']:
            y_neg = rng.choice(size=(10, max_length, x_data.shape[1]), a=model.idim, p=model.noise_probs).astype('int32')
            c, kl_divergence_cost, posterior_gaussian_mean_variance = train_batch(x_data, x_data_reversed, y_neg, max_length, x_cost_mask, x_reset, ran_gaussian_const_utterance, ran_uniform_const_utterance, ran_decoder_drop_mask)
        else:

            latent_piecewise_utterance_variable_approx_posterior_alpha = 0.0
            latent_piecewise_utterance_variable_prior_alpha = 0.0
            kl_divergences_between_piecewise_prior_and_posterior = 0.0
            kl_divergences_between_gaussian_prior_and_posterior = 0.0
            latent_piecewise_posterior_sample = 0.0
            posterior_gaussian_mean_variance = 0.0

            if model.add_latent_piecewise_per_utterance and model.add_latent_gaussian_per_utterance:
                c, kl_divergence_cost, posterior_gaussian_mean_variance, latent_piecewise_utterance_variable_approx_posterior_alpha, latent_piecewise_utterance_variable_prior_alpha, kl_divergences_between_piecewise_prior_and_posterior, kl_divergences_between_gaussian_prior_and_posterior, latent_piecewise_posterior_sample = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_gaussian_const_utterance, ran_uniform_const_utterance, ran_decoder_drop_mask)
            elif model.add_latent_gaussian_per_utterance:
                c, kl_divergence_cost, posterior_gaussian_mean_variance, kl_divergences_between_gaussian_prior_and_posterior = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_gaussian_const_utterance, ran_uniform_const_utterance, ran_decoder_drop_mask)
            elif model.add_latent_piecewise_per_utterance:
                c, kl_divergence_cost, kl_divergences_between_piecewise_prior_and_posterior = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_gaussian_const_utterance, ran_uniform_const_utterance, ran_decoder_drop_mask)
            else:
                c = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_gaussian_const_utterance, ran_uniform_const_utterance, ran_decoder_drop_mask)
                kl_divergence_cost = 0.0





        gamma_bounding()

        # Print batch statistics
        print('cost_sum', c)
        print('cost_mean', c / float(numpy.sum(x_cost_mask)))

        if model.add_latent_piecewise_per_utterance or model.add_latent_gaussian_per_utterance:
            print('kl_divergence_cost_sum', kl_divergence_cost)
            print('kl_divergence_cost_mean', kl_divergence_cost / float(len(numpy.where(x_data == model.eos_sym)[0])))

        if model.add_latent_gaussian_per_utterance:
            print('posterior_gaussian_mean_variance', posterior_gaussian_mean_variance)
            print('kl_divergences_between_gaussian_prior_and_posterior', numpy.sum(kl_divergences_between_gaussian_prior_and_posterior), numpy.min(kl_divergences_between_gaussian_prior_and_posterior), numpy.max(kl_divergences_between_gaussian_prior_and_posterior))

        if model.add_latent_piecewise_per_utterance:
            print('kl_divergences_between_piecewise_prior_and_posterior', numpy.sum(kl_divergences_between_piecewise_prior_and_posterior), numpy.min(kl_divergences_between_piecewise_prior_and_posterior), numpy.max(kl_divergences_between_piecewise_prior_and_posterior))


        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            gc.collect()
            continue

        train_cost += c
        train_kl_divergence_cost += kl_divergence_cost
        train_posterior_gaussian_mean_variance += posterior_gaussian_mean_variance

        train_done += batch['num_preds']
        train_dialogues_done += batch['num_dialogues']

        this_time = time.time()
        if step % state['train_freq'] == 0:
            elapsed = this_time - start_time

            # Keep track of training cost for the last 'train_freq' batches.
            current_train_cost = train_cost/train_done
            if prev_train_done >= 1 and abs(train_done - prev_train_done) > 0:
                current_train_cost = float(train_cost - prev_train_cost)/float(train_done - prev_train_done)

            if numpy.isinf(c) or numpy.isnan(c):
                current_train_cost = 0

            prev_train_cost = train_cost
            prev_train_done = train_done

            h, m, s = ConvertTimedelta(this_time - start_time)

            # We need to catch exceptions due to high numbers in exp
            try:
                print(".. %.2d:%.2d:%.2d %4d mb # %d bs %d maxl %d acc_cost = %.4f acc_word_perplexity = %.4f cur_cost = %.4f cur_word_perplexity = %.4f acc_mean_word_error = %.4f acc_mean_kl_divergence_cost = %.8f acc_mean_posterior_variance = %.8f" % (h, m, s,\
                                 state['time_stop'] - (time.time() - start_time)/60.,\
                                 step, \
                                 batch['x'].shape[1], \
                                 batch['max_length'], \
                                 float(train_cost/train_done), \
                                 math.exp(float(train_cost/train_done)), \
                                 current_train_cost, \
                                 math.exp(current_train_cost), \
                                 float(train_misclass)/float(train_done), \
                                 float(train_kl_divergence_cost/train_done), \
                                 float(train_posterior_gaussian_mean_variance/train_dialogues_done)))
            except:
                pass


        ### Inspection phase
        if (step % 20 == 0):
            if model.add_latent_gaussian_per_utterance and model.add_latent_piecewise_per_utterance:
                try:
                    print('posterior_gaussian_mean_combination', model.posterior_mean_combination.W.get_value())

                except:
                    pass

                print('latent_piecewise_utterance_variable_approx_posterior_alpha', numpy.mean(latent_piecewise_utterance_variable_approx_posterior_alpha), latent_piecewise_utterance_variable_approx_posterior_alpha)

                print('latent_piecewise_utterance_variable_prior_alpha', numpy.mean(latent_piecewise_utterance_variable_prior_alpha), latent_piecewise_utterance_variable_prior_alpha)

                print('latent_piecewise_utterance_variable_alpha_diff', (latent_piecewise_utterance_variable_approx_posterior_alpha-latent_piecewise_utterance_variable_prior_alpha))

                print('latent_piecewise_posterior_sample', numpy.min(latent_piecewise_posterior_sample), numpy.max(latent_piecewise_posterior_sample), latent_piecewise_posterior_sample[0, 0, :])
                print('ran_uniform_const_utterance', numpy.min(ran_uniform_const_utterance), numpy.max(ran_uniform_const_utterance), ran_uniform_const_utterance[0, 0, :])

            if model.utterance_decoder_gating.upper() == 'GRU' and model.decoder_bias_type.upper() == 'ALL':
                Wd_s_q = model.utterance_decoder.Wd_s_q.get_value()
                Wd_s_q_len = Wd_s_q.shape[0]
                print('model.utterance_decoder Wd_s_q full', numpy.mean(numpy.abs(Wd_s_q)), numpy.mean(Wd_s_q**2))

                if model.add_latent_gaussian_per_utterance and model.add_latent_piecewise_per_utterance:
                    Wd_s_q_gaussian = Wd_s_q[Wd_s_q_len-2*model.latent_piecewise_per_utterance_dim:Wd_s_q_len-model.latent_piecewise_per_utterance_dim, :]
                    Wd_s_q_piecewise = Wd_s_q[Wd_s_q_len-model.latent_piecewise_per_utterance_dim:Wd_s_q_len, :]

                    print('model.utterance_decoder Wd_s_q gaussian', numpy.mean(numpy.abs(Wd_s_q_gaussian)), numpy.mean(Wd_s_q_gaussian**2))
                    print('model.utterance_decoder Wd_s_q piecewise', numpy.mean(numpy.abs(Wd_s_q_piecewise)), numpy.mean(Wd_s_q_piecewise**2))

                    print('model.utterance_decoder Wd_s_q piecewise/gaussian', numpy.mean(numpy.abs(Wd_s_q_piecewise))/numpy.mean(numpy.abs(Wd_s_q_gaussian)), numpy.mean(Wd_s_q_piecewise**2)/numpy.mean(Wd_s_q_gaussian**2))

                elif model.add_latent_gaussian_per_utterance:
                    Wd_s_q_piecewise = Wd_s_q[Wd_s_q_len-model.latent_piecewise_per_utterance_dim:Wd_s_q_len, :]

                    print('model.utterance_decoder Wd_s_q piecewise', numpy.mean(numpy.abs(Wd_s_q_piecewise)), numpy.mean(Wd_s_q_piecewise**2))


                elif model.add_latent_piecewise_per_utterance:
                    Wd_s_q_gaussian = Wd_s_q[Wd_s_q_len-model.latent_piecewise_per_utterance_dim:Wd_s_q_len, :]

                    print('model.utterance_decoder Wd_s_q gaussian', numpy.mean(numpy.abs(Wd_s_q_gaussian)), numpy.mean(Wd_s_q_gaussian**2))



            if model.utterance_decoder_gating.upper() == 'BOW' and model.decoder_bias_type.upper() == 'ALL':
                Wd_bow_W_in = model.utterance_decoder.Wd_bow_W_in.get_value()
                Wd_bow_W_in_len = Wd_bow_W_in.shape[0]
                print('model.utterance_decoder Wd_bow_W_in full', numpy.mean(numpy.abs(Wd_bow_W_in)), numpy.mean(Wd_bow_W_in**2))

                if model.add_latent_gaussian_per_utterance and model.add_latent_piecewise_per_utterance:
                    Wd_bow_W_in_gaussian = Wd_bow_W_in[Wd_bow_W_in_len-2*model.latent_piecewise_per_utterance_dim:Wd_bow_W_in_len-model.latent_piecewise_per_utterance_dim, :]
                    Wd_bow_W_in_piecewise = Wd_bow_W_in[Wd_bow_W_in_len-model.latent_piecewise_per_utterance_dim:Wd_bow_W_in_len, :]

                    print('model.utterance_decoder Wd_bow_W_in gaussian', numpy.mean(numpy.abs(Wd_bow_W_in_gaussian)), numpy.mean(Wd_bow_W_in_gaussian**2))
                    print('model.utterance_decoder Wd_bow_W_in piecewise', numpy.mean(numpy.abs(Wd_bow_W_in_piecewise)), numpy.mean(Wd_bow_W_in_piecewise**2))

                    print('model.utterance_decoder Wd_bow_W_in piecewise/gaussian', numpy.mean(numpy.abs(Wd_bow_W_in_piecewise))/numpy.mean(numpy.abs(Wd_bow_W_in_gaussian)), numpy.mean(Wd_bow_W_in_piecewise**2)/numpy.mean(Wd_bow_W_in_gaussian**2))

                elif model.add_latent_gaussian_per_utterance:
                    Wd_bow_W_in_piecewise = Wd_bow_W_in[Wd_bow_W_in_len-model.latent_piecewise_per_utterance_dim:Wd_bow_W_in_len, :]

                    print('model.utterance_decoder Wd_bow_W_in piecewise', numpy.mean(numpy.abs(Wd_bow_W_in_piecewise)), numpy.mean(Wd_bow_W_in_piecewise**2))


                elif model.add_latent_piecewise_per_utterance:
                    Wd_bow_W_in_gaussian = Wd_bow_W_in[Wd_bow_W_in_len-model.latent_piecewise_per_utterance_dim:Wd_bow_W_in_len, :]

                    print('model.utterance_decoder Wd_bow_W_in gaussian', numpy.mean(numpy.abs(Wd_bow_W_in_gaussian)), numpy.mean(Wd_bow_W_in_gaussian**2))









        ### Evaluation phase
        if valid_data is not None and\
            step % state['valid_freq'] == 0 and step > 1:
                start_validation = True

        # Only start validation loop once it's time to validate and once all previous batches have been reset
        if start_validation and is_end_of_batch:
                start_validation = False
                valid_data.start()
                valid_cost = 0
                valid_kl_divergence_cost = 0
                valid_posterior_gaussian_mean_variance = 0

                valid_wordpreds_done = 0
                valid_dialogues_done = 0


                logger.debug("[VALIDATION START]")

                while True:
                    batch = valid_data.next()

                    # Validation finished
                    if not batch:
                        break


                    logger.debug("[VALID] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
                    x_data = batch['x']
                    x_data_reversed = batch['x_reversed']
                    max_length = batch['max_length']
                    x_cost_mask = batch['x_mask']

                    x_reset = batch['x_reset']
                    ran_gaussian_const_utterance = batch['ran_var_gaussian_constutterance']
                    ran_uniform_const_utterance = batch['ran_var_uniform_constutterance']

                    ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

                    posterior_gaussian_mean_variance = 0.0

                    c, c_list, kl_divergence_cost = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_gaussian_const_utterance, ran_uniform_const_utterance, ran_decoder_drop_mask)


                    # Rehape into matrix, where rows are validation samples and columns are tokens
                    # Note that we use max_length-1 because we don't get a cost for the first token
                    # (the first token is always assumed to be eos)
                    c_list = c_list.reshape((batch['x'].shape[1],max_length-1), order=(1,0))
                    c_list = numpy.sum(c_list, axis=1)
                    
                    words_in_dialogues = numpy.sum(x_cost_mask, axis=0)
                    c_list = c_list / words_in_dialogues
                    

                    if numpy.isinf(c) or numpy.isnan(c):
                        continue
                    
                    valid_cost += c
                    valid_kl_divergence_cost += kl_divergence_cost
                    valid_posterior_gaussian_mean_variance += posterior_gaussian_mean_variance

                    # Print batch statistics
                    print('valid_cost', valid_cost)
                    print('valid_kl_divergence_cost sample', kl_divergence_cost)
                    print('posterior_gaussian_mean_variance', posterior_gaussian_mean_variance)


                    valid_wordpreds_done += batch['num_preds']
                    valid_dialogues_done += batch['num_dialogues']

                logger.debug("[VALIDATION END]") 
                 
                valid_cost /= max(1.0, valid_wordpreds_done)
                valid_kl_divergence_cost /= max(1.0, valid_wordpreds_done)
                valid_posterior_gaussian_mean_variance /= max(1.0, valid_dialogues_done)

                if (len(timings["valid_cost"]) == 0) \
                    or (valid_cost < numpy.min(timings["valid_cost"])) \
                    or (save_model_on_first_valid and valid_rounds == 0):
                    patience = state['patience']

                    # Save model if there is decrease in validation cost
                    save(model, timings, train_data)
                    print('best valid_cost', valid_cost)
                elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
                    patience -= 1

                if args.save_every_valid_iteration:
                    save(model, timings, train_data, '_' + str(step) + '_')
                if args.auto_restart:
                    save(model, timings, train_data, '_auto_')


                # We need to catch exceptions due to high numbers in exp
                try:
                    print("** valid cost (NLL) = %.4f, valid word-perplexity = %.4f, valid kldiv cost (per word) = %.8f, valid mean posterior variance (per word) = %.8f, patience = %d" % (float(valid_cost), float(math.exp(valid_cost)), float(valid_kl_divergence_cost), float(valid_posterior_gaussian_mean_variance), patience))
                except:
                    try:
                        print("** valid cost (NLL) = %.4f, patience = %d" % (float(valid_cost), patience))
                    except:
                        pass


                timings["train_cost"].append(train_cost/train_done)
                timings["train_kl_divergence_cost"].append(train_kl_divergence_cost/train_done)
                timings["train_posterior_gaussian_mean_variance"].append(train_posterior_gaussian_mean_variance/train_dialogues_done)
                timings["valid_cost"].append(valid_cost)
                timings["valid_kl_divergence_cost"].append(valid_kl_divergence_cost)
                timings["valid_posterior_gaussian_mean_variance"].append(valid_posterior_gaussian_mean_variance)

                # Reset train cost, train misclass and train done metrics
                train_cost = 0
                train_done = 0
                prev_train_cost = 0
                prev_train_done = 0

                # Count number of validation rounds done so far
                valid_rounds += 1

        step += 1

    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")

    parser.add_argument("--force_train_all_wordemb", action='store_true', help="If true, will force the model to train all word embeddings in the encoder. This switch can be used to fine-tune a model which was trained with fixed (pretrained)  encoder word embeddings.")

    parser.add_argument("--save_every_valid_iteration", action='store_true', help="If true, will save a unique copy of the model at every validation round.")

    parser.add_argument("--auto_restart", action='store_true', help="If true, will maintain a copy of the current model parameters updated at every validation round. Upon initialization, the script will automatically scan the output directory and and resume training of a previous model (if such exists). This option is meant to be used for training models on clusters with hard wall-times. This option is incompatible with the \"resume\" and \"save_every_valid_iteration\" options.")

    parser.add_argument("--prototype", type=str, help="Prototype to use (must be specified)", default='prototype_state')

    parser.add_argument("--reinitialize-latent-variable-parameters", action='store_true', help="Can be used when resuming a model. If true, will initialize all latent variable parameters randomly instead of loading them from previous model.")

    parser.add_argument("--reinitialize-decoder-parameters", action='store_true', help="Can be used when resuming a model. If true, will initialize all parameters of the utterance decoder randomly instead of loading them from previous model.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    main(args)

    # grep 'valid cost' LSTM_Baseline_exp1/LOGS/python_train.py_prototype_twitter_LSTM_NormOp_ClusterExp1_2016-09-23_22-48-31.523628/dbi_146c0c3c23d.out-* | grep -o -P '(?<=word-perplexity = ).*(?=, valid kldiv)'
