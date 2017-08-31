import copy
import itertools
import logging
import math
import numpy as np
from SS_dataset import SSIterator

logger = logging.getLogger(__name__)


def add_random_variables_to_batch(state, rng, batch, prev_batch, evaluate_mode):
    """
    This is a helper function, which adds random variables to a batch.
    We do it this way, because we want to avoid Theano's random sampling both to speed up and to avoid
    known Theano issues with sampling inside scan loops.

    The random variable 'ran_var_gaussian_constutterance' is sampled from a standard Gaussian distribution,
    which remains constant during each utterance (i.e. between a pair of end-of-utterance tokens).

    The random variable 'ran_var_uniform_constutterance' is sampled from a uniform distribution [0, 1],
    which remains constant during each utterance (i.e. between a pair of end-of-utterance tokens).

    When not in evaluate mode, the random vector 'ran_decoder_drop_mask' is also sampled.
    This variable represents the input tokens which are replaced by unk when given to
    the decoder RNN. It is required for the noise addition trick used by Bowman et al. (2015).
    """

    # If none return none
    if not batch:
        return batch

    # Variables to store random vector sampled at the beginning of each utterance
    Ran_Var_Gaussian_ConstUtterance = np.zeros((batch['x'].shape[0], batch['x'].shape[1], state['latent_gaussian_per_utterance_dim']), dtype='float32')
    Ran_Var_Uniform_ConstUtterance = np.zeros((batch['x'].shape[0], batch['x'].shape[1], state['latent_piecewise_per_utterance_dim']), dtype='float32')

    # Go through each sample, find end-of-utterance indices and sample random variables
    for idx in range(batch['x'].shape[1]):
        # Find end-of-utterance indices
        eos_indices = np.where(batch['x'][:, idx] == state['eos_sym'])[0].tolist()

        # Make sure we also sample at the beginning of the utterance, and that we stop appropriately at the end
        if len(eos_indices) > 0:
            if not eos_indices[0] == 0:
                eos_indices = [0] + eos_indices
            if not eos_indices[-1] == batch['x'].shape[0]:
                eos_indices = eos_indices + [batch['x'].shape[0]]
        else:
            eos_indices = [0] + [batch['x'].shape[0]]

        # Sample random variables using NumPy
        ran_gaussian_vectors = rng.normal(loc=0, scale=1, size=(len(eos_indices), state['latent_gaussian_per_utterance_dim']))
        ran_uniform_vectors = rng.uniform(low=0.0, high=1.0, size=(len(eos_indices), state['latent_piecewise_per_utterance_dim']))

        for i in range(len(eos_indices)-1):
            for j in range(eos_indices[i], eos_indices[i+1]):
                Ran_Var_Gaussian_ConstUtterance[j, idx, :] = ran_gaussian_vectors[i, :]
                Ran_Var_Uniform_ConstUtterance[j, idx, :] = ran_uniform_vectors[i, :]

        # If a previous batch is given, and the last utterance in the previous batch
        # overlaps with the first utterance in the current batch, then we need to copy over
        # the random variables from the last utterance in the last batch to remain consistent.
        if prev_batch:
            if ('x_reset' in prev_batch) and (not np.sum(np.abs(prev_batch['x_reset'])) < 1) \
              and (('ran_var_gaussian_constutterance' in prev_batch) or ('ran_var_uniform_constutterance' in prev_batch)):
                prev_ran_gaussian_vector = prev_batch['ran_var_gaussian_constutterance'][-1,idx,:]
                prev_ran_uniform_vector = prev_batch['ran_var_uniform_constutterance'][-1,idx,:]
                if len(eos_indices) > 1:
                    for j in range(0, eos_indices[1]):
                        Ran_Var_Gaussian_ConstUtterance[j, idx, :] = prev_ran_gaussian_vector
                        Ran_Var_Uniform_ConstUtterance[j, idx, :] = prev_ran_uniform_vector
                else:
                    for j in range(0, batch['x'].shape[0]):
                        Ran_Var_Gaussian_ConstUtterance[j, idx, :] = prev_ran_gaussian_vector
                        Ran_Var_Uniform_ConstUtterance[j, idx, :] = prev_ran_uniform_vector

    # Add new random Gaussian variable to batch
    batch['ran_var_gaussian_constutterance'] = Ran_Var_Gaussian_ConstUtterance
    batch['ran_var_uniform_constutterance'] = Ran_Var_Uniform_ConstUtterance

    # Create word drop mask based on 'decoder_drop_previous_input_tokens_rate' option:
    if evaluate_mode:
        batch['ran_decoder_drop_mask'] = np.ones((batch['x'].shape[0], batch['x'].shape[1]), dtype='float32')
    else:
        if state.get('decoder_drop_previous_input_tokens', False):
            ran_drop = rng.uniform(size=(batch['x'].shape[0], batch['x'].shape[1]))
            batch['ran_decoder_drop_mask'] = (ran_drop <= state['decoder_drop_previous_input_tokens_rate']).astype('float32')
        else:
            batch['ran_decoder_drop_mask'] = np.ones((batch['x'].shape[0], batch['x'].shape[1]), dtype='float32')

    return batch


def create_padded_batch(state, rng, x, force_end_of_utterance_token = False):
    # If flag 'do_generate_first_utterance' is off, then zero out the mask for the first utterance.
    do_generate_first_utterance = True
    if 'do_generate_first_utterance' in state:
        if state['do_generate_first_utterance'] is False:
            do_generate_first_utterance = False

    # Skip utterance model
    if state.get('skip_utterance', False):
        do_generate_first_utterance = False

    #    x = copy.deepcopy(x)
    #    for idx in xrange(len(x[0])):
    #        eos_indices = numpy.where(numpy.asarray(x[0][idx]) == state['eos_sym'])[0]
    #        if not x[0][idx][0] == state['eos_sym']:
    #            eos_indices = numpy.insert(eos_indices, 0, state['eos_sym'])
    #        if not x[0][idx][-1] == state['eos_sym']:
    #            eos_indices = numpy.append(eos_indices, state['eos_sym'])
    #
    #        if len(eos_indices) > 2:
    #            first_utterance_index = rng.randint(0, len(eos_indices)-2)
    #
    #            # Predict next or previous utterance
    #            if state.get('skip_utterance_predict_both', False):
    #                if rng.randint(0, 2) == 0:
    #                    x[0][idx] = x[0][idx][eos_indices[first_utterance_index]:eos_indices[first_utterance_index+2]+1]
    #                else:
    #                    x[0][idx] = x[0][idx][eos_indices[first_utterance_index+1]:eos_indices[first_utterance_index+2]] + x[0][idx][eos_indices[first_utterance_index]:eos_indices[first_utterance_index+1]+1]
    #            else:
    #
    #        else:
    #            x[0][idx] = [state['eos_sym']]

    # Find max length in batch
    mx = 0
    for idx in range(len(x[0])):
        mx = max(mx, len(x[0][idx]))

    # Take into account that sometimes we need to add the end-of-utterance symbol at the start
    mx += 1

    n = state['bs']

    X = np.zeros((mx, n), dtype='int32')
    Xmask = np.zeros((mx, n), dtype='float32')

    # Variable to store each utterance in reverse form (for bidirectional RNNs)
    X_reversed = np.zeros((mx, n), dtype='int32')

    # Fill X and Xmask.
    # Keep track of number of predictions and maximum dialogue length.
    num_preds = 0
    max_length = 0
    for idx in range(len(x[0])):
        # Insert sequence idx in a column of matrix X
        dialogue_length = len(x[0][idx])

        # Fiddle-it if it is too long ..
        if mx < dialogue_length:
            continue

        # Make sure end-of-utterance symbol is at beginning of dialogue.
        # This will force model to generate first utterance too
        if not x[0][idx][0] == state['eos_sym']:
            X[:dialogue_length+1, idx] = [state['eos_sym']] + x[0][idx][:dialogue_length]
            dialogue_length = dialogue_length + 1
        else:
            X[:dialogue_length, idx] = x[0][idx][:dialogue_length]

        # Keep track of longest dialogue
        max_length = max(max_length, dialogue_length)

        # Set the number of predictions == sum(Xmask), for cost purposes, minus one (to exclude first eos symbol)
        num_preds += dialogue_length - 1

        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            if force_end_of_utterance_token:
                X[dialogue_length:, idx] = state['eos_sym']

        # Initialize Xmask column with ones in all positions that
        # were just set in X (except for first eos symbol, because we are not evaluating this).
        # Note: if we need mask to depend on tokens inside X, then we need to
        # create a corresponding mask for X_reversed and send it further in the model
        Xmask[0:dialogue_length, idx] = 1.

        # Reverse all utterances
        # TODO: For backward compatibility. This should be removed in future versions
        # i.e. move all the x_reversed computations to the model itself.
        eos_indices = np.where(X[:, idx] == state['eos_sym'])[0]
        X_reversed[:, idx] = X[:, idx]
        prev_eos_index = -1
        for eos_index in eos_indices:
            X_reversed[(prev_eos_index+1):eos_index, idx] = (X_reversed[(prev_eos_index+1):eos_index, idx])[::-1]
            prev_eos_index = eos_index
            if prev_eos_index > dialogue_length:
                break

        if not do_generate_first_utterance:
            eos_index_to_start_cost_from = eos_indices[0]
            if (eos_index_to_start_cost_from == 0) and (len(eos_indices) > 1):
                eos_index_to_start_cost_from = eos_indices[1]
                Xmask[0:eos_index_to_start_cost_from+1, idx] = 0.

            if np.sum(Xmask[:, idx]) < 2.0:
                Xmask[:, idx] = 0.

    if do_generate_first_utterance:
        assert num_preds == np.sum(Xmask) - np.sum(Xmask[0, :])

    batch = {'x': X,                                                 \
             'x_reversed': X_reversed,                               \
             'x_mask': Xmask,                                        \
             'num_preds': num_preds,                                 \
             'num_dialogues': len(x[0]),                             \
             'max_length': max_length                                \
            }

    return batch


class Iterator(SSIterator):
    def __init__(self, dialogue_file, batch_size, **kwargs):
        self.state = kwargs.pop('state', None)
        self.k_batches = kwargs.pop('sort_k_batches', 20)

        if ('skip_utterance' in self.state and
                'do_generate_first_utterance' in self.state):
            if self.state['skip_utterance']:
                assert not self.state.get('do_generate_first_utterance', False)

        # Store whether the iterator operates in evaluate mode or not
        self.evaluate_mode = kwargs.pop('evaluate_mode', False)
        print('Data Iterator Evaluate Mode: ', self.evaluate_mode)

        if self.evaluate_mode:
            SSIterator.__init__(self, dialogue_file, batch_size,                          \
                                seed=kwargs.pop('seed', 1234),                            \
                                max_len=kwargs.pop('max_len', -1),                        \
                                use_infinite_loop=kwargs.pop('use_infinite_loop', False), \
                                eos_sym=self.state['eos_sym'],                            \
                                skip_utterance=self.state.get('skip_utterance', False),   \
                                skip_utterance_predict_both=self.state.get('skip_utterance_predict_both', False))
        else:
            SSIterator.__init__(self, dialogue_file, batch_size,                          \
                                seed=kwargs.pop('seed', 1234),                            \
                                max_len=kwargs.pop('max_len', -1),                        \
                                use_infinite_loop=kwargs.pop('use_infinite_loop', False), \
                                init_offset=self.state['train_iterator_offset'],          \
                                init_reshuffle_count=self.state['train_iterator_reshuffle_count'],       \
                                eos_sym=self.state['eos_sym'],                                           \
                                skip_utterance=self.state.get('skip_utterance', False),                  \
                                skip_utterance_predict_both=self.state.get('skip_utterance_predict_both', False))

        self.batch_iter = None
        self.rng = np.random.RandomState(self.state['seed'])

        # Keep track of previous batch, because this is needed to specify random variables
        self.prev_batch = None
        self.last_returned_offset = 0

    def get_homogenous_batch_iter(self, batch_size = -1):
        while True:
            batch_size = self.batch_size if (batch_size == -1) else batch_size

            data = []
            for k in range(self.k_batches):
                batch = SSIterator.next(self)
                if batch:
                    data.append(batch)

            if not len(data):
                return

            number_of_batches = len(data)
            data = list(itertools.chain.from_iterable(data))

            # Split list of words from the offset index and reshuffle count
            data_x = []
            data_offset = []
            data_reshuffle_count = []
            for i in range(len(data)):
                data_x.append(data[i][0])
                data_offset.append(data[i][1])
                data_reshuffle_count.append(data[i][2])

            if len(data_offset) > 0:
                self.last_returned_offset = data_offset[-1]
                self.last_returned_reshuffle_count = data_reshuffle_count[-1]

            x = np.asarray(list(itertools.chain(data_x)))

            lens = np.asarray([map(len, x)])
            order = np.argsort(lens.max(axis=0))

            for k in range(number_of_batches):
                indices = order[k * batch_size:(k + 1) * batch_size]
                full_batch = create_padded_batch(self.state, self.rng, [x[indices]])

                if full_batch['num_dialogues'] < batch_size:
                    print('Skipping incomplete batch!')
                    continue

                if full_batch['max_length'] < 3:
                    print('Skipping small batch!')
                    continue

                # Then split batches to have size 'max_grad_steps'
                splits = int(math.ceil(float(full_batch['max_length']) / float(self.state['max_grad_steps'])))
                batches = []
                for i in range(0, splits):
                    batch = copy.deepcopy(full_batch)

                    # Retrieve start and end position (index) of current mini-batch
                    start_pos = self.state['max_grad_steps'] * i
                    if start_pos > 0:
                        start_pos = start_pos - 1

                    # We need to copy over the last token from each batch onto the next,
                    # because this is what the model expects.
                    end_pos = min(full_batch['max_length'], self.state['max_grad_steps'] * (i + 1))

                    batch['x'] = full_batch['x'][start_pos:end_pos, :]
                    batch['x_reversed'] = full_batch['x_reversed'][start_pos:end_pos, :]
                    batch['x_mask'] = full_batch['x_mask'][start_pos:end_pos, :]
                    batch['max_length'] = end_pos - start_pos
                    batch['num_preds'] = np.sum(batch['x_mask']) - np.sum(batch['x_mask'][0,:])

                    # For each batch we compute the number of dialogues as a fraction of the full batch,
                    # that way, when we add them together, we get the total number of dialogues.
                    batch['num_dialogues'] = float(full_batch['num_dialogues']) / float(splits)
                    batch['x_reset'] = np.ones(self.state['bs'], dtype='float32')

                    batches.append(batch)

                if len(batches) > 0:
                    batches[-1]['x_reset'] = np.zeros(self.state['bs'], dtype='float32')

                    # Trim the last very short batch
                    if batches[-1]['max_length'] < 3:
                        del batches[-1]
                        batches[-1]['x_reset'] = np.zeros(self.state['bs'], dtype='float32')
                        logger.debug("Truncating last mini-batch...")

                for batch in batches:
                    if batch:
                        yield batch

    def start(self):
        SSIterator.start(self)
        self.batch_iter = None

    def next(self, batch_size = -1):
        """
        We can specify a batch size,
        independent of the object initialization.
        """
        # If there are no more batches in list, try to generate new batches
        if not self.batch_iter:
            self.batch_iter = self.get_homogenous_batch_iter(batch_size)

        try:
            # Retrieve next batch
            batch = next(self.batch_iter)

            # Add Gaussian random variables to batch.
            # We add them separetly for each batch to save memory.
            # If we instead had added them to the full batch before splitting into mini-batches,
            # the random variables would take up several GBs for big batches and long documents.
            batch = add_random_variables_to_batch(self.state, self.rng, batch, self.prev_batch, self.evaluate_mode)
            # Keep track of last batch
            self.prev_batch = batch
        except StopIteration:
            return None
        return batch

    def get_offset(self):
        return self.last_returned_offset

    def get_reshuffle_count(self):
        return self.last_returned_reshuffle_count


def get_train_iterator(state):
    train_data = Iterator(
        state['train_dialogues'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=True,
        max_len=state.get('max_len', -1),
        evaluate_mode=False)

    valid_data = Iterator(
        state['valid_dialogues'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=state.get('max_len', -1),
        evaluate_mode=True)
    return train_data, valid_data


def get_test_iterator(state):
    assert 'test_dialogues' in state

    test_data = Iterator(
        state.get('test_dialogues'),
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=state.get('max_len', -1),
        evaluate_mode=True)
    return test_data
