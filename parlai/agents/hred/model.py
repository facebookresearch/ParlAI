import logging
import numpy
import theano
logger = logging.getLogger(__name__)

# This is the list of strings required to ignore, if we're going to take a pretrained HRED model 
# and fine-tune it as a variational model.
# parameter_strings_to_ignore = ["latent_utterance_prior", "latent_utterance_approx_posterior", "Wd_", "bd_"]


class Model(object):
    def __init__(self):
        self.floatX = theano.config.floatX
        # Parameters of the model
        self.params = []
    
    def save(self, filename):
        """
        Save the model to file `filename`
        """
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, **vals)

    def load(self, filename, parameter_strings_to_ignore=[]):
        """
        Load the model.

        Any parameter which has one of the strings inside parameter_strings_to_ignore as a substring,
        will not be loaded from the file (but instead initialized as a new model, which usually means random).
        """
        vals = numpy.load(filename)
        for p in self.params:
            load_parameter = True
            for string_to_ignore in parameter_strings_to_ignore:
                if string_to_ignore in p.name:
                     logger.debug('Initializing parameter {} as in new model'.format(p.name))
                     load_parameter = False

            if load_parameter:
                if p.name in vals:
                    logger.debug('Loading {} of {}'.format(p.name, p.get_value(borrow=True).shape))
                    if p.get_value().shape != vals[p.name].shape:
                        raise Exception('Shape mismatch: {} != {} for {}'.format(p.get_value().shape, vals[p.name].shape, p.name))
                    p.set_value(vals[p.name])
                else:
                    logger.error('No parameter {} given: default initialization used'.format(p.name))
                    unknown = set(vals.keys()) - {p.name for p in self.params}
                    if len(unknown):
                        logger.error('Unknown parameters {} given'.format(unknown))
