# Copyright 2004-present Facebook. All Rights Reserved.
import torch
import numpy as np
import logging
import regex
import copy

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from . import config
from .utils import build_feature_dict, vectorize, batchify, normalize_text
from .model import DocReaderModel

logger = logging.getLogger('DrQA')

# ------------------------------------------------------------------------------
# Dictionary.
# ------------------------------------------------------------------------------


class SimpleDictionaryAgent(DictionaryAgent):
    """Override DictionaryAgent to use a simple and fast regex tokenizer.

    * All tokens are NFD unicode normalized.
    * Includes a span_tokenize method.
    * Includes filtering words for which we have no pretrained embedding.
    * Adapted from NLTK RegexpTokenizer.

    Splits in order of:
    1) Digit sequences (maybe with $/,/.)
    3) Unicode alphanumeric/modifier sequences
    4) Punctuation (single chars)
    5) Non-whitespace (single chars)
    """
    DIGIT = r'\$?[\p{Nd}\.\,]+'
    ALPHA = r'[\p{L}\p{N}\p{M}]+'
    PUNCT = r'\p{P}'
    NONWS = r'[^\p{Z}]'

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        argparser.add_arg(
            '--pretrained_words', type='bool', default=True,
            help='Use only words found in provided embedding_file'
        )

    def __init__(self, *args, **kwargs):
        super(SimpleDictionaryAgent, self).__init__(*args, **kwargs)

        # Compile tokenizing regex
        self._regexp = regex.compile(
            '%s|%s|%s|%s' %
            (self.DIGIT, self.ALPHA, self.PUNCT, self.NONWS)
        )

        # Index words in embedding file
        if self.opt['pretrained_words'] and 'embedding_file' in self.opt:
            logger.info('[ Indexing words with embeddings... ]')
            self.embedding_words = set()
            with open(self.opt['embedding_file']) as f:
                for line in f:
                    w = normalize_text(line.rstrip().split(' ')[0])
                    self.embedding_words.add(w)
            logger.info('[ Num words in set = %d ]' %
                        len(self.embedding_words))
        else:
            self.embedding_words = None

    def tokenize(self, text, **kwargs):
        text = normalize_text(text)
        return self._regexp.findall(text)

    def span_tokenize(self, text):
        text = normalize_text(text)
        return [m.span() for m in self._regexp.finditer(text)]

    def add_to_dict(self, tokens):
        """Builds dictionary from the list of provided tokens.
        Only adds words contained in self.embedding_words, if not None.
        """
        for token in tokens:
            if (self.embedding_words is not None and
                token not in self.embedding_words):
                continue
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token


# ------------------------------------------------------------------------------
# Document Reader.
# ------------------------------------------------------------------------------


class DocReaderAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None, word_dict=None):
        # Only create an empty dummy class when sharing
        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        self.id = self.__class__.__name__
        self.opt = copy.deepcopy(opt)
        config.set_defaults(self.opt)

        if 'pretrained_model' in self.opt:
            logger.info('[ Loading params from %s ]' %
                        self.opt['pretrained_model'])
            params = torch.load(self.opt['pretrained_model'])
            self.state_dict = params['state_dict']
            self.dict = params['word_dict']
            self.feature_dict = params['feature_dict']
            config.override_args(self.opt, params['config'])
        else:
            self.state_dict = None
            self.dict = word_dict
            self.feature_dict = build_feature_dict(self.opt)

        self.opt['vocab_size'] = len(self.dict)
        self.opt['num_features'] = len(self.feature_dict)

        # Intialize model
        logger.info('[ Initializing DocReaderModel ]')
        self.model = DocReaderModel(self.opt, self.dict, self.state_dict)
        self.n_examples = 0

    def act(self):
        """Update or predict on a single example (batchsize = 1)."""
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        reply = {'id': self.getID()}

        ex = self._build_ex(self.observation)
        if ex is None:
            return reply
        batch = batchify([ex], null=self.dict['<NULL>'])

        # Either train or predict
        if 'labels' in self.observation:
            self.n_examples += 1
            self.model.update(batch)
            self._log()
        else:
            reply['text'] = self.model.predict(batch)[0]

        return reply

    def batch_act(self, observations):
        """Update or predict on a batch of examples.
        More efficient than act().
        """
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # Some examples will be None (no answer found). Filter them.
        examples = [self._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batchsize) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]

        # If all examples are invalid, return an empty batch.
        if len(examples) == 0:
            return batch_reply

        # Else, use what we have (hopefully everything).
        batch = batchify(examples, null=self.dict['<NULL>'])

        # Either train or predict
        if 'labels' in observations[0]:
            self.n_examples += len(examples)
            self.model.update(batch)
            self._log()
        else:
            predictions = self.model.predict(batch)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]

        return batch_reply

    def save(self, filename):
        """Save the parameters of the agent to a file."""
        params = {
            'state_dict': {
                'network': self.model.network.state_dict(),
                'optimizer': self.model.optimizer.state_dict(),
            },
            'word_dict': self.dict,
            'feature_dict': self.feature_dict,
            'config': self.opt,
        }
        torch.save(params, filename)

    # --------------------------------------------------------------------------
    # Helper functions.
    # --------------------------------------------------------------------------

    def _build_ex(self, ex):
        """Find the token span of the answer in the context for this example.
        If a token span cannot be found, return None. Otherwise, torchify.
        """
        # Split out document + question
        inputs = {}
        fields = ex['text'].split('\n')
        document, question = ' '.join(fields[:-1]), fields[-1]
        inputs['document'] = self.dict.tokenize(document)
        inputs['question'] = self.dict.tokenize(question)
        inputs['target'] = None

        # Find targets (if labels provided).
        # Return if we were unable to find an answer.
        if 'labels' in ex:
            inputs['target'] = self._find_target(inputs['document'],
                                                 ex['labels'])
            if inputs['target'] is None:
                return

        # Vectorize.
        inputs = vectorize(self.opt, inputs, self.dict, self.feature_dict)

        # Return inputs with original text + spans (keep for prediction)
        return inputs + (document, self.dict.span_tokenize(document))

    def _find_target(self, document, labels):
        """Find the start/end token span for all labels in document.
        Return a random one for training.
        """
        def _positions(d, l):
            for i in range(len(d)):
                for j in range(i, min(len(d) - 1, i + len(l))):
                    if l == d[i:j + 1]:
                        yield(i, j)
        targets = []
        for label in labels:
            targets.extend(_positions(document, self.dict.tokenize(label)))
        if len(targets) == 0:
            return
        return targets[np.random.choice(len(targets))]

    def _log(self):
        if self.model.updates % self.opt['display_iter'] == 0:
            logger.info(
                '[train] updates = %d | train loss = %.2f | exs = %d' %
                (self.model.updates, self.model.train_loss.avg, self.n_examples)
            )
