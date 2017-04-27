# Copyright 2004-present Facebook. All Rights Reserved.
import torch
import time
import unicodedata
from collections import Counter


# ------------------------------------------------------------------------------
# Data/model utilities.
# ------------------------------------------------------------------------------


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors."""
    embeddings = torch.Tensor(len(word_dict), opt['embedding_dim'])
    embeddings.normal_(0, 1)

    # Fill in embeddings
    with open(opt['embedding_file']) as f:
        for line in f:
            parsed = line.rstrip().split(' ')
            assert(len(parsed) == opt['embedding_dim'] + 1)
            w = normalize_text(parsed[0])
            if w in word_dict:
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                embeddings[word_dict[w]].copy_(vec)

    # Zero NULL token
    embeddings[word_dict['<NULL>']].fill_(0)

    return embeddings


def build_feature_dict(opt):
    """Make mapping of feature option to feature index."""
    feature_dict = {}
    if opt['use_in_question']:
        feature_dict['in_question'] = len(feature_dict)
        feature_dict['in_question_uncased'] = len(feature_dict)
    if opt['use_tf']:
        feature_dict['tf'] = len(feature_dict)
    return feature_dict


# ------------------------------------------------------------------------------
# Torchified input utilities.
# ------------------------------------------------------------------------------


def vectorize(opt, ex, word_dict, feature_dict):
    """Turn tokenized text inputs into feature vectors."""
    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    # Create extra features vector
    features = torch.zeros(len(ex['document']), len(feature_dict))

    # f_{exact_match}
    if opt['use_in_question']:
        q_words_cased = set([w for w in ex['question']])
        q_words_uncased = set([w.lower() for w in ex['question']])
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0

    # f_{tf}
    if opt['use_tf']:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if ex['target'] is None:
        return document, features, question

    # ...or with target
    start = torch.LongTensor(1).fill_(ex['target'][0])
    end = torch.LongTensor(1).fill_(ex['target'][1])

    return document, features, question, start, end


def batchify(batch, null=0):
    """Collate inputs into batches."""
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 2

    # Get elements
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    text = [ex[-2] for ex in batch]
    spans = [ex[-1] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).fill_(null).pin_memory()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1).pin_memory()
    x1_f = torch.zeros(len(docs), max_length, features[0].size(1)).pin_memory()
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        x1_f[i, :d.size(0)].copy_(features[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).fill_(null).pin_memory()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1).pin_memory()
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x2, x2_mask, text, spans

    # ...Otherwise add targets
    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        y_s = torch.cat([ex[3] for ex in batch])
        y_e = torch.cat([ex[4] for ex in batch])
        return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, text, spans

    # ...Otherwise wrong number of inputs
    raise RuntimeError('Wrong number of inputs per batch')


# ------------------------------------------------------------------------------
# General logging utilities.
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
