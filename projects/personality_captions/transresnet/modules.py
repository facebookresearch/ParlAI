#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Model Code.
"""

import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
from parlai.utils.io import PathManager


class TransresnetModel(nn.Module):
    """
    Actual model code for the Transresnet Agent.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add command line arguments.
        """
        Transformer.add_common_cmdline_args(argparser)
        agent = argparser.add_argument_group('TransresnetModel arguments')
        agent.add_argument(
            '--truncate',
            type=int,
            default=32,
            help='Max amount of tokens allowed in a text sequence',
        )
        agent.add_argument(
            '--image-features-dim',
            type=int,
            default=2048,
            help='dimensionality of image features',
        )
        agent.add_argument(
            '--embedding-type',
            type=str,
            default=None,
            choices=[None, 'fasttext_cc'],
            help='Specify if using pretrained embeddings',
        )
        agent.add_argument(
            '--load-encoder-from',
            type=str,
            default=None,
            help='Specify if using a pretrained transformer encoder',
        )
        agent.add_argument(
            '--hidden-dim',
            type=int,
            default=300,
            help='Hidden dimesionality of personality and image encoder',
        )
        agent.add_argument(
            '--num-layers-all',
            type=int,
            default=-1,
            help='If >= 1, number of layers for both the text ' 'and image encoders.',
        )
        agent.add_argument(
            '--num-layers-text-encoder',
            type=int,
            default=1,
            help='Number of layers for the text encoder',
        )
        agent.add_argument(
            '--num-layers-image-encoder',
            type=int,
            default=1,
            help='Number of layers for the image encoder',
        )
        agent.add_argument(
            '--no-cuda',
            dest='no_cuda',
            action='store_true',
            help='If True, perform ops on CPU only',
        )
        agent.add_argument(
            '--learningrate',
            type=float,
            default=0.0005,
            help='learning rate for optimizer',
        )
        agent.add_argument(
            '--additional-layer-dropout',
            type=float,
            default=0.2,
            help='dropout for additional linear layer',
        )
        argparser.set_params(
            ffn_size=1200, attention_dropout=0.2, relu_dropout=0.2, n_positions=1000
        )

    def __init__(self, opt, personalities_list, dictionary):
        super().__init__()
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        self.opt = opt
        self.dictionary = dictionary
        self.truncate_length = opt['truncate']
        if opt['num_layers_all'] != -1:
            n_layers_text = n_layers_img = opt['num_layers_all']
        else:
            n_layers_text = opt['num_layers_text_encoder']
            n_layers_img = opt['num_layers_image_encoder']
        self.text_encoder_frozen = False

        # Initialize personalities dictionary
        self._build_personality_dictionary(personalities_list)

        # Text encoder
        self._build_text_encoder(n_layers_text)

        # Image encoder
        self._build_image_encoder(n_layers_img)

        # Personality Encoder
        self._build_personality_encoder()

        # optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.opt['learningrate'],
        )

    def _build_personality_dictionary(self, personalities_list):
        """
        Build the personality dictionary mapping personality to id.

        :param personalities_list:
            list of personalities
        """
        self.personalities_list = personalities_list
        self.personality_to_id = {p: i for i, p in enumerate(personalities_list)}
        self.num_personalities = len(self.personalities_list) + 1

    def _build_text_encoder(self, n_layers_text):
        """
        Build the text (candidate) encoder.

        :param n_layers_text:
            how many layers the transformer will have
        """
        self.embeddings = nn.Embedding(len(self.dictionary), self.opt['embedding_size'])
        if (
            self.opt.get('load_encoder_from') is None
            and self.opt['embedding_type'] == 'fasttext_cc'
        ):
            self.embeddings = load_fasttext_embeddings(
                self.dictionary, self.opt['embedding_size'], self.opt['datapath']
            )

        self.text_encoder = TransformerEncoder(
            n_heads=self.opt['n_heads'],
            n_layers=self.opt['n_layers'],
            embedding_size=self.opt['embedding_size'],
            ffn_size=self.opt['ffn_size'],
            vocabulary_size=len(self.dictionary),
            embedding=self.embeddings,
            dropout=self.opt['dropout'],
            attention_dropout=self.opt['attention_dropout'],
            relu_dropout=self.opt['relu_dropout'],
            padding_idx=self.dictionary.tok2ind[self.dictionary.null_token],
            learn_positional_embeddings=self.opt['learn_positional_embeddings'],
            embeddings_scale=False,
            n_positions=self.opt['n_positions'],
            activation=self.opt['activation'],
            variant=self.opt['variant'],
            n_segments=self.opt['n_segments'],
        )
        if self.opt.get('load_encoder_from') is not None:
            self._load_text_encoder_state()

        self.additional_layer = LinearWrapper(
            self.opt['embedding_size'],
            self.opt['hidden_dim'],
            dropout=self.opt['additional_layer_dropout'],
        )

    def _build_image_encoder(self, n_layers_img):
        """
        Build the image encoder mapping raw image features to the appropriate space.

        :param n_layers_img:
            number of feed-forward layers for the image encoder
        """
        image_layers = [
            nn.BatchNorm1d(self.opt['image_features_dim']),
            nn.Dropout(p=self.opt['dropout']),
            nn.Linear(self.opt['image_features_dim'], self.opt['hidden_dim']),
        ]
        for _ in range(n_layers_img - 1):
            image_layers += [
                nn.ReLU(),
                nn.Dropout(p=self.opt['dropout']),
                nn.Linear(self.opt['hidden_dim'], self.opt['hidden_dim']),
            ]
        self.image_encoder = nn.Sequential(*image_layers)

    def _build_personality_encoder(self):
        personality_layers = [
            nn.BatchNorm1d(self.num_personalities),
            nn.Dropout(p=self.opt['dropout']),
            nn.Linear(self.num_personalities, self.opt['hidden_dim']),
        ]
        self.personality_encoder = nn.Sequential(*personality_layers)

    def forward(
        self, image_features, personalities, captions, personalities_tensor=None
    ):
        """
        Model forward pass.

        :param image_features:
            list of tensors of image features, one per example
        :param personalities:
            list of personalities, one per example
        :param captions:
            list of captions, one per example
        :param personalities_tensor:
            (optional) list of personality representations, usually a one-hot
            vector if specified

        :return:
            the encoded context and the encoded captions.
        """
        captions_encoded = None
        context_encoded = None
        img_encoded = None

        # encode captions
        if captions is not None:
            indexes, mask = self.captions_to_tensor(captions)
            captions_encoded = self.text_encoder(indexes)
            if self.text_encoder_frozen:
                captions_encoded = captions_encoded.detach()
            captions_encoded = self.additional_layer(captions_encoded)

        # encode personalities
        pers_encoded = self.forward_personality(personalities, personalities_tensor)

        # encode images
        img_encoded = self.forward_image(image_features)

        context_encoded = self.sum_encodings([pers_encoded, img_encoded])
        return context_encoded, captions_encoded

    def forward_personality(self, personalities, personalities_tensor):
        """
        Encode personalities.

        :param personalities:
            list of personalities, one per example
        :param personalities_tensor:
            (optional) list of personality representations, usually a one-hot
            vector if specified

        :return:
            encoded representation of the personalities
        """
        pers_encoded = None
        if personalities is not None:
            if personalities_tensor is not None:
                pers_feature = personalities_tensor
            else:
                res = torch.FloatTensor(
                    len(personalities), self.num_personalities
                ).fill_(0)
                p_to_i = self.personalities_to_index(personalities)
                for i, index in enumerate(p_to_i):
                    res[i, index] = 1  # no personality corresponds to 0
                if self.use_cuda:
                    res = res.cuda()
                pers_feature = res
            pers_encoded = self.personality_encoder(pers_feature)

        return pers_encoded

    def forward_image(self, image_features):
        """
        Encode image features.

        :param image_features:
            list of image features

        :return:
            encoded representation of the image features
        """
        img_encoded = None
        if image_features is not None:
            stacked = torch.stack(image_features)
            if self.use_cuda:
                stacked = stacked.cuda()
            img_encoded = self.image_encoder(stacked)

        return img_encoded

    def train_batch(self, image_features, personalities, captions):
        """
        Batch train on a set of examples.

        Uses captions from other examples as negatives during training

        :param image_features:
            list of tensors of image features
        :param personalities:
            list of personalities
        :param captions:
            list of captions

        :return:
            the total loss, the number of correct examples, and the number of
            examples trained on
        """
        self.zero_grad()
        self.train()
        context_encoded, captions_encoded = self.forward(
            image_features, personalities, captions
        )
        loss, num_correct = self.evaluate_one_batch(
            context_encoded, captions_encoded, during_train=True
        )
        loss.backward()
        self.optimizer.step()

        # re-run forward pass to compute hits@1 metrics
        loss, num_correct, num_examples = self.eval_batch_of_100(
            context_encoded, captions_encoded
        )
        return loss, num_correct, num_examples

    def eval_batch(self, image_features, personalities, captions):
        """
        Evaluate performance of model on one batch.

        Batch is split into chunks of 100 to evaluate hits@1/100

        :param image_features:
            list of tensors of image features
        :param personalities:
            list of personalities
        :param captions:
            list of captions

        :return:
            the total loss, the number of correct examples, and the number of
            examples trained on
        """
        if personalities is None:
            personalities = [''] * len(image_features)
        if len(image_features) == 0:
            return 0, 0, 1
        self.eval()
        context_encoded, captions_encoded = self.forward(
            image_features, personalities, captions
        )
        loss, num_correct, num_examples = self.eval_batch_of_100(
            context_encoded, captions_encoded
        )
        return loss, num_correct, num_examples

    def choose_best_caption(
        self, image_features, personalities, candidates, candidates_encoded=None, k=1
    ):
        """
        Choose the best caption for each example.

        :param image_features:
            list of tensors of image features
        :param personalities:
            list of personalities
        :param candidates:
            list of candidates, one set per example
        :param candidates_encoded:
            optional; if specified, a fixed set of encoded candidates that is
            used for each example
        :param k:
            number of ranked candidates to return. if < 1, we return the ranks
            of all candidates in the set.

        :return:
            a set of ranked candidates for each example
        """
        self.eval()
        context_encoded, _ = self.forward(image_features, personalities, None)
        context_encoded = context_encoded.detach()
        one_cand_set = True
        if candidates_encoded is None:
            one_cand_set = False
            candidates_encoded = [
                self.forward(None, None, c)[1].detach() for c in candidates
            ]
        chosen = []
        for img_index in range(len(context_encoded)):
            context_encoding = context_encoded[img_index : img_index + 1, :]
            scores = torch.mm(
                candidates_encoded[img_index].to(context_encoding)
                if not one_cand_set
                else candidates_encoded.to(context_encoding),
                context_encoding.transpose(0, 1),
            )
            if k >= 1:
                _, index_top = torch.topk(scores, k, dim=0)
            else:
                _, index_top = torch.topk(scores, scores.size(0), dim=0)
            chosen.append(
                [
                    candidates[img_index][idx] if not one_cand_set else candidates[idx]
                    for idx in index_top.unsqueeze(1)
                ]
            )

        return chosen

    def eval_batch_of_100(self, context_encoded, captions_encoded):
        """
        Evaluate a batch of 100 examples.

        The captions of the other examples are used as negatives.

        :param context_encoded:
            the encoded context
        :param captions_encoded:
            the encoded captions

        :return:
            the total loss, the total number of correct examples, and the
            total number of examples evaluated.
        """
        total_loss = 0
        total_ok = 0
        num_examples = 0
        for i in range(0, len(context_encoded), 100):
            if i + 100 > len(context_encoded):
                break
            num_examples += 100
            loss, num_correct = self.evaluate_one_batch(
                context_encoded[i : i + 100, :], captions_encoded[i : i + 100, :]
            )
            total_loss += loss.data.cpu().item()
            total_ok += num_correct.data.cpu().item()
        return total_loss, total_ok, num_examples

    def evaluate_one_batch(self, context_encoded, captions_encoded, during_train=False):
        """
        Compute loss - and number of correct examples - for one batch.

        :param context_encoded:
            the encoded context
        :param captions_encoded:
            the encoded captions
        :param during_train:
            true if training, else False

        :return:
            the batch loss and the number of correct examples
        """
        if not during_train:
            self.zero_grad()
            self.eval()
        dot_products = context_encoded.mm(captions_encoded.t())
        log_prob = torch.nn.functional.log_softmax(dot_products, dim=1)
        targets = torch.arange(0, len(context_encoded), dtype=torch.long)
        if self.use_cuda:
            targets = targets.cuda()
        loss = torch.nn.functional.nll_loss(log_prob, targets)
        num_correct = (log_prob.max(dim=1)[1] == targets).float().sum()
        return loss, num_correct

    def freeze_text_encoder(self):
        """
        Freeze the text (candidate) encoder.
        """
        self.text_encoder_frozen = True

    def unfreeze_text_encoder(self):
        """
        Unfreeze the text (candidate) encoder.
        """
        self.text_encoder_frozen = False

    def sum_encodings(self, addends):
        """
        Add up a list of encodings, some of which may be `None`.

        :param addends:
            tensors to add

        :return:
            sum of non-`None` addends
        """
        addends = [a for a in addends if a is not None]
        return sum(addends) if len(addends) > 0 else None

    def personalities_to_index(self, personalities):
        """
        Map personalities to their index in the personality dictionary.

        :param personalities:
            list of personalities

        :return:
            list of personality ids
        """
        res = []
        for p in personalities:
            if p in self.personality_to_id:
                res.append(self.personality_to_id[p] + 1)
            else:
                res.append(0)
        return res

    def captions_to_tensor(self, captions):
        """
        Tokenize a list of sentences into a 2D float tensor.

        :param captions:
            list of sentences to tokenize

        :return:
            a (batchsize X truncate_length) tensor representation of the captions,
            and a similarly sized mask tensor
        """
        max_length = self.truncate_length
        indexes = []
        for c in captions:
            vec = self.dictionary.txt2vec(c)
            if len(vec) > max_length:
                vec = vec[:max_length]
            indexes.append(self.dictionary.txt2vec(c))
        longest = max([len(v) for v in indexes])
        res = torch.LongTensor(len(captions), longest).fill_(
            self.dictionary.tok2ind[self.dictionary.null_token]
        )
        mask = torch.FloatTensor(len(captions), longest).fill_(0)
        for i, inds in enumerate(indexes):
            res[i, 0 : len(inds)] = torch.LongTensor(inds)
            mask[i, 0 : len(inds)] = torch.FloatTensor([1] * len(inds))
        if self.use_cuda:
            res = res.cuda()
            mask = mask.cuda()
        return res, mask

    def _load_text_encoder_state(self):
        try:
            state_file = self.opt.get('load_encoder_from')
            with PathManager.open(state_file, 'b') as f:
                model = torch.load(f)
            states = model['model']
            self.text_encoder.load_state_dict(states)
        except Exception as e:
            print(
                'WARNING: Cannot load transformer state; please make sure '
                'specified file is a dictionary with the states in `model`. '
                'Additionally, make sure that the appropriate options are '
                'specified. Error: {}'.format(e)
            )


def load_fasttext_embeddings(dic, embedding_dim, datapath):
    """
    Load weights from fasttext_cc and put them in embeddings.weights.
    """
    print('Initializing embeddings from fasttext_cc')
    from parlai.zoo.fasttext_cc_vectors.build import download

    pretrained = download(datapath)
    print(
        'Done Loading vectors from fasttext. {} embeddings loaded.'.format(
            len(pretrained)
        )
    )
    used = 0
    res = nn.Embedding(len(dic), embedding_dim)
    for word in dic.tok2ind.keys():
        index = dic.tok2ind[word]
        if word in pretrained and res.weight.data.shape[0] > index:
            res.weight.data[index] = pretrained[word]
            used += 1
    print('{} have been initialized on pretrained over {} words'.format(used, len(dic)))
    return res


class LinearWrapper(nn.Module):
    """
    Linear layer with dropout.
    """

    def __init__(self, in_dim, out_dim, dropout):
        super(LinearWrapper, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, input):
        """
        Forward pass.
        """
        return self.lin(self.dp(input))
