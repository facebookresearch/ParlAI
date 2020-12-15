#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension

See https://arxiv.org/abs/1910.13461.

The BART agent can be instantiated as simply `-m bart`,
however it is recommended to specify `--init-model zoo:bart/bart_large/model`
or `-mf zoo:bart/bart_large/model` to ensure correct dictionaries are saved.
"""
import os
import torch
from typing import Optional, Dict, Any

from parlai.agents.bart.convert_fairseq_to_parlai import ConversionScript
from parlai.agents.bart.modules import BartModel
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import compare_init_model_opts
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import History
from parlai.core.torch_generator_agent import PPLMetric
from parlai.core.metrics import AverageMetric
from parlai.utils.typing import TShared
from parlai.utils.io import PathManager
from parlai.zoo.bart.build import download, CONVERSION_ARGS, BART_ARGS


class BartAgent(TransformerGeneratorAgent):
    """
    BART Agent.

    Relies on the BART model implemented in fairseq.

    If you have a fine-tuned BART model from fairseq, you can specify the
    `--init-fairseq-model` arg, which will convert your fine-tuned model
    to a ParlAI model.
    """

    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        """
        Override to add init-fairseq-model arg.
        """
        TransformerGeneratorAgent.add_cmdline_args(argparser)
        group = argparser.add_argument_group('Bart Args')
        group.add_argument(
            '--init-fairseq-model',
            type=str,
            default=None,
            help='fairseq checkpoint for bart',
        )
        group.add_argument(
            '--output-conversion-path',
            type=str,
            default=None,
            help='where to save fairseq conversion',
        )
        argparser.set_defaults(dict_tokenizer='gpt2')
        argparser.set_defaults(**BART_ARGS)

    def __init__(self, opt: Opt, shared: TShared = None):
        if not shared:
            opt = self._initialize_bart(opt)
        super().__init__(opt, shared)

    def _initialize_bart(self, opt: Opt) -> Opt:
        """
        Download and convert BART pre-trained models.

        Additionally, convert `init-fairseq-model` if necessary.

        :param opt:
            ParlAI-parsed options

        :return opt:
            return opt with BART-specific args.
        """
        init_model, _ = self._get_init_model(opt, None)
        if not opt.get('converting') and (
            init_model is None or not PathManager.exists(init_model)
        ):
            download(opt['datapath'])
            opt['init_model'] = os.path.join(
                opt['datapath'], 'models/bart/bart_large/model'
            )
        if opt.get('init_fairseq_model'):
            opt = self._convert_model(opt)

        compare_init_model_opts(opt, opt)
        return opt

    def _get_conversion_args(self, opt: Opt) -> Dict[str, Any]:
        """
        Get args for fairseq model conversion.

        :param opt:
            ParlAI Opt

        :return args:
            returns dictionary of args to send to conversion script.
        """
        model_name = os.path.split(opt['init_fairseq_model'])[-1]
        args = CONVERSION_ARGS.copy()

        args['input'] = [opt['init_fairseq_model']]
        if opt.get('model_file') and not os.path.exists(opt['model_file']):
            args['output'] = opt['model_file']
        elif opt.get('output_conversion_path'):
            args['output'] = opt['output_conversion_path']
        else:
            args['output'] = os.path.join(
                opt['datapath'], 'models/converted_fairseq_models/', model_name
            )

        return args

    def _convert_model(self, opt: Opt) -> Opt:
        """
        Convert fairseq init model to ParlAI Model.

        :param opt:
            options

        :return opt:
            return opt with new init_model path
        """
        args = self._get_conversion_args(opt)
        ConversionScript.main(**args)
        opt['init_model'] = args['output']
        return opt

    def build_model(self) -> BartModel:
        """
        Build and return model.
        """
        model = BartModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def _set_text_vec(
        self, obs: Message, history: History, truncate: Optional[int]
    ) -> Message:
        """
        Override to prepend start token and append end token.
        """
        obs = super()._set_text_vec(obs, history, truncate)
        if 'text' not in obs or 'text_vec' not in obs:
            return obs
        vec = obs['text_vec']
        if truncate is not None:
            vec = torch.LongTensor(  # type: ignore
                self._check_truncate(obs['text_vec'], truncate - 2, True)
            )
        obs.force_set(
            'text_vec', self._add_start_end_tokens(vec, add_start=True, add_end=True)
        )
        return obs

    def _get_initial_decoder_input(
        self, bsz: int, beam_size: int, dev: torch.device
    ) -> torch.LongTensor:
        """
        Override to seed decoder with EOS BOS token.
        """
        return (
            torch.LongTensor(  # type: ignore
                [self.END_IDX, self.START_IDX]
            )
            .expand(bsz * beam_size, 2)
            .to(dev)
        )

    def compute_loss(self, batch, return_output=False):
        """
        Override TGA.compute_loss to ignore start token.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output

        if scores.size(1) != batch.label_vec.size(1):
            # ignore start
            scores = scores[:, 1:, :]
            preds = preds[:, 1:]

        score_view = scores.reshape(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def _construct_token_losses(self, labels, model_output):
        """
        Override TGA._construct_token_losses to ignore start token.
        """
        # Get non-aggregated losses
        scores, _, _ = model_output
        scores = scores[:, 1:, :]  # ignore start token
        score_view = scores.reshape(-1, scores.size(-1))
        losses = self.criterion(score_view, labels.view(-1)).view(len(labels), -1)

        # Zip decoded tokens with losses
        token_losses = []
        for i, label in enumerate(labels):
            token_losses.append(
                list(
                    zip(
                        [self.dict[token] for token in label.tolist()],
                        losses[i].tolist(),
                    )
                )
            )
        return token_losses
