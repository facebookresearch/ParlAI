#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from parlai.core.agents import create_agent
import torch.distributed as dist
import parlai.utils.testing as testing_utils
import parlai.scripts.multiprocessing_train as mp_train
import parlai.scripts.build_dict as build_dict
import os
import copy


class TestHuggingFaceDict(unittest.TestCase):
    def test_custom_special_tokens(self):
        from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent
        from parlai.core.params import ParlaiParser

        parser = ParlaiParser(False, False)
        parser.set_defaults(gpt2_size="small", add_special_tokens=True)
        Gpt2DictionaryAgent.add_cmdline_args(parser)
        with testing_utils.tempdir() as tmpdir:
            opt = parser.parse_kwargs(dict_file=os.path.join(tmpdir, 'dict'))
            dict_agent = Gpt2DictionaryAgent(opt)
            oldtokens = dict_agent.txt2vec("Hi VOLDEMORT")
            prevlen = len(dict_agent)
            dict_agent.add_additional_special_tokens(["VOLDEMORT"])
            newlen = len(dict_agent)
            assert newlen == prevlen + 1
            tokens = dict_agent.txt2vec("Hi VOLDEMORT")
            assert tokens != oldtokens
            assert len(tokens) < len(oldtokens)


class TestGpt2(unittest.TestCase):
    def _test_batchsize(self, batchsize, add_start_token):
        utterances = [
            'Just keep swimming -',
            'I wish I knew how to quit you. -',
            "I'm going to make him an offer he can't refuse. -",
            "Toto, I've got a feeling we're not in Kansas anymore. -",
        ]
        opt = {
            'model': 'hugging_face/gpt2',
            'gpt2_size': 'small',
            'text_truncate': 16,
            'label_truncate': 8,
            'beam_min_length': 8,
            'inference': 'beam',
            'beam_size': 1,
            'add_special_tokens': True,
            'batchsize': batchsize,
            'add_start_token': add_start_token,
        }
        gpt2 = create_agent(opt)

        results_single = []
        agents = [gpt2.clone() for _ in utterances]
        for u, a in zip(utterances, agents):
            a.observe({'text': u, 'episode_done': True})
            generation = a.act()['text']
            results_single.append(generation)

        results_batched = []
        for idx in range(len(utterances) // batchsize):
            agents = [gpt2.clone() for _ in range(batchsize)]
            batch = utterances[idx * batchsize : (idx + 1) * batchsize]
            obs = []
            for i, a in enumerate(agents):
                obs.append(a.observe({'text': batch[i], 'episode_done': True}))
            generations = [x['text'] for x in gpt2.batch_act(obs)]
            results_batched += generations

        assert results_single == results_batched

    def test_start_token(self):
        with self.assertRaises(RuntimeError):
            create_agent(
                {
                    'model': 'hugging_face/gpt2',
                    'add_special_tokens': False,
                    'add_start_token': True,
                }
            )

    def test_batchsize(self):
        """
        Ensures gpt2 provides the same generation results regardless of batchsize.
        """
        for add_start_token in [True, False]:
            for batchsize in [1, 2, 4]:
                with self.subTest(
                    f'test_batchsize with bs={batchsize} and ast={add_start_token}'
                ):
                    self._test_batchsize(batchsize, add_start_token)

    def test_nospecialtok(self):
        with self.assertRaises(RuntimeError):
            create_agent(
                {
                    'model': 'hugging_face/gpt2',
                    'add_special_tokens': False,
                    'batchsize': 2,
                }
            )

        opt = {
            'model': 'hugging_face/gpt2',
            'gpt2_size': 'small',
            'text_truncate': 16,
            'label_truncate': 8,
            'beam_min_length': 8,
            'inference': 'beam',
            'beam_size': 1,
            'batchsize': 1,
            'add_special_tokens': False,
        }
        gpt2 = create_agent(opt)
        gpt2.observe({'text': 'My name is', 'episode_done': True})
        response = gpt2.act()
        assert response['text'] == " John. I'm a man of"


@testing_utils.skipUnlessGPU
class TestDistributed(unittest.TestCase):
    _base_config = {
        'task': 'integration_tests:overfit',
        'model': 'hugging_face/gpt2',
        'gpt2_size': 'small',
        'text_truncate': 16,
        'label_truncate': 8,
        'beam_min_length': 8,
        'inference': 'beam',
        'beam_size': 1,
        'batchsize': 4,
        'add_special_tokens': True,
        'validation_metric': 'ppl',
    }

    def setUp(self):
        print(f'[Setting up test {self._testMethodName}]')

    def _forced_parse(self, parser, opt):
        # TODO: Kill this after dictionaries build correctly
        parser.set_params(**opt)
        parser.set_params(log_every_n_sec=10)
        popt = parser.parse_args([])
        # in some rare cases, like for instance if the model class also
        # overrides its default params, the params override will not
        # be taken into account.
        for k, v in opt.items():
            popt[k] = v
        return popt

    def _distributed_train_model(self, opt):
        with testing_utils.tempdir() as tmpdir:
            if 'model_file' not in opt:
                opt['model_file'] = os.path.join(tmpdir, 'model')
            if 'dict_file' not in opt:
                opt['dict_file'] = os.path.join(tmpdir, 'model.dict')

            parser = mp_train.setup_args()
            # TODO: Kill this after dictionaries build correctly
            popt = self._forced_parse(parser, opt)

            # we need a prebuilt dictionary
            parser = build_dict.setup_args()
            build_dict.build_dict(popt)

            valid, test = mp_train.launch_and_train(popt, 31338)
            dist.destroy_process_group()

        return (valid, test)

    @testing_utils.retry()
    def test_distributed(self):
        config = copy.deepcopy(self._base_config)
        config['num_epochs'] = 50
        config['task'] = 'integration_tests:overfit'
        config['batchsize'] = 2
        config['dropout'] = 0.0
        config['attention_dropout'] = 0.0
        config['learningrate'] = 1.0
        config['momentum'] = 0.90
        config['skip_generation'] = True
        valid, test = self._distributed_train_model(config)

        self.assertLessEqual(valid['ppl'], 10)
        self.assertLessEqual(test['ppl'], 10)
