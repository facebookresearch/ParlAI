# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but they were
used to achieve the pre-trained model.
"""

from parlai.scripts.train_model import setup_args, TrainLoop

def set_defaults(parser):
    """Defaults for baseline model"""

    parser.set_defaults(
        task='fromfile:parlaiformat',
        evaltask='fromfile:parlaiformat2',
        fromfile_datapath='/u/scr/abisee/ParlAI/data/ConvAI2_controllable/train.txt',
        fromfile_datapath2='/u/scr/abisee/ParlAI/data/ConvAI2_controllable/valid.txt',
        model='parlai_internal.projects.controllable_dialog.controllable_seq2seq.controllable_seq2seq:ControllableSeq2seqAgent',
        model_file='/tmp/throwaway',
        dict_file="/u/scr/abisee/control_dialogue_stuff/models/models_public_version/dict_twit30k_train_split",
        dict_lower=True,
        dict_include_valid=True,
        dict_maxexs=-1,
        datatype='train',
        batchsize=64,
        hiddensize=1024,
        embeddingsize=300,
        attention='general',
        numlayers=2,
        rnn_class='lstm',
        learningrate=3,
        dropout=0.1,
        gradient_clip=0.1,
        lookuptable='enc_dec',
        optimizer='sgd',
        embedding_type='glove',
        momentum=0.9,
        bidirectional=False,
        context_length=-1,
        person_tokens=True,
        add_p1_after_newln=True,
        beam_min_n_best=30,
        validation_every_n_secs=90,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=12,
        log_every_n_secs=10,
        dict_tokenizer='split',
        tensorboard_log=True,
        tensorboard_metrics='loss,ppl',
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    parser = set_defaults(parser)
    opt = parser.parse_args()
    TrainLoop(opt).train()
