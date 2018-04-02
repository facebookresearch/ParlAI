from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from examples.eval_model import eval_model
from projects.personachat.persona_seq2seq import PersonachatSeqseqAgentSplit

"""Evaluate pre-trained profile memory model trained on ConvAI2 using persona 'self'
"""

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-n', '--num-examples', default=100000000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    PersonachatSeqseqAgentSplit.add_cmdline_args(parser)
    parser.set_defaults(
        dict_file='models:convai2/profilememory/profilememory_convai2.dict',
        rank_candidates=True,
        task='convai2:self',
        model='projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit',
        model_file='models:convai2/profilememory/profilememory_convai2_model',
        datatype='test'
    )

    opt = parser.parse_args()
    opt['model_type'] = 'profilememory'
    # build profile memory models
    fnames = ['profilememory_convai2_model',
              'profilememory_convai2.dict']
    download_models(opt, fnames, 'convai2', use_model_type=True)

    eval_model(parser)
