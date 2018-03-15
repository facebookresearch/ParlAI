from download_models import build
from parlai.core.params import ParlaiParser
from examples.eval_model import eval_model
from projects.personachat.persona_seq2seq import PersonachatSeqseqAgentBasic

'''Evaluate pre-trained model trained for hits@1 metric
Generative model trained on personachat using persona 'self'
Run from ParlAI directory
'''

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-n', '--num-examples', default=100000000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.set_defaults(
        task='personachat:self',
        model='projects.personachat.persona_seq2seq:PersonachatSeqseqAgentBasic',
        model_file='data/models/personachat/seq2seq_personachat/seq2seq_no_dropout0.2_lstm_1024_1e-3',
        datatype='test'
    )
    PersonachatSeqseqAgentBasic.add_cmdline_args(parser)

    opt = parser.parse_args()
    opt['model_type'] = 'seq2seq_personachat' # for builder
    # build all profile memory models
    fnames = ['seq2seq_no_dropout0.2_lstm_1024_1e-3',
              'fulldict.dict']
    build(opt, fnames)

    # add additional model args
    opt['dict_file'] = 'data/models/personachat/seq2seq_personachat/fulldict.dict'
    opt['rank_candidates'] = True

    eval_model(opt, parser)
