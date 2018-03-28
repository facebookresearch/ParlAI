from download_models import build
from parlai.core.params import ParlaiParser
from examples.interactive import interactive
from projects.personachat.persona_seq2seq import PersonachatSeqseqAgentBasic

'''Interact with pre-trained model
Generative model trained on personachat using persona 'self'
Run from ParlAI directory
'''

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(
        task='parlai.agents.local_human.local_human:LocalHumanAgent',
        model='projects.personachat.persona_seq2seq:PersonachatSeqseqAgentBasic',
        model_file='models:personachat/seq2seq_personachat/seq2seq_no_dropout0.2_lstm_1024_1e-3'
    )
    PersonachatSeqseqAgentBasic.add_cmdline_args(parser)

    opt = parser.parse_args()
    opt['model_type'] = 'seq2seq_personachat' # for builder
    # build all profile memory models
    fnames = ['seq2seq_no_dropout0.2_lstm_1024_1e-3',
              'fulldict.dict']
    build(opt, fnames)

    # add additional model args
    opt['dict_file'] = 'models:personachat/profile_memory/fulldict.dict'
    opt['interactive_mode'] = True

    interactive(opt)
