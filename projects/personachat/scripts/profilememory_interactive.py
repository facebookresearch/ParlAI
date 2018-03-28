from download_models import build
from parlai.core.params import ParlaiParser
from examples.interactive import interactive
from projects.personachat.persona_seq2seq import PersonachatSeqseqAgentSplit

'''Interact with pre-trained model
Profile memory model trained on personachat using persona 'self'
Run from ParlAI directory
'''

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(
        task='parlai.agents.local_human.local_human:LocalHumanAgent',
        model='projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit',
        model_file='models:personachat/profile_memory/profilememory_learnreweight_sharelt_encdropout0.4_s2s_usepersona_self_useall_attn_general_lstm_1024_1_1e-3_0.1',
    )
    PersonachatSeqseqAgentSplit.add_cmdline_args(parser)

    opt = parser.parse_args()
    opt['model_type'] = 'profile_memory' # for builder
    # build all profile memory models
    fnames = ['profilememory_mem2_reweight_sharelt_encdropout0.2_selfpersona_useall_attn_general_lstm_1024_1_1e-3_0.1',
              'profilememory_learnreweight_sharelt_encdropout0.4_s2s_usepersona_self_useall_attn_general_lstm_1024_1_1e-3_0.1',
              'fulldict.dict']
    build(opt, fnames)

    # add additional model args
    opt['dict_file'] = 'models:personachat/profile_memory/fulldict.dict'
    opt['interactive_mode'] = True

    interactive(opt)
