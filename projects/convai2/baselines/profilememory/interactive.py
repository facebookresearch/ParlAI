from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from examples.interactive import interactive
from projects.personachat.persona_seq2seq import PersonachatSeqseqAgentSplit

"""Interact with pre-trained model
Profile Memory model trained on ConvAI2 using persona 'self'
"""

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(
        task='parlai.agents.local_human.local_human:LocalHumanAgent',
        model='projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit',
        model_file='models:convai2/profilememory/profilememory_convai2_model',
    )
    PersonachatSeqseqAgentSplit.add_cmdline_args(parser)

    opt = parser.parse_args()
    opt['model_type'] = 'profilememory' # for builder
    # build profile memory models
    fnames = ['profilememory_convai2_model',
              'profilememory_convai2.dict']
    download_models(opt, fnames, 'convai2', use_model_type=True)

    # add additional model args
    opt['dict_file'] = 'models:convai2/profilememory/profilememory_convai2.dict'
    opt['interactive_mode'] = True

    interactive(opt)
