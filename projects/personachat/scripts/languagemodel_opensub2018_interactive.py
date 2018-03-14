from download_models import build
from parlai.core.params import ParlaiParser
from examples.interactive import interactive
from parlai.agents.language_model.language_model import LanguageModelAgent

'''Interact with pre-trained model
Language model trained on Opensubtitles 2018 dataset
Run from ParlAI directory
'''

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(
        task='parlai.agents.local_human.local_human:LocalHumanAgent',
        model='language_model',
        model_file='data/models/personachat/language_model/languagemodel_esz512_hid1024_nl2.pt'
    )
    LanguageModelAgent.add_cmdline_args(parser)

    opt = parser.parse_args()
    opt['model_type'] = 'language_model' # for builder
    # build all profile memory models
    fnames = ['languagemodel_esz512_hid1024_nl2.pt',
              'opensubtitles2018.dict']
    build(opt, fnames)

    # add additional model args
    opt['dict_file'] = 'data/models/personachat/language_model/opensubtitles2018.dict'
    opt['sampling_mode'] = True

    interactive(opt)
