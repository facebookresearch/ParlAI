from download_models import build
from parlai.core.params import ParlaiParser
from examples.interactive import interactive

'''Interact with pre-trained model
Key-Value Memory Net model trained on personachat using persona 'self'
Run from ParlAI directory
'''

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(
        task='parlai.agents.local_human.local_human:LocalHumanAgent',
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='data/models/personachat/kvmemnn/kvmemnn/persona-self_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-500_margin-0.1_tfidf-False_shareEmb-True_hops1_lins0_model',
    )
    opt = parser.parse_args()
    opt['model_type'] = 'kvmemnn' # for builder
    # build all profile memory models
    fnames = ['kvmemnn.tgz']
    build(opt, fnames)

    # add additional model args
    opt['interactive_mode'] = True

    interactive(opt)
