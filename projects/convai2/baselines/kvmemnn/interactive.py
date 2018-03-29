from projects.convai2.baselines.download_models import download
from parlai.core.params import ParlaiParser
from examples.interactive import interactive

'''Interact with pre-trained model
Key-Value Memory Net model trained on personachat using persona 'self'
[Note: no persona in this example code is actually given to the model.]
'''

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(
        task='parlai.agents.local_human.local_human:LocalHumanAgent',
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='models:convai2/kvmemnn/model'
    )
    opt = parser.parse_args()
    # build all profile memory models
    fnames = 'kvmemnn.tgz'
    opt['model_type'] = 'kvmemnn' # for builder
    download(opt, 'convai2', fnames)

    # add additional model args
    opt['override'] = ['interactive_mode']
    opt['interactive_mode'] = True

    interactive(opt)
