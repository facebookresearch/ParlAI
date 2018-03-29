from projects.convai2.baselines.download_models import download
from parlai.core.params import ParlaiParser
from examples.eval_model import setup_args, eval_model

'''Evaluate pre-trained model trained for hits@1 metric
Key-Value Memory Net model trained on personachat using persona 'self'
'''

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='convai2',
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='models:convai2/kvmemnn/model',
        datatype='valid',
        numthreads=8
    )
    opt = parser.parse_args()
    # build all profile memory models
    fnames = 'kvmemnn.tgz'
    opt['model_type'] = 'kvmemnn' # for builder
    download(opt, 'convai2', fnames)

    # add additional model args
    opt['interactive_mode'] = False

    eval_model(parser)
