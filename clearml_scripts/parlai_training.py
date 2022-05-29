from parlai.scripts.train_model import TrainModel
from parlai.scripts.display_model import DisplayModel

# Training a Model on a ParlAI Task (Dataset)
TrainModel.main(# we MUST provide a filename
    model_file='from_scratch_model/model',
    # train on empathetic dialogues
    task='clearmldata',
    # limit training time to 2 minutes, and a batchsize of 16
    max_train_time= 60,
    batchsize=16,
    # we specify the model type as seq2seq
    model='seq2seq',
    # some hyperparamter choices. We'll use attention. We could use pretrained
    # embeddings too, with embedding_type='fasttext', but they take a long
    # time to download.
    attention='dot',
    # tie the word embeddings of the encoder/decoder/softmax.
    lookuptable='all',
    # truncate text and labels at 64 tokens, for memory and time savings
    truncate=64,
    tensorboard_log=True,
    display_examples=True)

# Displaying Models Prediction on a particular task
DisplayModel.main(
    task='clearmldata',
    model_file='from_scratch_model/model',
    num_examples=4,
)