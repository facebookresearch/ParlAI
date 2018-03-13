# Evaluate pre-trained model trained for hits@1 metric
# Generative model trained on personachat using persona 'self'
# Run from ParlAI directory

if [ ! -d data/models/personachat ]; then
  mkdir -p data/models/personachat
fi
if [ ! -f data/models/personachat/seq2seq_no_dropout0.2_lstm_1024_1e-3 ];  then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/seq2seq_no_dropout0.2_lstm_1024_1e-3 -P data/models/personachat
fi
if [ ! -f data/models/personachat/fulldict.dict ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/fulldict.dict -P data/models/personachat
fi
python examples/eval_model.py -mf 'data/models/personachat/seq2seq_no_dropout0.2_lstm_1024_1e-3' -m 'projects.personachat.persona_seq2seq:PersonachatSeqseqAgentBasic' --dict-file 'data/models/personachat/fulldict.dict' -t personachat:self -dt test --rank-candidates True
