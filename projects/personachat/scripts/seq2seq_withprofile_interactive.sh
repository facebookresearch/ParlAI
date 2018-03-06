# Interact with pre-trained model
# Generative model trained on personachat using persona 'self'
# Run from ParlAI directory

if [ ! -f seq2seq_no_dropout0.2_lstm_1024_1e-3 ];  then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/seq2seq_no_dropout0.2_lstm_1024_1e-3
fi
if [ ! -f fulldict.dict ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/fulldict.dict
fi
python examples/interactive.py -mf 'seq2seq_no_dropout0.2_lstm_1024_1e-3' -m 'projects.personachat.persona_seq2seq:PersonachatSeqseqAgentBasic' --dict-file 'fulldict.dict' --interactive-mode True
