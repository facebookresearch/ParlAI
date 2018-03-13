# Interact with pre-trained model
# Profile memory model trained on personachat using persona 'self'
# Run from ParlAI directory

if [ ! -d data/models/personachat ]; then
  mkdir -p data/models/personachat
fi
if [ ! -f data/models/personachat/profilememory_learnreweight_sharelt_encdropout0.4_s2s_usepersona_self_useall_attn_general_lstm_1024_1_1e-3_0.1 ];  then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/profilememory_learnreweight_sharelt_encdropout0.4_s2s_usepersona_self_useall_attn_general_lstm_1024_1_1e-3_0.1 -P data/models/personachat
fi
if [ ! -f data/models/personachat/fulldict.dict ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/fulldict.dict -P data/models/personachat
fi
python examples/interactive.py -mf 'data/models/personachat/profilememory_learnreweight_sharelt_encdropout0.4_s2s_usepersona_self_useall_attn_general_lstm_1024_1_1e-3_0.1' -m 'projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit' --dict-file 'data/models/personachat/fulldict.dict' --interactive-mode True
