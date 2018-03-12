# Evaluate pre-trained model trained for hits@1 metric
# Profile memory model trained on personachat using persona 'self'
# Run from ParlAI directory

if [ ! -f profilememory_learnreweight_sharelt_encdropout0.4_s2s_usepersona_self_useall_attn_general_lstm_1024_1_1e-3_0.1 ];  then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/profilememory_learnreweight_sharelt_encdropout0.4_s2s_usepersona_self_useall_attn_general_lstm_1024_1_1e-3_0.1
fi
if [ ! -f fulldict.dict ]; then
  wget https://s3.amazonaws.com/fair-data/parlai/_models/personachat/fulldict.dict
fi
python examples/eval_model.py -mf 'profilememory_learnreweight_sharelt_encdropout0.4_s2s_usepersona_self_useall_attn_general_lstm_1024_1_1e-3_0.1' -m 'projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit' -t personachat:self --dict-file 'fulldict.dict' -dt test --rank-candidates True
