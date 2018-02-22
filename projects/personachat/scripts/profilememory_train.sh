# Trains a generative model on personachat data using the 'self' profile
# Run from ParlAI directory

python examples/train_model.py -m 'projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit' -t personachat:self -enc lstm -hs 1024 -nl 1 --dropout 0.1 --attention general --personachat_attnsentlevel --personachat_sharelt --personachat_reweight 'use' --personachat_learnreweight --dict-file /tmp/pc
