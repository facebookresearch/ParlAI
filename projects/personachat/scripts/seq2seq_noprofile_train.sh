# This trains a generative model on personachat without using the profile
# Run from ParlAI directory

python examples/train_model.py -m 'projects.personachat.persona_seq2seq:Seq2seqAgent' -t personachat:none -enc lstm -hs 1024 -nl 1 --dropout 0.1  --dict-file /tmp/pc
