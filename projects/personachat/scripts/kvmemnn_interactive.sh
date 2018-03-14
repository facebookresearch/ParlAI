# Interact with pre-trained model
# Profile memory model trained on personachat using persona 'self'
# Run from ParlAI directory

python -u -m pdb -c continue  examples/interactive.py -mf 'data/models/personachat/kvmemnn/persona-self_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-500_margin-0.1_tfidf-False_shareEmb-True_hops1_lins0_model' -m 'projects.personachat.kvmemnn.kvmemnn:Kvmemnn'
