# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This trains a (KV) Memory Net model on personachat using the (original) self profile
# Run from ParlAI directory

python examples/train_model.py -m projects.personachat.kvmemnn.kvmemnn:KvmemnnAgent -t personachat -ltim 60 -vme 10000 -vtim 900 --hops 1 --lins 0 -vp -1 -vmt accuracy -ttim 28800 -shareEmb True -bs 1 -lr 0.1 -esz 500 --margin 0.1 --tfidf False --numthreads 40 -mf "/tmp/persona-self_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-500_margin-0.1_tfidf-False_shareEmb-True_hops1_lins0_model"
