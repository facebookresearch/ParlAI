# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train KVMemNN model.
"""

python -u examples/train_model.py  -m projects.personachat.kvmemnn.kvmemnn:KvmemnnAgent -t convai2:SelfRevisedTeacher -et convai2:SelfOriginalTeacher -ltim 60 -vtim 900 -vme 100000 --hops 1 --lins 0 -vp -1 -vmt accuracy -ttim 28800 -shareEmb True -bs 1 -lr 0.1 -esz 1000 --margin 0.1 --tfidf False --numthreads 40 -mf /tmp/persona-self_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-1000_margin-0.1_tfidf-False_shareEmb-True_hops1_lins0_model
