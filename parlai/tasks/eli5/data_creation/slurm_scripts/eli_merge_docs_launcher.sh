# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

for nm in 'explainlikeimfive'
do
    for c in {0..9}
    do
        sbatch --export=ALL,NM=$nm,C=$c eli_merge_docs.sbatch
    done
done
