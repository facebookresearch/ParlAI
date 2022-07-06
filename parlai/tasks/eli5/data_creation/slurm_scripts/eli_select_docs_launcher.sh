# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

for nm in 'explainlikeimfive'
do
    export ns=15
    export nc=1
    for c in {0..99}
    do
        sbatch --export=ALL,NM=$nm,C=$c,NS=$ns,NC=$nc eli_select_docs.sbatch
    done
done
