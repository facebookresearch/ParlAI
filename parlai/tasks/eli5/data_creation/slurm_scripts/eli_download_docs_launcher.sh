# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

for i in {0..99}; do sbatch --export=SCNUM=$i eli_download_docs.sbatch; done
