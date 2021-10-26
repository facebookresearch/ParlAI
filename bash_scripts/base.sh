# setup NCCL to use EFA
export FI_PROVIDER=efa
export FI_OFI_RXR_RX_COPY_UNEXP=1
export FI_OFI_RXR_RX_COPY_OOO=1
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

export NCCL_NET_SHARED_BUFFERS=0

if [ -d "/usr/local/cuda-10.1/bin/" ]; then
        export PATH=$PATH:/usr/local/cuda/bin:/usr/local/lib/:/home/ubuntu/src/aws-ofi-nccl/out/lib/:/home/ubuntu/src/nccl/build/lib/:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/home/ubuntu/src/aws-ofi-nccl/out/lib/:/home/ubuntu/src/nccl/build/lib/:
    fi
