#!/usr/bin/env bash
GPUS=4
NOISE=1.0
LBD=4.0
BLR=0.0001
BATCH=32
ACCUM_ITER=4
RESUME=false
LOAD_FROM=""


while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --ngpus)
    GPUS="$2"
    shift 
    shift 
    ;;
    --noise)
    NOISE="$2"
    shift 
    shift 
    ;;
    --num_noises)
    NUM_NOISES="$2"
    shift 
    shift 
    ;;
    --lbd)
    LBD="$2"
    shift 
    shift 
    ;;
    --eps)
    EPS="$2"
    shift
    shift 
    ;;
    --blr)
    BLR="$2"
    shift
    shift
    ;;
    --batch)
    BATCH="$2"
    shift 
    shift 
    ;;
    --accum_iter)
    ACCUM_ITER="$2"
    shift 
    shift 
    ;;
    --resume) 
    RESUME=true
    shift 
    ;;
    --load_from) 
    LOAD_FROM="$2"
    shift 
    shift
    ;;
    *)
    echo "Unknown option $key"
    exit 1
    ;;
  esac
done

if [ $GPUS -lt ${MAX_GPUS:-8} ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-${MAX_GPUS:-8}}
fi


echo "==> Starting FT-CADIS."

ID=$RANDOM
MASTER_PORT=$RANDOM

COMMAND="python3 -m torch.distributed.launch --nnodes ${TOTAL_NODES:-1} \
    --node_rank ${NODE_ID:-0} --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-$MASTER_PORT} \
    train.py \
    --id $ID --dataset cifar10 \
    --arch cifar10_vit_base \
    --ft_method full-ft \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --layer_decay 0.65 --drop_path 0.2 \
    --blr $BLR --batch $BATCH --accum_iter $ACCUM_ITER \
    --warmup_epochs 3 --epochs 30 --warmup_eps 10 --eps_double\
    --eps 64 --num-steps 4 \
    --train_noise_sd $NOISE --test_noise_sd $NOISE --num_noises 4 --lbd $LBD \
    --clip_grad 0.3 --use_fp16 true --warm_start"


if [ "$RESUME" = true ]; then
    COMMAND+=" --resume"
    COMMAND+=" --load_from $LOAD_FROM"
fi

COMMAND+=" ${@:1}"

$COMMAND
