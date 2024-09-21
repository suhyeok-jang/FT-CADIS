#!/usr/bin/env bash
GPUS=4
NOISE=1.0
LBD=2.0
BLR=0.0004
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
    --weight_decay)
    WEIGHT_DECAY="$2"
    shift 
    shift 
    ;;
    --weight_decay_end)
    WEIGHT_DECAY_END="$2"
    shift 
    shift 
    ;;
    --layer_decay)
    LAYER_DECAY="$2"
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
    --eps_double) 
    EPS_DOUBLE=true
    shift 
    ;;
    --warmup_eps) 
    WARMUP_EPS="$2"
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
    --id $ID --dataset imagenet \
    --arch imagenet_vit_base \
    --ft_method lora \
    --weight_decay $WEIGHT_DECAY --weight_decay_end $WEIGHT_DECAY_END \
    --layer_decay $LAYER_DECAY --drop_path 0.0 \
    --blr $BLR --batch $BATCH --accum_iter $ACCUM_ITER \
    --warmup_epochs 1 --epochs 10 \
    --eps 64 --num-steps 1 \
    --train_noise_sd $NOISE --test_noise_sd $NOISE --num_noises 2 --lbd $LBD \
    --clip_grad 1.0 --use_fp16 true --warm_start"

if [ "$EPS_DOUBLE" = true ]; then
    COMMAND+=" --eps_double"
    COMMAND+=" --warmup_eps $WARMUP_EPS"
fi

if [ "$RESUME" = true ]; then
    COMMAND+=" --resume"
    COMMAND+=" --load_from $LOAD_FROM"
fi

COMMAND+=" ${@:1}"

$COMMAND
