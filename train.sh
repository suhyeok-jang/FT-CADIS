#!/usr/bin/env bash
DATASET=""
ARCH=""
FT_METHOD=""
GPUS=4
NOISE=1.0
NUM_NOISES=4
LBD=2.0
EPS=64
BLR=0.001
BATCH=32
ACCUM_ITER=1
EPS_DOUBLE=false
RESUME=false
LOAD_FROM=""


while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --dataset)
    DATASET="$2"
    shift 
    shift 
    ;;
    --arch)
    ARCH="$2"
    shift 
    shift 
    ;;
    --ft_method)
    FT_METHOD="$2"
    shift 
    shift 
    ;;
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
    shift # past argument
    shift # past value
    ;;
    --blr)
    BLR="$2"
    shift # past argument
    shift # past value
    ;;
    --batch)
    BATCH="$2"
    shift # past argument
    shift # past value
    ;;
    --accum_iter)
    ACCUM_ITER="$2"
    shift # past argument
    shift # past value
    ;;
    --eps_double) 
    EPS_DOUBLE=true
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
    --id $ID --dataset $DATASET \
    --arch $ARCH \
    --ft_method $FT_METHOD \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --layer_decay 0.65 --drop_path 0.0 \
    --blr $BLR --batch $BATCH --accum_iter $ACCUM_ITER \
    --warmup_epochs 2 --epochs 20 --warmup_eps 10 \
    --eps $EPS --num-steps 1 \
    --train_noise_sd $NOISE --valid_noise_sd $NOISE --num_noises $NUM_NOISES --lbd $LBD \
    --clip_grad 1.0 --use_fp16 true --warm_start"

if [ "$EPS_DOUBLE" = true ]; then
    COMMAND+=" --eps_double"
fi

if [ "$RESUME" = true ]; then
    COMMAND+=" --resume"
    COMMAND+=" --load_from $LOAD_FROM"
fi

COMMAND+=" ${@:1}"

$COMMAND
