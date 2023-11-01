#!/bin/bash
source config.sh

gpus=1
mem=8G
cpus=2

PRETRAINED_PATH=$1
BASENAME=${2:-"bert-base-uncased"}
MAX_LEN=${3:-512}

SAVE_PATH=${PRETRAINED_PATH}
mkdir -p ${SAVE_PATH}/logs


for SEED in 1 2 3 4 5; do
    TASK_NAME="MNLI"
    LR=1e-5
    EPOCHS=10
    BATCH_SIZE=32

    mnli_jid=$(sbatch \
            --parsable \
            --gres=gpu:${gpus} \
            --mem $mem \
            -c $cpus \
            --output ${SAVE_PATH}/logs/${TASK_NAME}_${EPOCHS}_${LR}_${BATCH_SIZE}_${SEED} \
            ${cluster_info} \
            glue_eval/run_glue_for_seed_task.sh ${PRETRAINED_PATH} ${TASK_NAME} ${SEED} ${EPOCHS} ${LR} ${BATCH_SIZE} ${CACHE} ${BASENAME})
    echo -n "${mnli_jid} "

    # Following RoBERTa paper, initialize RTE/MRPC/STS from MNLI
    for TASK_NAME in "RTE" "MRPC" "STSB"; do
        if [[ "${TASK_NAME}" = "RTE" ]]; then
            LR=2e-5
            EPOCHS=10
            BATCH_SIZE=16
        elif [[ "${TASK_NAME}" = "MRPC" ]]; then
            LR=1e-5
            EPOCHS=10
            BATCH_SIZE=16
        elif [[ "${TASK_NAME}" = "STSB" ]]; then
            LR=2e-5
            EPOCHS=10
            BATCH_SIZE=16
        fi
        jid=$(sbatch \
                --parsable \
                --dependency ${mnli_jid} \
                --gres=gpu:${gpus} \
                --mem $mem \
                -c $cpus \
                --output ${SAVE_PATH}/logs/${TASK_NAME}_${EPOCHS}_${LR}_${BATCH_SIZE}_${SEED} \
                ${cluster_info} \
                glue_eval/run_glue_for_seed_task.sh ${SAVE_PATH}/finetune-runs/MNLI_EPOCHS10_BS32_LR1e-5_seed${SEED} ${TASK_NAME} ${SEED} ${EPOCHS} ${LR} ${BATCH_SIZE} ${CACHE}  ${BASENAME} ${SAVE_PATH} )
        echo -n "${jid} "
    done
done

for TASK_NAME in "COLA"; do
LR=1e-5
EPOCHS=10
BATCH_SIZE=16
for SEED in 1 2 3 4 5; do
jid=$(sbatch \
        --parsable \
        --gres=gpu:${gpus} \
        --mem $mem \
        -c $cpus \
        --output ${SAVE_PATH}/logs/${TASK_NAME}_${EPOCHS}_${LR}_${BATCH_SIZE}_${SEED} \
        ${cluster_info} \
        glue_eval/run_glue_for_seed_task.sh ${PRETRAINED_PATH} ${TASK_NAME} ${SEED} ${EPOCHS} ${LR} ${BATCH_SIZE} ${CACHE} ${BASENAME})
echo -n "${jid} "
    done
done

for TASK_NAME in "QQP" "SST2" "QNLI"; do
if [[ "${TASK_NAME}" = "QQP" ]]; then
    LR=1e-5
    EPOCHS=10
    BATCH_SIZE=32
elif [[ "${TASK_NAME}" = "SST2" ]]; then
    LR=1e-5
    EPOCHS=10
    BATCH_SIZE=32
elif [[ "${TASK_NAME}" = "QNLI" ]]; then
    LR=1e-5
    EPOCHS=10
    BATCH_SIZE=32
fi

for SEED in 1 2 3 4 5; do
    jid=$(sbatch \
        --parsable \
        --gres=gpu:${gpus} \
        --mem $mem \
        -c $cpus \
        --output ${SAVE_PATH}/logs/${TASK_NAME}_${EPOCHS}_${LR}_${BATCH_SIZE}_${SEED} \
        ${cluster_info} \
        glue_eval/run_glue_for_seed_task.sh ${PRETRAINED_PATH} ${TASK_NAME} ${SEED} ${EPOCHS} ${LR} ${BATCH_SIZE} ${CACHE} ${BASENAME})
    echo -n "${jid} "
    done
done

