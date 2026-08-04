export DEXORA_LEROBOT_ROOT=/home/ubuntu/myh/expirement/Dexora/lerobot_data/two_arm_can_sort_random
export DEXORA_T5=google/t5-v1_1-small
export DEXORA_SIGLIP=google/siglip-so400m-patch14-384
export DEXORA_STATS=/home/ubuntu/myh/expirement/Dexora/lerobot_data/new_lerobot_stats/dataset_statistics.json


"""
huggingface-cli download google/siglip-so400m-patch14-384 \
    --local-dir google/siglip-so400m-patch14-384 --local-dir-use-symlinks False
huggingface-cli download google/t5-v1_1-small \
    --local-dir google/t5-v1_1-small              --local-dir-use-symlinks False


python -m data.lerobot_vla_dataset --stat \
    --num_samples 1000 \
    --repo_dir   /home/ubuntu/myh/expirement/Dexora/lerobot_data/two_arm_can_sort_random \
    --output_dir new_lerobot_stats
"""

echo '---1---'

NUM_GPUS=1 MAX_TRAIN_STEPS=100000 \
OUTPUT_DIR=checkpoints/dexora-400m-pretrain \
    bash s1_pretrain.sh

echo '---2a---'

SPRE_DIR=runs/spre bash s2a_analyze_jerk.sh
# → runs/spre/complete_analysis_results.json

echo '---2b---'

SPRE_DIR=runs/spre SHIGH_FILE=runs/shigh.json \
REPLAY_VERIFIER=trust_spre \
    bash s2b_replay.sh

echo '---2c---'

# (i) log-π proxy
STAGE1_CKPT=checkpoints/dexora-400m-pretrain \
LOGPI_FILE=runs/logpi/logpi.json \
    bash s2c_compute_logpi.sh

# (ii) discriminator
OUTPUT_DIR=checkpoints/dexora-scoring \
LOGPI_FILE=runs/logpi/logpi.json \
SPRE_FILE=runs/spre/complete_analysis_results.json \
SHIGH_FILE=runs/shigh.json \
    bash s2c_train_scoring.sh

echo '---3---'

STAGE1_CKPT=checkpoints/dexora-400m-pretrain \
SCORING_CKPT=checkpoints/dexora-scoring/final_model/pytorch_model.bin \
OUTPUT_DIR=checkpoints/dexora-400m-posttrain \
    bash s3_post_train.sh

echo '---Done'
