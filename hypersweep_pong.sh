#!/bin/bash

# Activate the virtual environment
source ../envs/vdprl/bin/activate

OUTPUT=$(wandb sweep yamls/dqn/Pong-v5-async-sweep.yaml -p "DQN Envpool Async Sweep" -e "naddeok" --verbose 2>&1)
SWEEP_ID=$(echo "$OUTPUT" | awk -F"'" '/Run sweep agent with:/ { print $2 }')

# Now you can use $SWEEP_ID for whatever you need
echo "Full Output: $OUTPUT "
echo "Sweep ID: $SWEEP_ID"

# List of GPUs you want to use
GPU_LIST=("0" "1" "2" "3")
N_PER_GPU=30  # Number of parallel runs on each GPU

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Loop over each GPU and start N_PER_GPU runs
# Loop over each GPU and start N_PER_GPU runs
for gpu in "${GPU_LIST[@]}"; do
  for i in $(seq 1 $N_PER_GPU); do
    unique_id="gpu${gpu}_run${i}"
    output_file="outputs/${unique_id}.out"
    nohup env CUDA_VISIBLE_DEVICES=$gpu wandb agent "$SWEEP_ID" --count 1 >> "$output_file" 2>&1 &
    sleep 2  # Small delay to avoid collisions
  done
done

