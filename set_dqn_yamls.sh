#!/bin/bash

# Define a list of environments
declare -a envs=("CartPole-v1" "MountainCar-v0" "Acrobot-v1" "LunarLander-v2" "Breakout-v4" "Pong-v4" "SpaceInvaders-v4" )

# Directory to save the YAML files
mkdir -p yamls/dqn

# Default hyperparameters for ConvDQN
conv1_out_channels=32
conv1_kernel_size=8
conv1_stride=4
conv2_out_channels=64
conv2_kernel_size=4
conv2_stride=2
conv3_out_channels=64
conv3_kernel_size=3
conv3_stride=1
use_conv=false

# Loop through the environments and create a YAML file for each
for env in "${envs[@]}"; do
  # Default hyperparameters
  fc1_size=128
  fc2_size=128
  gamma=0.99
  epsilon=1.0
  epsilon_min=0.01
  epsilon_decay=0.9995
  learning_rate=0.0001
  memory_size=10000
  batch_size=32
  episodes=10000
  max_steps=300
  reward_plateau_threshold=0.0001
  loss_plateau_threshold=0.00001
  plateau_window=50
  use_conv=false

  # Custom hyperparameters for specific environments
  case $env in
    "CartPole-v1")
      epsilon_decay=0.995
      ;;
    "MountainCar-v0")
      epsilon_decay=0.9999999
      fc1_size=24
      fc2_size=24
      ;;
    "LunarLander-v2")
      fc1_size=256
      fc2_size=256
      ;;
    "BipedalWalker-v3")
      learning_rate=0.0003
      ;;
    "Breakout-v4"|"Pong-v4"|"SpaceInvaders-v4")
      fc1_size=1024
      fc2_size=256
      epsilon_decay=0.999
      episodes=20000
      use_conv=true
  esac

  if [ "$use_conv" = true ]; then
    cat > "yamls/dqn/$env.yaml" <<EOL
fc1_size: $fc1_size
fc2_size: $fc2_size
gamma: $gamma
epsilon: $epsilon
epsilon_min: $epsilon_min
epsilon_decay: $epsilon_decay
learning_rate: $learning_rate
memory_size: $memory_size
batch_size: $batch_size
episodes: $episodes
max_steps: $max_steps
environment: $env
reward_plateau_threshold: $reward_plateau_threshold
loss_plateau_threshold: $loss_plateau_threshold
plateau_window: $plateau_window
use_conv: $use_conv
conv1_out_channels: $conv1_out_channels
conv1_kernel_size: $conv1_kernel_size
conv1_stride: $conv1_stride
conv2_out_channels: $conv2_out_channels
conv2_kernel_size: $conv2_kernel_size
conv2_stride: $conv2_stride
conv3_out_channels: $conv3_out_channels
conv3_kernel_size: $conv3_kernel_size
conv3_stride: $conv3_stride
EOL
  else
    cat > "yamls/dqn/$env.yaml" <<EOL
fc1_size: $fc1_size
fc2_size: $fc2_size
gamma: $gamma
epsilon: $epsilon
epsilon_min: $epsilon_min
epsilon_decay: $epsilon_decay
learning_rate: $learning_rate
memory_size: $memory_size
batch_size: $batch_size
episodes: $episodes
max_steps: $max_steps
environment: $env
reward_plateau_threshold: $reward_plateau_threshold
loss_plateau_threshold: $loss_plateau_threshold
plateau_window: $plateau_window
use_conv: $use_conv
EOL
  fi
done

echo "YAML files created in yamls/dqn/"
