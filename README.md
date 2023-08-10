# Code for "Uncertainty-Aware Reinforcement Learning: Variational Density Propagation through Deep Q-Networks"

This repository contains the code for the paper authored by Kyle Naddeo, which introduces a novel approach to uncertainty-aware reinforcement learning (RL) using a specialized Bayesian neural network (BNN). The code aims to demonstrate the key concepts and findings of the paper.

## Study Overview:

The study focuses on addressing the challenge of uncertainty in RL by proposing a deterministic BNN that effectively propagates the uncertainty of the state throughout the network. This approach provides both action values and the associated uncertainty for each action, offering innovative methods for exploration/exploitation, risk assessment, and regret evaluation.

## What the Code Intends to Show:

1. **Implementation of the Proposed Method**: The code provides a detailed implementation of the uncertainty-aware RL approach described in the paper, including the specialized BNN architecture.

2. **Demonstration of Robustness in Noisy State Spaces**: Through simulations and experiments, the code illustrates the inherent robustness of the proposed method in environments with noisy observations.

3. **Exploration/Exploitation Strategies**: The code includes specific strategies for uncertainty-based exploration and correlation-based action selection, showcasing how uncertainty guides the exploration process.

4. **Risk Assessment and Regret Calculation**: The code demonstrates how to incorporate risk penalties in the reward function and calculate regret using the Mahalanobis distance, as described in the paper.

5. **Reproducibility of Results**: The code is designed to allow researchers and practitioners to reproduce the results of the paper, providing insights into the effectiveness of the proposed method in various RL scenarios.

## Conclusion:

This code serves as a practical guide to understanding and implementing the uncertainty-aware RL method introduced in "Uncertainty-Aware Reinforcement Learning: Variational Density Propagation through Deep Q-Networks." By exploring the code, users can gain a deeper understanding of how to leverage uncertainty in RL for improved control in challenging environments with limited or unreliable observations.

---

For a comprehensive understanding of the theoretical foundations and methodologies, refer to the full paper.

# Reinforcement Learning Experiments Checklist

This repository serves as a checklist for evaluating a new reinforcement learning method that incorporates uncertainty estimation. Follow the steps below to ensure comprehensive evaluation.

## 1. Checklist for Environments and Networks

### Select Environments:

#### Classic Control:
- [ ] CartPole-v1
- [ ] MountainCar-v0
- [ ] Pendulum-v0
- [ ] Acrobot-v1

#### Box2D:
- [ ] LunarLander-v2
- [ ] BipedalWalker-v3

#### Atari:
- [ ] Breakout-v4
- [ ] Pong-v4
- [ ] SpaceInvaders-v4

#### MuJoCo:
- [ ] HalfCheetah-v2
- [ ] Hopper-v2
- [ ] Ant-v2
- [ ] Humanoid-v2

### Select Models for Comparison:

#### Value-Based Methods:
- [ ] DQN
- [ ] C51
- [ ] Rainbow

#### Policy Gradient Methods:
- [ ] REINFORCE
- [ ] PPO
- [ ] TRPO
- [ ] A3C
- [ ] A2C

#### Actor-Critic Methods:
- [ ] SAC
- [ ] TD3

#### Uncertainty Estimation Methods:

- [ ] Bayesian Neural Networks (BNNs) in RL
- [ ] Bootstrapped DQN
- [ ] Gaussian Processes (GPs) in RL

## 2. Checklist for Testing Uncertainty-Aware Reinforcement Learning
 
#### Implementation of the Proposed Method
- [ ] Verify the correct implementation of the Bayesian neural network (BNN) architecture.
- [ ] Test the deterministic propagation of uncertainty through the network.
- [ ] Validate the output of both action values and associated uncertainty for each action.

#### Demonstration of Robustness in Noisy State Spaces
- [ ] Run experiments in environments with different levels of observation noise.
- [ ] Compare the performance with traditional RL methods without uncertainty propagation.
- [ ] Analyze the stability and accuracy of value function estimates in noisy conditions.

#### Exploration/Exploitation Strategies
##### Uncertainty-Based Exploration
- [ ] Test the effectiveness of uncertainty-based exploration using the determinant of the covariance matrix.
- [ ] Compare exploration patterns with and without uncertainty guidance.

##### Correlation-Based Action Selection
- [ ] Validate the correlation-based action selection during the exploration phase.
- [ ] Analyze how correlation with the best action influences action choices.

#### Risk Assessment
- [ ] Implement and test the risk penalty in the reward function using the trace of the covariance matrix.
- [ ] Evaluate how risk assessment affects decision-making in various scenarios.

#### Regret Calculation
- [ ] Validate the calculation of regret using the Mahalanobis distance.
- [ ] Analyze how regret evaluation impacts the learning process and policy optimization.

#### Reproducibility of Results
- [ ] Ensure that the code is well-documented and can be run with different configurations.
- [ ] Verify that the results are consistent across multiple runs and different hardware setups.

#### Improved Uncertainty-Based Exploration (Additional Aspect)
- [ ] Test the new exploration strategies against other models.
- [ ] Evaluate how improved uncertainty-based exploration enhances learning efficiency and policy optimization.


