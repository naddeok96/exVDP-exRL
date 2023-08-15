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

---

### How to Run `train_distill_vdpdqn.py`

To run the `train_distill_vdpdqn.py` script, you can execute the following command in your terminal:

```bash
python train_distill_vdpdqn.py
```

Make sure you have all the required dependencies installed in your environment. You may need to modify the script or provide additional command-line arguments depending on your specific configuration and requirements.

### Understanding `VDPDQNAgent` and `DQNAgent`

In the provided code, two different agents are used for training: `VDPDQNAgent` and `DQNAgent`. Here's how they are different and how they are used together:

#### `VDPDQNAgent`

The `VDPDQNAgent` class is defined in the `vdp_dqn.py` file. It represents an agent that utilizes the Variational Deep Q-Network (VDQN) algorithm. This agent is designed to handle specific tasks that require a more complex understanding of the environment, possibly involving uncertainty or stochasticity.

#### `DQNAgent`

The `DQNAgent` class is defined in the `dqn.py` file. It represents a standard Deep Q-Network (DQN) agent, which is a widely used reinforcement learning algorithm. This agent is typically used for tasks with deterministic environments.

#### How They Work Together

In the `train_distill_vdpdqn.py` script, both agents are used together in a distillation process. The idea is to leverage the more complex VDPDQNAgent to guide the training of the simpler DQNAgent. By doing so, the DQNAgent can learn more efficiently and possibly achieve better performance in certain tasks.

### Insights on VDP Agents and Distillation Process

Currently, the results show that VDP (Variational Deep Q-Network) agents struggle to learn from the sparse reward signal alone. This challenge is addressed through a distillation process, where the knowledge from a standard DQN (Deep Q-Network) agent is transferred to the VDP agent.

#### Struggles with Sparse Reward Signal

The VDP agents, implemented through the `VDPDQNAgent` class, have shown difficulties in learning effectively from sparse reward signals. This can lead to suboptimal performance in environments where the feedback is limited or infrequent.

#### Learning Through Distillation

To overcome this challenge, a distillation process is employed. By training a standard DQN agent first and then using its learned knowledge to guide the training of the VDP agent, the latter can learn more effectively. This process allows the VDP agent to mimic the behavior of the DQN agent, thus benefiting from its ability to handle sparse reward signals.

#### Future Potential of VDP Model

Once the VDP model works as well as the standard DQN, insights from its uncertainty propagation can be harnessed. This could lead to more robust decision-making and a deeper understanding of the underlying environment. The VDP model's ability to handle uncertainty can provide valuable insights that go beyond what traditional DQN models can offer.

---

# Reinforcement Learning Experiments Checklist

This repository serves as a checklist for evaluating a new reinforcement learning method that incorporates uncertainty estimation. Follow the steps below to ensure comprehensive evaluation.

## 1. Checklist for Environments and Networks

## **Select Environments**:

<small>**(D)**: Discrete</small>  
<small>**(C)**: Continuous</small>

---

### **Classic Control**:
- [ ] **CartPole-v1** - (D)
- [ ] **MountainCar-v0** - (D)
- [ ] **Pendulum-v0** - (C)
- [ ] **Acrobot-v1** - (D)

---

### **Box2D**:
- [ ] **LunarLander-v2** - (D)
- [ ] **BipedalWalker-v3** - (C)

---

### **Atari**:
- [ ] **Breakout-v4** - (D)
- [ ] **Pong-v4** - (D)
- [ ] **SpaceInvaders-v4** - (D)

---

### **MuJoCo**:
- [ ] **HalfCheetah-v2** - (C)
- [ ] **Hopper-v2** - (C)
- [ ] **Ant-v2** - (C)
- [ ] **Humanoid-v2** - (C)

---


## **Select Models for Comparison**:

---

### **Value-Based Methods**:
- [ ] **DQN** - (D)
- [ ] **C51** - (D)
- [ ] **Rainbow** - (D)
- [ ] **Continuous Q-Learning (CQL)** - (C)

---

### **Policy Gradient Methods**:
- [ ] **REINFORCE** - (D/C)
- [ ] **PPO** - (D/C)
- [ ] **TRPO** - (D/C)

---

### **Actor-Critic Methods**:
- [ ] **A2C** - (D/C)
- [ ] **A3C** - (D/C)
- [ ] **SAC** - (C)
- [ ] **TD3** - (C)

---

### **Uncertainty Estimation Methods**:
- [ ] **Bayesian Neural Networks (BNNs)** in RL - (D/C)
- [ ] **Bootstrapped DQN** - (D)
- [ ] **Gaussian Processes (GPs)** in RL - (D/C)

---
### **Discrete (D) Table**:

|                 | DQN | C51 | Rainbow | REINFORCE | PPO | TRPO | A2C | A3C | Bootstrapped DQN |
|-----------------|-----|-----|---------|-----------|-----|------|-----|-----|------------------|
| CartPole-v1     |   ✔  |     |         |           |     |      |     |     |                  |
| MountainCar-v0  |   ✔  |     |         |           |     |      |     |     |                  |
| Acrobot-v1      |   ✔  |     |         |           |     |      |     |     |                  |
| LunarLander-v2  |   ✔  |     |         |           |     |      |     |     |                  |
| Breakout-v4     |     |     |         |           |     |      |     |     |                  |
| Pong-v4         |     |     |         |           |     |      |     |     |                  |
| SpaceInvaders-v4|     |     |         |           |     |      |     |     |                  |

---

### **Continuous (C) Table**:

|                 | CQL | REINFORCE | PPO | TRPO | A2C | A3C | SAC | TD3 | Gaussian Processes (GPs) |
|-----------------|-----|-----------|-----|------|-----|-----|-----|-----|---------------------------|
| Pendulum-v0     |     |           |     |      |     |     |     |     |                           |
| BipedalWalker-v3|     |           |     |      |     |     |     |     |                           |
| HalfCheetah-v2  |     |           |     |      |     |     |     |     |                           |
| Hopper-v2       |     |           |     |      |     |     |     |     |                           |
| Ant-v2          |     |           |     |      |     |     |     |     |                           |
| Humanoid-v2     |     |           |     |      |     |     |     |     |                           |

---


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


