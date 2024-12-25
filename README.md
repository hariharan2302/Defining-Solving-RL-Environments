# Defining-Solving-RL-Environments
A comprehensive implementation of Q-Learning and SARSA algorithms in both deterministic and stochastic environments, along with a stock trading application using reinforcement learning techniques.
## Key Features

### GridWorld Environments
- Custom GridWorld implementation with:
  - Deterministic environment with predictable outcomes
  - Stochastic environment with probabilistic state transitions
  - Positive (+1), negative (-1), and goal (+10) rewards
  - 36 states and 4 actions (UP, DOWN, LEFT, RIGHT)

### Learning Algorithms
- Q-Learning implementation with hyperparameter tuning
- SARSA (State-Action-Reward-State-Action) implementation
- Comparative analysis between both algorithms

### Stock Trading Environment
- RL-based stock trading simulation
- Historical data analysis and decision-making
- Performance visualization and reward tracking

## Hyperparameter Settings

- Episodes: Tested with 50, 200, and 350 episodes
- Gamma (discount factor): Tested with 0.3, 0.5, and 0.7
- Learning rate (α): 0.1
- Epsilon (exploration rate): Decaying from 1.0 to 0.01

## Results

### Key Findings
- Q-Learning and SARSA show comparable performance in deterministic environments
- SARSA demonstrates more stable learning in stochastic environments
- Higher gamma values (0.7) lead to better long-term reward optimization
- Increasing episodes (up to 350) improves policy convergence

### Stock Trading Performance
- Successful adaptation to market conditions
- Consistent reward improvement during training
- Stable performance in testing phase

## Safety Measures

- Bounded action space enforcement
- State-space validation
- Reward clipping for stability
- Environment boundary checks

## Author
Hariharan Venkatraman

## Course Information
CSE 446/546: Reinforcement Learning (Spring 2024)
University at Buffalo

## License
This project is for educational purposes as part of CSE 446/546 coursework.

