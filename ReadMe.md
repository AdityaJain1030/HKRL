# HKRL

RL agents to play Hollow Knight. Includes a Hollow Knight Mod to get the game state and a gym-like python environment to train on.

Currently Implemented:

- Mod to hook into Hollow Knight and send data to websocket server
- Websocket Gym Environment
- (Kind of) Vectorized environment
- DQN

TODO:

- find best dqn hyperparameters
- Move inference and training to GPU
- Implement Algorithms
  - PPO
  - DDPG
  - SAC
- Add frame stacking
- Move to shared memory for faster IPC
- Environment manager for extensible env development
- Add stablebaselines support