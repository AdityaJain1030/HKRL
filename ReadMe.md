# HKRL

RL agents to play Hollow Knight. Includes a Hollow Knight Mod to get the game state and a gym-like python environment to train on.

Currently Implemented:

- Mod to hook into Hollow Knight and send data to websocket server
- Websocket Gym Environment
- Vectorized environment
- Multi Instance Manager
- Frame stacking
- GPU utilization
- DQN
- PPO
  

TODO:
- Increase scenes trained on from from 4 to more than 4
- Reduce memory and gpu consumption of each individual scene (find unity docs for this)
- find best dqn hyperparameters
- Implement Algorithms
  - DDPG
  - SAC
- Move to shared memory for faster IPC
- Environment manager for extensible env development
- Add stablebaselines support
