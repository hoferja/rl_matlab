# rl_matlab: The Reinforcement Learning MATLAB project

## Code changes with respect to generic template

**RL_environment.m**:
* Added a function (buildObservation) that creates the states **s**_k = [x_k, y_k, w_k, h_k]' as in the paper by the prof.
* initialized the states
* reward function based on paper -> please try out different ones, the current one does not seem to work that well (avg reward after ~1000 iterations ~ -1360)

  **RL_main.m**
  * renamed bool `train` to `doTrain` as otherwise the program crashes because of the MATLAB RL Toolbox function `train()`
  * corrected the `doTrain` bool logic for the if clause and added two lines to load the saved agent if `doTrain == false`
  * added some AI-coded analytics plotting functionality, havent't tested that one yet
