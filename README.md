# Vectorised multi-agent Q-learning

This is a reimplementation of a previous project, using vectorisation to achieve a speedup of around 40x.
Instead of storing the agents in a list, a single PyTorch tensor is used to store Q-values and update them
all in one vector operation.