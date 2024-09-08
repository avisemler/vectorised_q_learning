# Vectorised multi-agent Q-learning

This is a reimplementation of a previous project, using vectorisation to achieve a speedup of around 40x.
Instead of storing the agents in a list, a single PyTorch tensor is used to store Q-values and update them
at once using vector operations.

This representation is in some ways very natural. For example, social influence between agents can be seen 
as left-multiplication by the (weighted) adjacency matrix that represents the graph of social interaction strengths.