# Vectorised multi-agent Q-learning

Implements an agent-based simulation of traffic modality choice with each agent using independent Q-learning to
learn a policy that maximises their reward. The rewards are based on the congestion of each resource, and influence from
neighbours of the agent in a social network.

This is a reimplementation of a previous project, using vectorisation to achieve a speedup of around 40x.
Instead of storing the agents in a list, a single PyTorch tensor is used to store Q-values and update them
simultaneously using vector operations.

This representation is in some ways very natural. For example, social influence between agents can be seen 
as left-multiplication by the (weighted) adjacency matrix that represents the graph of social interaction strengths.

## Setup instructions

After cloning the repository and navigating to its directory, install the requirements:

```
pip install -r requirements.txt
```

## Running simulations

To run a basic simulation, use the following command line arguments:

```
python main.py --n_iterations=1 --output_dir=[insert desired output directory here]
```

This will run a simulation and save the result to the directory whose name you insert.

To plot the 

A full list of command line arguments is given below:

```
-h, --help            show this help message and exit
  --timesteps TIMESTEPS
                        Number of timesteps
  --n_actions N_ACTIONS
                        Number of actions agents can pick
  --n_agents N_AGENTS   Size of agent population
  --n_iterations N_ITERATIONS
                        Number of times to run simulation
  --update_rate UPDATE_RATE
                        The alpha parameter
  --discount_rate DISCOUNT_RATE
                        The gamma parameter
  --selection_mode SELECTION_MODE
                        Policy type
  --random_init RANDOM_INIT
                        Should Q-values be randomly initialised
  --intervention_start INTERVENTION_START
                        Timestep to start intervention at
  --intervention_end INTERVENTION_END
                        Time to revert intervention
  --intervention_type INTERVENTION_TYPE
                        Options are walk or car
  --social_graph SOCIAL_GRAPH
                        Generator. Options are none, ba, er, or ws
  --graph_connectivity GRAPH_CONNECTIVITY
                        Options are low, high, or ultra
  --output_dir OUTPUT_DIR
                        Directory to create output folder in
  --save SAVE           Whether to save the data
```