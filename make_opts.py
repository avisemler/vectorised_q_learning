import argparse

def make(*args):
    parser = argparse.ArgumentParser()

    #simulation
    parser.add_argument("--timesteps", default=6000, type=int, help="Number of timesteps")
    parser.add_argument("--n_actions", default=3, type=int, help="Number of actions agents can pick")
    parser.add_argument("--n_agents", default=3000, type=int, help="Size of agent population")
    parser.add_argument("--n_iterations", default=10, type=int, help="Number of times to run simulation")

    #Q-learning
    parser.add_argument("--update_rate", default=0.1, type=float, help="The alpha parameter")
    parser.add_argument("--discount_rate", default=0.75, type=float, help="The gamma parameter")
    parser.add_argument("--selection_mode", default="epsilon_greedy", type=str, help="Policy type")
    parser.add_argument("--random_init", default=0, type=int, help="Should Q-values be randomly initialised")

    #intervention
    parser.add_argument("--intervention_start", default=None, type=int, help="Timestep to start intervention at")
    parser.add_argument("--intervention_end", default=None, type=int, help="Time to revert intervention")
    parser.add_argument("--intervention_type", default="walk", type=str, help="Options are walk or car")
    parser.add_argument("--social_graph", default="none", type=str, help="Generator. Options are none, ba, er, or ws")
    parser.add_argument("--graph_connectivity", default="low", type=str, help="Options are low, high, or ultra")

    #files
    parser.add_argument("--output_dir", default="/dcs/large/u2107995/res", type=str, help="Directory to create output folder in")
    parser.add_argument("--save", default=1, type=int, help="Whether to save the data")

    return parser.parse_args(args)