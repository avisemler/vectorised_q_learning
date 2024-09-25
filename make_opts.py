import argparse

def make(*args):
    parser = argparse.ArgumentParser()

    #simulation
    parser.add_argument("--timesteps", default=6000, type=int, help="")
    parser.add_argument("--n_actions", default=3, type=int, help="")
    parser.add_argument("--n_agents", default=3000, type=int, help="")
    parser.add_argument("--n_iterations", default=10, type=int, help="")

    #Q-learning
    parser.add_argument("--update_rate", default=0.1, type=float, help="")
    parser.add_argument("--discount_rate", default=0.75, type=float, help="")
    parser.add_argument("--selection_mode", default="epsilon_greedy", type=str, help="")

    #intervention
    parser.add_argument("--intervention_start", default=None, type=int, help="")
    parser.add_argument("--intervention_end", default=None, type=int, help="")
    parser.add_argument("--intervention_type", default="walk", type=str, help="")
    parser.add_argument("--social_graph", default="none", type=str, help="")
    parser.add_argument("--graph_connectivity", default="low", type=str, help="")

    #files
    parser.add_argument("--output_dir", default="/dcs/large/u2107995/res", type=str, help="")
    parser.add_argument("--save", default=0, type=int, help="")

    return parser.parse_args(args)