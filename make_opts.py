import argparse
import ast

def make(*args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--timesteps", default=6000, type=int, help="")
    parser.add_argument("--n_actions", default=3, type=int, help="")
    parser.add_argument("--n_agents", default=3000, type=int, help="")

    parser.add_argument("--update_rate", default=0.001, type=float, help="")

    return parser.parse_args(args)