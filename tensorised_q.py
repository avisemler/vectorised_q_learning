import time

import torch

from utils import *
import make_opts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reward_function(opts, actions, value_functions):
    #calculate congestedness of each action
    proportions = torch.sum(actions, dim=0) / opts.n_agents

    #apply value functions to proportions, action by action
    for action in range(n_actions):
        proportions[action] = value_functions[action](proportions[action])
    valuations = proportions

    sensitivity_adjusted = sensitivities * valuations * actions
    sensitivity_adjusted = sensitivity_adjusted.sum(dim=-1)
    return sensitivity_adjusted 

def run_simulation(opts):
    start_time = time.time()

    q_values = torch.zeros(opts.n_agents, opts.n_actions).to(device)
    sensitivities = torch.ones_like(q_values)
    sensitivities[:1000] = torch.tensor([2,1,1])
    value_functions = [lambda x: x for i in range(opts.n_actions)]

    result = torch.zeros(opts.timesteps, opts.n_actions)
    for step in range(opts.timesteps):
        #make each agent pick an action
        probabilities = torch.nn.functional.softmax(q_values, dim=-1)
        dist = torch.distributions.categorical.Categorical(probs=probabilities)
        actions = dist.sample()
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=opts.n_actions)

        #calcuate reward of actions
        rewards = reward_function(opts, actions_one_hot, value_functions)

        #update Q values
        q_values[actions_one_hot>0] = (1-opts.update_rate) * q_values[actions_one_hot>0] + opts.update_rate * (rewards + torch.max(q_values, dim=-1)[0])
    print("took time:", time.time() - start_time)

if __name__ == "__main__":
    import sys
    opts = make_opts.make(*sys.argv[1:])
    run_simulation(opts)