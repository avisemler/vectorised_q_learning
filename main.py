import time

import torch
import numpy as np
import networkx as nx

import utils
import plotting
import make_opts

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reward_function(opts, actions, value_functions, sensitivities, costs):
    #calculate congestedness of each action
    proportions = torch.sum(actions, dim=0) / opts.n_agents

    #apply value functions to proportions, action by action
    for action in range(opts.n_actions):
        proportions[action] = value_functions[action](proportions[action])
    valuations = proportions

    sensitivity_adjusted = sensitivities * valuations * actions
    sensitivity_adjusted = sensitivity_adjusted.sum(dim=-1)
    cost_adjustment = costs * actions
    cost_adjustment = cost_adjustment.sum(dim=-1)
    return sensitivity_adjusted - cost_adjustment

def run_simulation(opts):
    q_values = torch.zeros(opts.n_agents, opts.n_actions).to(device)
    sensitivities = torch.ones_like(q_values)
    sensitivities[:1000] = torch.tensor([1.3,1,1])
    sensitivities[1000:2000] = torch.tensor([1,1.35,1.4])
    sensitivities[2000:3000] = torch.tensor([1.4,0.7,0.8])
    costs = torch.zeros_like(q_values)
    costs[1000:2000] = torch.tensor([0.1,0.11,0])
    costs[2000:3000] = torch.tensor([-0.2,0,0])
    value_functions = [utils.RightTailGaussianFunc(0.0, 1, 0.4), utils.RightTailGaussianFunc(0.4, 1.14, 0.35), lambda x: 1]

    #generate social graph
    if opts.social_graph == "er" and opts.graph_connectivity == "low":
        graph = nx.erdos_renyi_graph(opts.n_agents, 0.00133377792)
        social_matrix = torch.eye(opts.n_agents, opts.n_agents).to(device)
        social_matrix += torch.from_numpy(nx.adjacency_matrix(graph, weight=0.1)).to(device)
    else:
        social_matrix = torch.eye(opts.n_agents, opts.n_agents).to(device)

    result = torch.zeros(opts.timesteps, opts.n_agents, opts.n_actions).to(device)
    for step in range(opts.timesteps):
        #make each agent pick an action
        if opts.selection_mode == "softmax":
            probabilities = torch.nn.functional.softmax(q_values, dim=-1)
            dist = torch.distributions.categorical.Categorical(probs=probabilities/(1/((i+1)**1.1)))
            actions = dist.sample()
        elif opts.selection_mode == "epsilon_greedy":
            #some agents should explore actions uniformly at random, some should exploit by picking maximum
            decisions = torch.rand(opts.n_agents).to(device)
            #get maximum Q-values, breaking ties randomly by first shuffling
            max_vals = q_values.max(dim=-1, keepdim=True)[0]
            max_mask = (q_values == max_vals)
            random_values = torch.rand_like(q_values) * max_mask
            randomly_tie_broken_argmax = random_values.argmax(dim=-1)
            #also make uniform random choices
            uniform_probs = torch.ones_like(q_values).to(device)/opts.n_actions
            randomly_chosen = torch.distributions.categorical.Categorical(probs=uniform_probs).sample()
            #combine, randomly using an epsilon
            actions = torch.where(decisions > 0.05, randomly_tie_broken_argmax, randomly_chosen)

        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=opts.n_actions)

        #calcuate reward of actions
        rewards = reward_function(opts, actions_one_hot, value_functions, sensitivities, costs)

        #update Q values
        q_values[actions_one_hot>0] = (1-opts.update_rate) * q_values[actions_one_hot>0] + opts.update_rate * (rewards + opts.discount_rate * torch.max(q_values, dim=-1)[0])
        
        #take into account social graph
        q_values = social_matrix @ q_values
        #store actions in result
        result[step] = actions_one_hot

        if step == opts.intervention_start:
            #perform an intervention
            costs[:,0] += 0.2 #tax cars
            sensitivities[2000:,2] += 0.4 #incentivise walking among those who walk least

    return result

if __name__ == "__main__":
    import sys, os

    start_time = time.time()

    opts = make_opts.make(*sys.argv[1:])
    results = []
    for i in range(opts.n_iterations):
        print(f"~~Iteration {i}~~")
        results.append(run_simulation(opts).cpu().numpy())

    plotting.plot(opts, results)

    #save results in folder
    output_dir = os.path.join("results", f"output_{opts.social_graph}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, arr in enumerate(results):
        np.save(os.path.join(output_dir, f"{i}.npy"), arr)
    print("took time:", time.time() - start_time)
