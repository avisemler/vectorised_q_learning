import math
import os
import json
from types import SimpleNamespace
from statistics import stdev

import matplotlib.pyplot as plt 
import numpy as np
import einops

from utils import gaussian_density

LABEL_SIZE = 9
LINE_WIDTH = 1.0
AGENT_TYPES = 3
action_names = ["car", "bus", "walk"]
COLOURS = ["blue", "red", "green", "green", "purple"]

def calculate_authority_utility(opts, result, timestep):
    #to ensure the result is not affected by noise, first calculate
    #the mean of some timesteps around timestep
    action_mean = np.mean(result[timestep-10:timestep,:], axis=0)

    car_congestion = action_mean[0] / opts.n_agents
    bus_congestion = action_mean[1] / opts.n_agents
    car_value = gaussian_density(car_congestion, 0, 0.45)
    bus_value = gaussian_density(bus_congestion, 0.3, 0.32)
    total_utility = 1.2 + car_value + bus_value

    return total_utility

def mean_and_std_of_groups(l):
    """
    Computes mean and standard deviation for a list l of numpy
    arrays that is potentially too large to store in memory as a single stacked 
    array.
    """
    #first, sum up each agent group
    l = [einops.reduce(arr, "t (g 1000) a -> t g a", "sum") for arr in l]
    sum = np.zeros_like(l[0])
    for arr in l:
        sum += arr
    mean = sum / len(l)

    squared_deviations = []
    for arr in l:
        squared_deviations.append(np.square(mean - arr))
    sum_squared_deviations = np.zeros_like(l[0])
    for arr in squared_deviations:
        sum_squared_deviations += arr

    return mean, np.sqrt(sum_squared_deviations / len(l))

def plot(opts, results=None, show=False, title="", name_suffix=""):
    #compute mean and standard deviation over each run
    mean, std = mean_and_std_of_groups(results)

    #plot the action counts for each agent type and summed over the types
    dimension = math.ceil(math.sqrt(AGENT_TYPES + 1))
    fig, axis = plt.subplots(dimension, dimension)
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(right=0.83, left=0.12, wspace=0.45, hspace=0.45)
    plot_count = 0
    for agent_type in range(AGENT_TYPES):
        for j in range(opts.n_actions):
            #plot a line for the ith action
            x_coord = plot_count // dimension
            y_coord = plot_count % dimension
            plot_line = mean[:,agent_type,j]
            band_width = std[:,agent_type,j]
            axis[x_coord, y_coord].plot(np.arange(opts.timesteps),
                plot_line,
                label=action_names[j],
                color=COLOURS[j],
                lw=str(LINE_WIDTH)
            )
            axis[x_coord, y_coord].fill_between(np.arange(opts.timesteps),
                plot_line - band_width,
                plot_line + band_width,
                facecolor=COLOURS[j],
                alpha=0.4,
            )
            axis[x_coord, y_coord].grid(axis='y')
            axis[x_coord, y_coord].set_title("Group " + str(agent_type + 1) + " " + title)
            axis[x_coord, y_coord].set_ylim(0, 1000)
            axis[x_coord, y_coord].set_xlabel("Time")
            axis[x_coord, y_coord].set_ylabel("Number of agents")

            xlab = axis[x_coord, y_coord].xaxis.get_label()
            ylab = axis[x_coord, y_coord].yaxis.get_label()
            ttl = axis[x_coord, y_coord].title
            ttl.set_weight('demibold')
            ttl.set_size(13)

            #xlab.set_style('italic')
            xlab.set_size(LABEL_SIZE)
            #ylab.set_style('italic')
            ylab.set_size(LABEL_SIZE)

        plot_count += 1

    #plot the sum over all agent types too
    all_types_std = np.std(np.stack([r.sum(axis=1) for r in results]), axis=0)
    for j in range(opts.n_actions):
        x_coord = plot_count // dimension
        y_coord = plot_count % dimension
        plot_line = mean[:,:,j].sum(axis=-1)
        axis[x_coord, y_coord].plot(np.arange(opts.timesteps),
            plot_line,
            label=action_names[j],
            color=COLOURS[j],
            lw=str(LINE_WIDTH)
        )
        axis[x_coord, y_coord].fill_between(np.arange(opts.timesteps),
            plot_line - all_types_std[:,j],
            plot_line + all_types_std[:,j],
            facecolor=COLOURS[j],
            alpha=0.4,
        )
        axis[x_coord, y_coord].grid(axis='y')
        axis[x_coord, y_coord].set_title("Full population " + title)
        axis[x_coord, y_coord].set_ylim(0, 3000)
        axis[x_coord, y_coord].set_xlabel("Time")
        axis[x_coord, y_coord].set_ylabel("Number of agents")

        xlab = axis[x_coord, y_coord].xaxis.get_label()
        ylab = axis[x_coord, y_coord].yaxis.get_label()
        ttl = axis[x_coord, y_coord].title
        ttl.set_weight('demibold')
        ttl.set_size(13)

        #xlab.set_style('italic')
        xlab.set_size(LABEL_SIZE)
        #ylab.set_style('italic')
        ylab.set_size(LABEL_SIZE)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper right")
    image_name = opts.run_name + name_suffix + ".png"
    plt.savefig(os.path.join("images", image_name), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import glob

    authority_utilities = {}
    run_names = []

    directories_to_plot = [d for d in glob.glob("/dcs/large/u2107995/res/*/") if d.endswith("_")]
    for d in directories_to_plot:
        #load options
        with open(os.path.join(d, "opts.json"), "r") as f:
            opts = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
            print(opts.run_name)
            run_names.append(opts.run_name)
            #load results
            results = []
            for i, np_file in enumerate(glob.glob(os.path.join(d, "*.npy"))):
                arr = np.load(np_file, allow_pickle=True).astype(np.float32)
                results.append(arr)
            
            plot(opts, results)

            #also perform an authority utility calculation, and compute its mean
            run_utilities = []
            for res in results:
                run_utilities.append(calculate_authority_utility(opts, res.sum(axis=-2), opts.timesteps-1))
            authority_utilities[opts.run_name] = sum(run_utilities) / len(run_utilities)

            if "late" in opts.run_name:
                #if there was an intervention, calculate its mean effect size
                effect_sizes = []
                utility_changes = []
                intervention_index = action_names.index(opts.intervention_type)
                for res in results:
                    baseline = calculate_authority_utility(opts, res.sum(axis=-2), opts.intervention_start-2)
                    size = authority_utilities[opts.run_name] - baseline
                    utility_changes.append(size)

                    pre_intervention_level = np.mean(res.sum(axis=-2)[opts.intervention_start-10:opts.intervention_start, intervention_index])
                    post_intervention_level = np.mean(res.sum(axis=-2)[opts.timesteps-10:opts.timesteps, intervention_index])
                    print(post_intervention_level - pre_intervention_level)
                    effect_sizes.append(post_intervention_level - pre_intervention_level)
                authority_utilities[opts.run_name + "_effect_size"] = sum(effect_sizes) / len(effect_sizes)
                authority_utilities[opts.run_name + "_effect_size_std"] = 0 if len(effect_sizes)== 1 else stdev(effect_sizes)
                authority_utilities[opts.run_name + "util_change"] = sum(utility_changes) / len(utility_changes)

    with open("authority_utilities.json", "w") as f:
        json.dump(dict(sorted(authority_utilities.items())), f)

    for i_type in ["walk", "car"]:
        #plot intervention effect sizes over graph types
        generators = {"er": "Erdős–Rényi", "ba": "Barabasi-Albert", "ws": "Watts-Strogatz"}
        for g in generators:
            plt.errorbar(["Empty network", "Low connectivity", "High connectivity", "Ultra-high connectivity"],
                [authority_utilities[f"none_late{i_type}_effect_size"], authority_utilities[f"{g}_low_late{i_type}_effect_size"], authority_utilities[f"{g}_high_late{i_type}_effect_size"], authority_utilities[f"{g}_ultra_late{i_type}_effect_size"]],
                label=generators[g],
                yerr=[authority_utilities[f"none_late{i_type}_effect_size_std"], authority_utilities[f"{g}_low_late{i_type}_effect_size_std"], authority_utilities[f"{g}_high_late{i_type}_effect_size_std"], authority_utilities[f"{g}_ultra_late{i_type}_effect_size_std"]],
                ecolor="black",
                capsize=3,
            )
        
        plt.xlabel("Social network connecivity")
        plt.ylabel("Effect size")
        plt.legend()

        plt.savefig(f"effect_sizes_{i_type}.png")
        plt.close()