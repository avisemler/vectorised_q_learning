import math
import os
import json
from types import SimpleNamespace

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
        axis[x_coord, y_coord].set_ylim(0, 2000)
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
    print("Saving as:", image_name)
    plt.savefig("viz_" + image_name, dpi=300, bbox_inches="tight")
    plt.close()

    #also perform an authority utility calculation, and compute its mean
    authority_utilities = []
    for res in results:
        authority_utilities.append(calculate_authority_utility(opts, res.sum(axis=-2), opts.timesteps-1))

    #mean_authority_utilies = np.mean(np.stack(authority_utilities)).tolist()

    return sum(authority_utilities)/len(authority_utilities)
if __name__ == "__main__":
    import glob

    authority_utilities = {}

    directories_to_plot = glob.glob("/dcs/large/u2107995/res/*/")
    for d in directories_to_plot:
        #load options
        with open(os.path.join(d, "opts.json"), "r") as f:
            opts = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

            #load results
            results = []
            for i, np_file in enumerate(glob.glob(os.path.join(d, "*.npy"))):
                arr = np.load(np_file, allow_pickle=True).astype(np.float32)
                results.append(arr)
            
            authority_utilities[opts.run_name] = plot(opts, results)

    with open("authority_utilities.txt", "w") as f:
        f.write(str(authority_utilities))