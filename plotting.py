import math

import matplotlib.pyplot as plt 
import numpy as np

LABEL_SIZE = 9
LINE_WIDTH = 1.0
AGENT_TYPES = 3
action_names = ["car", "bus", "walk"]
COLOURS = ["blue", "red", "green", "green", "purple"]

def mean_and_std(l):
    """
    Computes mean and standard deviation for a list l of numpy
    arrays that is potentially too large to store in memory as a single stacked 
    array.
    """
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

def plot(opts, results, show=False, title=""):
    #compute mean and standard deviation over each run
    mean, std = mean_and_std(results)

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
            plot_line = mean[:,1000*agent_type:1000*(agent_type + 1),j].sum(axis=-1)
            band_width = std[:,1000*agent_type:1000*(agent_type + 1),j].sum(axis=-1)
            axis[x_coord, y_coord].plot(np.arange(opts.timesteps),
                plot_line,
                label=action_names[j],
                color=COLOURS[j],
                lw=str(LINE_WIDTH)
            )
            axis[x_coord, y_coord].fill_between(np.arange(opts.timesteps),
                plot_line - band_width/2,
                plot_line + band_width/2,
                facecolor=COLOURS[j],
                alpha=0.2,
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
    for j in range(opts.n_actions):
        x_coord = plot_count // dimension
        y_coord = plot_count % dimension
        plot_line = mean[:,:,j].sum(axis=-1)
        band_width = std[:,:,j].sum(axis=-1)
        axis[x_coord, y_coord].plot(np.arange(opts.timesteps),
            plot_line,
            label=action_names[j],
            color=COLOURS[j],
            lw=str(LINE_WIDTH)
        )
        axis[x_coord, y_coord].fill_between(np.arange(opts.timesteps),
            plot_line - band_width/2,
            plot_line + band_width/2,
            facecolor=COLOURS[j],
            alpha=0.2,
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
    plt.savefig("output.png", dpi=300, bbox_inches="tight")