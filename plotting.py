import math

import matplotlib.pyplot as plt 
import numpy as np

LABEL_SIZE = 9
LINE_WIDTH = 1.0
AGENT_TYPES = 3
action_names = ["car", "bus", "walk"]
COLOURS = ["blue", "red", "orange", "green", "purple"]

def plot(opts, result, show=False, title="Plot"):
    #plt.plot(result.sum(axis=-2))
    #plt.savefig("output.png", bbox_inches="tight")

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
            axis[x_coord, y_coord].plot(np.arange(opts.timesteps),
                result[:,1000*agent_type:1000*(agent_type + 1),j].sum(axis=-1),
                label=action_names[j],
                color=COLOURS[j],
                lw=str(LINE_WIDTH)
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
        axis[x_coord, y_coord].plot(np.arange(opts.timesteps),
            result[:,:,j].sum(axis=-1),
            label=action_names[j],
            color=COLOURS[j],
            lw=str(LINE_WIDTH)
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