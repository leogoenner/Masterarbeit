import matplotlib.pyplot as plt
from tueplots import bundles
import numpy as np
import pandas as pd

# Load values
folder = "2025-01-05 2226"
path = "models and results/" + folder + "/"
df = pd.read_csv(path + "stats.csv")
taus = df["taus"]
y_params = df["n_params"]
y_err = 1 - df["accuracies"]

with open(path + "standard_model.txt") as f:
    lines = f.readlines()
    n_params_st = float(lines[0].split()[1])
    acc_st = float(lines[2].split()[1])
    err_rate_st = 1 - acc_st

# Parameters
c1 = np.array([34,136,51])/255
c2 = np.array([170,51,119])/255

# Set plotting style
with plt.rc_context(bundles.icml2022(column='full', nrows=1, ncols=1, usetex=True)):
    
    # Create plot
    fig, ax1 = plt.subplots(1,1)
    ax1.set_xlabel("Tolerance $\\tau$")
    plt.xscale('log')
    plt.xlim([0.8*min(taus), 1.2*max(taus)])

    # Horizontal line for standard model
    ax1.hlines(y=n_params_st, color=c1, linestyle='--', linewidth=0.5, xmin=0, xmax=10)
    t = plt.text(1e-4, n_params_st, r'\# parameters standard model', ha='left', va='center', color=c1, size='small')
    t.set_bbox(dict(facecolor='white', alpha=1, linewidth=0)) # based on https://stackoverflow.com/a/23698794

    # Number of parameters
    ax1.scatter(taus, y_params, marker='x', color=c1, label="Number of parameters") # color from https://personal.sron.nl/~pault/
    ax1.figure.canvas.draw() # based on https://stackoverflow.com/a/45766598
    offset = ax1.yaxis.get_major_formatter().get_offset()
    ax1.yaxis.set_label_text(f"Number of parameters [{offset}]")
    plt.ylim([min(y_params)-0.05*(max(y_params)-min(y_params)), max(y_params)+0.05*(max(y_params)-min(y_params))])

    # Second axis (based on https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html)
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.set_ylabel("Error rate")
    ax2.scatter(taus, y_err, color=c2, label="Error rate") # color from https://personal.sron.nl/~pault/

    # Horizontal line for standard model
    ax2.hlines(y=err_rate_st, color=c2, linestyle='--', linewidth=0.5, xmin=0, xmax=10)
    t = plt.text(2e-2, err_rate_st, 'error rate standard model', ha='left', va='center', color=c2, size='small')
    t.set_bbox(dict(facecolor='white', alpha=1, linewidth=0)) # based on https://stackoverflow.com/a/23698794

    # Move scatter plot for number of parameters in front
    ax1.set_zorder(ax2.get_zorder()+1) # from https://stackoverflow.com/a/59395256
    ax1.patch.set_visible(False)

    # Hide offset text, show legend
    ax1.yaxis.offsetText._visible = False
    fig.legend(loc='center', bbox_to_anchor=(0.62,0.95))

# Save figure
fig.savefig("Masterarbeit/fig/plt_ranks.pdf")