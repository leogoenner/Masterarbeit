import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots import cycler
from tueplots.constants.color import palettes
import numpy as np

# Load dataframes
standard_folder = "2025-01-03 1255"
tucker_folder = "2025-01-03 1421"
standard_path = "models and results/" + standard_folder + "/stats.csv"
tucker_path = "models and results/" + tucker_folder + "/stats.csv"
df_standard = pd.read_csv(standard_path)
df_tucker = pd.read_csv(tucker_path)
df_standard = df_standard
df_tucker = df_tucker.head(6)

# Set plotting style
plt.rcParams.update(bundles.icml2022(column="full", nrows=1, ncols=1, usetex=False))
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
width = 0.3  # bar width

# Create plot
fig, ax = plt.subplots(1,1)
x = df_standard.index
y = np.arange(0,1.1,0.1)
ax.bar(x - width/2, 1 - df_standard["accuracies"], width, label='Standard') # based on https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
ax.bar(x + width/2, 1 - df_tucker["accuracies"], width, label='Tucker')

# Add labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Epoch')
ax.set_ylabel('Error rate')
ax.set_xticks(x)
ax.set_yticks(y)
ax.set_xticklabels(x)
ax.legend()
plt.ylim([0,1])
ax.set_axisbelow(True)
plt.grid(axis='y', lw=0.1)

# Save figure
fig.savefig("Masterarbeit/fig/plt_errors.pdf")

# Runtimes
print(f"Standard average time per epoch: {df_standard["times"].mean():.1f}s")
print(f"Tucker average time per epoch: {df_tucker["times"].mean():.1f}s")