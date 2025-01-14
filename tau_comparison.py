import parameters as p
import utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow warnings
from datetime import datetime
import shutil
import pandas as pd
import numpy as np

# Taus to be tested
taus = np.logspace(-4, 0, 17)

# Initialize lists for tracking
ranks = []
times = []
losses = []
accuracies = []
n_params = []
n_ops = []

# Create model
model = utils.create_keras_model(p.cnn)

# Get data
train_set, val_set, test_set = utils.get_processed_data(p.dataset, p.batchsize, p.shuffle_buffersize, p.validation_split)

# Train the CNN
for i,tau in enumerate(taus):
    inst = utils.create_model(model, "tucker")
    if i == 0:
        for l in inst.conv_layers:
            print(f"Kernel in layer {l} has shape {inst.C_dict[l].shape[0:2] + tuple([U.shape[0] for U in inst.Us_dict[l]])}.")
    print(f"tau = {tau}:")
    utils.training(inst, 1, train_set, test_set, "tucker", True, tau, p.stepsize, p.decay_rate)
    ranks += inst.ranks
    times += inst.times
    losses += inst.losses
    accuracies += inst.accuracies
    n_params.append(inst.get_n_params())
    n_ops.append(inst.get_n_ops())
    del inst # instance is not needed anymore, compare https://stackoverflow.com/a/3306310

# Stats
stats = {'taus': taus, 'ranks': ranks, 'times': times, 'losses': losses, 'accuracies': accuracies, 'n_params': n_params, 'n_ops': n_ops}
df_stats = pd.DataFrame(data=stats).set_index('taus')
print(df_stats)

# Create folder
date = datetime.today().strftime('%Y-%m-%d %H%M')
save_folder = "models and results/" + date
cwd = os.getcwd()
os.makedirs(os.path.join(cwd, save_folder))
# Save stats
df_stats.to_csv(save_folder + '/stats.csv')
# Save parameter file
shutil.copy('parameters.py', save_folder)
# Save information regarding the standard model
inst = utils.create_model(model, "standard")
utils.training(inst, 1, train_set, test_set, "standard", None, None, p.stepsize, p.decay_rate)
filename = "models and results/" + date + "/standard_model.txt"
f = open(filename, "x")
f.write(f"n_params {inst.get_n_params()}" + "\n")
f.write(f"n_ops {inst.get_n_ops()}" + "\n")
[accuracy] = inst.accuracies
[time] = inst.times
[loss] = inst.losses
f.write(f"accuracy {accuracy}" + "\n")
f.write(f"time {time}" + "\n")
f.write(f"loss {loss}")
f.close()