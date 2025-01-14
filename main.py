import parameters as p
import utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow warnings
import pandas as pd

# Print input information
if p.mode == "tucker":
    if p.adaptive == False:
        print(p.cnn, "/", "fixed rank tucker mode")
    else:
        print(p.cnn, "/", "adaptive tucker mode")
else:
    print(p.cnn, "/", p.mode, "mode")

# Load / create model
if p.load == True:
    inst = utils.load_model(p.load_name)
else:
    model = utils.create_keras_model(p.cnn)
    inst = utils.create_model(model, p.mode)

# Print information
print(f"Multiplications per evaluation: {inst.get_n_ops():,}")
print(f"Total number of parameters: {inst.get_n_params():,}")

# Get data
train_set, val_set, test_set = utils.get_processed_data(p.dataset, p.batchsize, p.shuffle_buffersize, p.validation_split)

# Initial stats
print("Validation set performance before training:", end=" ")
loss, acc = inst.evaluate(test_set)
if inst.epochs == []:
    if p.mode == "tucker":
        rks = {l: inst.C_dict[l].shape for l in inst.conv_layers}
        inst.save_stats(0, None, loss, acc, rks)
    else:
        inst.save_stats(0, None, loss, acc)

# Initial ranks / shapes
for l in inst.conv_layers:
    if p.mode == "tucker":
        print(f"Kernel in layer {l} has ranks {inst.C_dict[l].shape} and shape {inst.C_dict[l].shape[0:2] + tuple([U.shape[0] for U in inst.Us_dict[l]])}.")
    elif p.mode == "standard":
        print(f"Kernel in layer {l} has shape {inst.W_dict[l].shape}.")

# Train the CNN
utils.training(inst, p.epochs, train_set, test_set, p.mode, p.adaptive, p.tau, p.stepsize, p.decay_rate) # test_set is only used for evaluation

# Stats
stats = {'epochs': inst.epochs, 'times': inst.times, 'losses': inst.losses, 'accuracies': inst.accuracies}
if p.mode == "tucker":
    stats['ranks'] = inst.ranks
df_stats = pd.DataFrame(data=stats).set_index('epochs')
print(df_stats)

# Save CNN & parameters
if p.save == True:
    utils.save_model_and_results(inst, df_stats)