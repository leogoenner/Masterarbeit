import pandas as pd
import utils

# Calculate times
standard_folder = "2025-01-03 1255"
tucker_folder = "2025-01-03 1421"

standard_path = "models and results/" + standard_folder + "/stats.csv"
tucker_path = "models and results/" + tucker_folder + "/stats.csv"

standard_df = pd.read_csv(standard_path)
tucker_df = pd.read_csv(tucker_path)

standard_times = standard_df["times"]
tucker_times = tucker_df["times"]

print(f"Average time standard: {standard_times.mean():.1f}s")
print(f"Average time tucker: {tucker_times.head(6).mean():.1f}s")
print(f"Reduction by: {1 - tucker_times.head(6).sum()/standard_times.sum():.1%}")

faster_than_five_standard = tucker_times.cumsum() < standard_times.sum()
max_faster = faster_than_five_standard[faster_than_five_standard==True].last_valid_index()
print(f"{max_faster} tucker iterations took {standard_times.sum() - tucker_times.head(max_faster+1).sum():.1f}s less than 5 standard iterations.\n")


# Calculate error rates
print(f"Error rate after 5 standard iterations: {1 - standard_df["accuracies"][5]:.0%}")
print(f"Error rate after {max_faster} tucker iterations: {1 - tucker_df["accuracies"][max_faster]:.0%}\n")


# Calculate numbers of parameters
n_parameters_standard = utils.load_model(standard_folder).get_n_params()
n_parameters_tucker = utils.load_model(tucker_folder).get_n_params()

print(f"Number of parameters standard model: {n_parameters_standard:,}")
print(f"Number of parameters tucker model: {n_parameters_tucker:,}")
print(f"Reduction by: {1 - n_parameters_tucker/n_parameters_standard:.1%}")