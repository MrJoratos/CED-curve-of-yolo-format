import numpy as np
import matplotlib.pyplot as plt
import os

# Function to extract filename from path
def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

# Define a function for moving average
def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Function to filter out values beyond a threshold
def filter_values(data, threshold):
    return data[data <= threshold]

# Initialize lists to store NME results and filenames
nme_results_list = []
filename_list = []

# Directory containing the txt files
directory = "RMSE"

# Iterate over each txt file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        filename_list.append(get_filename(file_path))
        
        # Load NME results from the file
        nme_results = []
        with open(file_path, "r") as file:
            for line in file:
                nme_results.append(float(line.strip()))
        
        # Filter out values beyond 0.2
        nme_results_filtered = filter_values(np.array(nme_results), 0.2)
        
        nme_results_list.append(np.sort(nme_results_filtered))

# Plot CED curves for each file with moving average smoothing
plt.figure(figsize=(10, 6))
for i in range(len(nme_results_list)):
    sorted_nme = nme_results_list[i]
    cumulative_percentage = np.arange(1, len(sorted_nme) + 1) / len(sorted_nme) 
    smoothed_nme = moving_average(sorted_nme, window_size=10)  # Adjust window_size as needed
    smoothed_percentage = moving_average(cumulative_percentage, window_size=10)  # Adjust window_size as needed
    plt.plot(smoothed_nme, smoothed_percentage, marker='o', linestyle='-', markersize=0.01, linewidth=0.8,label=filename_list[i])
    mean_nme = np.mean(nme_results_list[i])
    print(f"Mean RMSE for {filename_list[i]}: {mean_nme:.6f}")

plt.xlabel('Root Mean Squared Error (RMSE)')
plt.ylabel('Cumulative Percentage ')
plt.title('Cumulative Error Distribution (CED) Curve ')
plt.legend()
plt.grid(True)
plt.xlim(0, 0.005) 
 


# Save the image as EPS file
plt.savefig('CED_curve.eps', format='eps',dpi=300)

plt.show()
