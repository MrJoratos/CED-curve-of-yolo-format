import os
import itertools
import math

def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_data(lines):
    data = []
    for line in lines:
        parts = line.strip().split()
        label = float(parts[0])  # Parse the label as a float
        keypoints = [float(x) for x in parts[1:]]
        data.append((label, keypoints))
    return data

def calculate_mae(gt_keypoints, pred_keypoints):
    assert len(gt_keypoints) == len(pred_keypoints), "Number of keypoints must be the same."
        
    # Calculate Absolute Error for each keypoint
    errors = [(abs(gt_keypoints[i] - pred_keypoints[i])**2 +
                  abs(gt_keypoints[i+1] - pred_keypoints[i+1])**2)
                 for i in range(1, len(gt_keypoints), 3)]
        
    # Calculate MAE for each target
    target_mae = (sum(errors) / len(errors))**0.5

    return target_mae

def find_matching_row(gt_data, pred_row):
    x1, y1, w1, h1 = pred_row[1][:4]
    for gt_row in gt_data:
        x2, y2, w2, h2 = gt_row[1][:4]
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if distance <= 0.5 * math.sqrt(w1**2 + h1**2):
            return gt_row
    return None

# Folder paths
gt_folder = "/media/hitcrt/6a071232-a52f-4f53-89ca-fdde738abfd8/assignment10_19/data_original/4kp_data/labeled/labels1"
pred_folder = "/media/hitcrt/6a071232-a52f-4f53-89ca-fdde738abfd8/ultralytics-main(2)/runs/pose/pose+L2/labels"
output_file = "RMSE/pose+L2.txt"

# Get file names from predicted folder
pred_files = os.listdir(pred_folder)

# Open the output file for writing
with open(output_file, 'w') as file:
    # Iterate over each file in the predicted folder
    for pred_file in pred_files:
        # Check if corresponding file exists in the ground truth folder
        gt_file = os.path.join(gt_folder, pred_file)
        if not os.path.exists(gt_file):
            print(f"No corresponding file found for {pred_file}. Skipping.")
            continue

        # Read data from files
        gt_lines = read_txt(gt_file)
        pred_lines = read_txt(os.path.join(pred_folder, pred_file))
        
        # Parse data
        gt_data = parse_data(gt_lines)
        pred_data = parse_data(pred_lines)

        # Iterate through predicted data
        for pred_row in pred_data:
            # Find matching row in ground truth data
            matching_row = find_matching_row(gt_data, pred_row)
            if matching_row:
                gt_keypoints = matching_row[1]
                pred_keypoints = pred_row[1]
                mae_per_row = calculate_mae(gt_keypoints, pred_keypoints)
                file.write(f"{mae_per_row}\n")
        
print("MAE results for all matching rows have been saved to", output_file)
