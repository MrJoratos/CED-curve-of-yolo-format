import os

def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_data(lines):
    data = []
    for line in lines:
        parts = line.strip().split()
        label = float(parts[0])  # Parse the label as a float
        keypoints = [float(x) for x in parts[5:]]
        data.append((label, keypoints))
    return data


def initialize_missing_rows(data, max_rows):
    while len(data) < max_rows:
        data.append((0, [0.0] * 8))  # Assuming 8 keypoints are present
    return data

def calculate_nme(gt_data, pred_data):
    assert len(gt_data) == len(pred_data), "Number of targets must be the same."
    
    total_nme = 0.0
    for gt, pred in zip(gt_data, pred_data):
        gt_keypoints = gt[1]
        # print(gt_keypoints)
        pred_keypoints = pred[1]
        
        assert len(gt_keypoints) == len(pred_keypoints), "Number of keypoints must be the same."
        
        # Calculate Euclidean distance for each keypoint
        distances = [((gt_keypoints[i] - pred_keypoints[i]) ** 2 +
                      (gt_keypoints[i+1] - pred_keypoints[i+1]) ** 2) ** 0.5
                     for i in range(0, len(gt_keypoints), 2)]
        
        # Calculate NME for each target
        target_nme = sum(distances) / len(distances)
        total_nme += target_nme
    
    # Calculate average NME
    average_nme = total_nme / len(gt_data)
    return average_nme

# Folder paths
gt_folder = "/media/hitcrt/6a071232-a52f-4f53-89ca-fdde738abfd8/assignment10_19/data_original/4kp_data/labeled/labels"
pred_folder = "/media/hitcrt/6a071232-a52f-4f53-89ca-fdde738abfd8/ultralytics-main(2)/runs/pose/Myloss192/labels"
output_file = "NME/nme_results_Myloss-X.txt"

# Get file names from both folders
gt_files = os.listdir(gt_folder)
pred_files = os.listdir(pred_folder)

# Sort file names to ensure correspondence
gt_files.sort()
pred_files.sort()

# Open the output file for writing
with open(output_file, 'w') as file:
    # Calculate NME for each pair of corresponding files
    for gt_file, pred_file in zip(gt_files, pred_files):
        # Read data from files
        gt_lines = read_txt(os.path.join(gt_folder, gt_file))
        pred_lines = read_txt(os.path.join(pred_folder, pred_file))
        
        # Parse data
        gt_data = parse_data(gt_lines)
        pred_data = parse_data(pred_lines)
        
        # Initialize missing rows if needed
        max_rows = max(len(gt_data), len(pred_data))
        gt_data = initialize_missing_rows(gt_data, max_rows)
        pred_data = initialize_missing_rows(pred_data, max_rows)
        
        # Calculate NME
        nme = calculate_nme(gt_data, pred_data)
        
        # Write results to the output file
        file.write(f"{nme}\n")

print("NME results have been saved to", output_file)
