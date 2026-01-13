import numpy as np
import argparse
import os

def process_file(input_filepath, output_filepath, removal_prob):
    try:
        loaded = np.load(input_filepath, allow_pickle=True).item()
    except Exception as e:
        print(f"Failed to load {input_filepath}: {e}")
        return

    if not isinstance(loaded, dict) or "data" not in loaded or "target" not in loaded:
        print(f"File {input_filepath} does not contain the required keys. Skipping.")
        return

    data = loaded["data"]
    target = loaded["target"]

    # If target is 6 or 7, enforce removal of at least 20% of the events.
    if target in [0,1,2,3,4,5,6]:
        effective_removal_prob = max(removal_prob, 0.1)
        num_events = data.shape[0]
        mask = np.random.rand(num_events) >= effective_removal_prob
        new_data = data[mask]
        removed = num_events - new_data.shape[0]
        print(f"{os.path.basename(input_filepath)}: Target {target} detected. "
              f"Removed {removed} events out of {num_events} (~{100 * removed / num_events:.2f}%).")
        loaded["data"] = new_data
    else:
        print(f"{os.path.basename(input_filepath)}: Target {target} is not 6 or 7. No events removed.")

    # Save the (possibly modified) data to the output filepath.
    np.save(output_filepath, loaded)

def main():
    parser = argparse.ArgumentParser(
        description="Process a directory of neuromorphic event stream files. "
                    "For files with target 6 or 7, remove random events (at least 20%)."
    )
    parser.add_argument("input_dir", type=str,
                        help="Input directory containing .npy files")
    parser.add_argument("output_dir", type=str,
                        help="Output directory to save processed .npy files")
    parser.add_argument("--removal_prob", type=float, default=0.2,
                        help="Desired removal probability (forced to at least 0.2 for targets 6 or 7, default: 0.2)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate over each file in the input directory.
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".npy"):
            input_filepath = os.path.join(args.input_dir, filename)
            output_filepath = os.path.join(args.output_dir, filename)
            process_file(input_filepath, output_filepath, args.removal_prob)

if __name__ == "__main__":
    main()


# import os

# def rename_npy_files(directory):
#     # Get a list of all .npy files in the directory
#     npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
#     # Sort the files to ensure a consistent naming order
#     npy_files.sort()

#     # Loop over the files and rename them
#     for idx, file in enumerate(npy_files):
#         old_file_path = os.path.join(directory, file)
#         new_file_name = f"{idx}.npy"
#         new_file_path = os.path.join(directory, new_file_name)

#         # Rename the file
#         os.rename(old_file_path, new_file_path)
#         print(f"Renamed {file} to {new_file_name}")

# # Usage
# directory = "/home/waseem/workspace/microsaccades/NDA_SNN-main/us_dataset/left-resampled/train"  # Replace with the path to your directory
# rename_npy_files(directory)
