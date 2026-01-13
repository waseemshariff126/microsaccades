# import os
# import numpy as np
# from sklearn.model_selection import train_test_split

# # 1. Define the mapping for classes to numerical labels
# class_mapping = {
#     '0.5_left': 0,
#     '0.75_left': 1,
#     '1.0_left': 2,
#     '1.25_left': 3,
#     '1.5_left': 4,
#     '1.75_left': 5,
#     '2.0_left': 6
# }

# # class_mapping = {
# #     '0.5_right': 0,
# #     '0.75_right': 1,
# #     '1.0_right': 2,
# #     '1.25_right': 3,
# #     '1.5_right': 4,
# #     '1.75_right': 5,
# #     '2.0_right': 6
# # }
# # 2. Gather samples from each class folder
# samples = []
# base_dir = '/home/waseem/workspace/microsaccades/NDA_SNN-main/us_dataset/left-resampled'  # Change this to your actual base directory
# for class_name, target in class_mapping.items():
#     # npy_dir = os.path.join(base_dir, class_name, 'NPYs')
#     npy_dir = os.path.join(base_dir, class_name)#, 'NPYs')
#     for file in os.listdir(npy_dir):
#         if file.endswith('.npy'):
#             file_path = os.path.join(npy_dir, file)
#             original_data = np.load(file_path, allow_pickle=True)
#             sample = {'data': original_data, 'target': target}
#             samples.append(sample)

# # 3. Shuffle and split into train and test sets
# # Stratify by target to ensure an even distribution across classes
# train_samples, test_samples = train_test_split(
#     samples,
#     test_size=0.2,
#     random_state=42,
#     stratify=[s['target'] for s in samples]
# )

# # 4. Save the new files into 'train' and 'test' folders
# # Create the 'train' directory inside base_dir
# os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)

# # Similarly, create the 'test' directory
# os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)

# for i, sample in enumerate(train_samples):
#     np.save(os.path.join(base_dir, 'train', f'sample_{i}.npy'), sample, allow_pickle=True)

# for i, sample in enumerate(test_samples):
#     np.save(os.path.join(base_dir, 'test', f'sample_{i}.npy'), sample, allow_pickle=True)



# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# import psutil

# # 1. Define the mapping for classes to numerical labels
# class_mapping = {
#     '0.5_left': 0,
#     '0.75_left': 1,
#     '1.0_left': 2,
#     '1.25_left': 3,
#     '1.5_left': 4,
#     '1.75_left': 5,
#     '2.0_left': 6
# }

# # 2. Gather samples from each class folder
# samples = []
# base_dir = '/home/waseem/workspace/microsaccades/NDA_SNN-main/us_dataset/left-resampled'  # Change this to your actual base directory

# # Function to load data in chunks
# def load_data_in_chunks(file_path, chunk_size=1000):
#     # Open the .npy file and load data in chunks
#     data = np.load(file_path, allow_pickle=True)
#     for start in range(0, data.shape[0], chunk_size):
#         end = min(start + chunk_size, data.shape[0])
#         yield data[start:end]

# for class_name, target in class_mapping.items():
#     npy_dir = os.path.join(base_dir, class_name)
#     for file in os.listdir(npy_dir):
#         if file.endswith('.npy'):
#             file_path = os.path.join(npy_dir, file)
#             for chunk in load_data_in_chunks(file_path):
#                 sample = {'data': chunk, 'target': target}
#                 samples.append(sample)

# # 3. Shuffle and split into train and test sets
# train_samples, test_samples = train_test_split(
#     samples,
#     test_size=0.2,
#     random_state=42,
#     stratify=[s['target'] for s in samples]
# )

# # 4. Save the new files into 'train' and 'test' folders
# # Create the 'train' directory inside base_dir
# os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)

# # Similarly, create the 'test' directory
# os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)

# # Define a function to log memory usage
# def log_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     print(f"Memory used: {mem_info.rss / 1024 / 1024:.2f} MB")

# # 5. Process and save data in smaller batches to avoid memory overflow
# batch_size = 1000  # Adjust the batch size as needed

# # Save training samples in batches
# for i in range(0, len(train_samples), batch_size):
#     batch_samples = train_samples[i:i+batch_size]
#     for j, sample in enumerate(batch_samples):
#         # Save each sample individually
#         np.save(os.path.join(base_dir, 'train', f'sample_{i + j}.npy'), sample, allow_pickle=True)
#     log_memory_usage()  # Log memory usage after processing each batch

# # Save testing samples in batches
# for i in range(0, len(test_samples), batch_size):
#     batch_samples = test_samples[i:i+batch_size]
#     for j, sample in enumerate(batch_samples):
#         # Save each sample individually
#         np.save(os.path.join(base_dir, 'test', f'sample_{i + j}.npy'), sample, allow_pickle=True)
#     log_memory_usage()  # Log memory usage after processing each batch

# print("Data preparation complete.")


import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# 1. Define the mapping for classes to numerical labels
class_mapping = {
    '0.5_left': 0,
    '0.75_left': 1,
    '1.0_left': 2,
    '1.25_left': 3,
    '1.5_left': 4,
    '1.75_left': 5,
    '2.0_left': 6
}

# 2. Gather and process samples in batches
def load_and_process_batch(batch_files, batch_size=100):
    batch_samples = []
    
    for file_path in batch_files:
        original_data = np.load(file_path, allow_pickle=True)
        # Load data into GPU memory
        data_tensor = torch.tensor(original_data).to('cuda')
        
        # Add to batch
        batch_samples.append({'data': data_tensor, 'target': None})  # No target yet, will add later

        if len(batch_samples) == batch_size:
            yield batch_samples
            batch_samples = []  # Reset for next batch
    
    if batch_samples:
        yield batch_samples  # Yield any remaining samples that are less than batch_size

# 3. Split and save data in batches
def process_and_save_data():
    base_dir = '/home/waseem/workspace/microsaccades/NDA_SNN-main/us_dataset/left-resampled'
    # Create the 'train' and 'test' directories if they do not exist
    os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)


    # Function to process samples and split into train/test
    total_samples = 0  # Keep track of total samples processed
    global_sample_index = 0  # Global index for naming files uniquely across all classes

    for class_name, target in class_mapping.items():
        npy_dir = os.path.join(base_dir, class_name)
        files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')]
        
        print(f"Processing class: {class_name}, {len(files)} files found.")  # Print file count

        batch_size = 100  # Process 100 samples at a time
        class_train_samples = []  # List for train samples of this class
        class_test_samples = []   # List for test samples of this class

        # Iterate through batches of files
        for batch_index, batch_samples in enumerate(load_and_process_batch(files, batch_size)):
            print(f"Processing batch {batch_index+1}...")  # Print batch processing
            
            # Assign target labels
            for sample in batch_samples:
                sample['target'] = target

            # Split into train and test samples for this batch
            train_batch, test_batch = train_test_split(batch_samples, test_size=0.2, stratify=[s['target'] for s in batch_samples])

            class_train_samples.extend(train_batch)
            class_test_samples.extend(test_batch)

            # Save the train and test batches directly to disk with unique filenames
            save_batch_to_disk(train_batch, os.path.join(base_dir, 'train'), batch_index, global_sample_index)
            save_batch_to_disk(test_batch, os.path.join(base_dir, 'test'), batch_index, global_sample_index)

            total_samples += len(batch_samples)  # Update total samples processed
            global_sample_index += len(batch_samples)  # Update global index for unique filenames

        print(f"Finished processing class: {class_name}, Total samples: {len(class_train_samples)} train, {len(class_test_samples)} test.")

    print(f"Data processing complete. Total samples processed: {total_samples}.")

# 4. Save a batch of samples to disk with unique filenames
def save_batch_to_disk(samples, directory, batch_index, global_sample_index):
    for i, sample in enumerate(samples):
        # Create a unique filename by combining global_sample_index and batch_index
        filename = f'sample_{global_sample_index + i}.npy'
        
        # Convert tensors to CPU if they are on GPU; leave other items unchanged
        sample_cpu = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
        np.save(os.path.join(directory, filename), sample_cpu, allow_pickle=True)

# 5. Run the processing and saving
process_and_save_data()

print("All data has been processed and saved successfully.")

