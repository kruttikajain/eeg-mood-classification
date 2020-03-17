import os
import numpy as np

base_path = "/mood-detection/numpy_data_original_size"
concentration_data_array = None
relaxed_data_array = None
print("loading data from .npy files...")

for filename in os.listdir(base_path):
    filename = os.path.join(base_path, filename)
    if "concentration" in filename:
        print(filename)
        concentration_data_array = np.concatenate((concentration_data_array, np.load(filename))) if concentration_data_array is not None else np.load(filename)
    if "relaxed_array" in filename:
        print(filename)
        relaxed_data_array = np.concatenate((relaxed_data_array, np.load(filename))) if relaxed_data_array is not None else np.load(filename)
print(concentration_data_array.shape, relaxed_data_array.shape)
print(len(concentration_data_array), len(relaxed_data_array))
