import os
import numpy as np
import logging
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img

logging.basicConfig(level=logging.INFO, filename='image_to_numpy_10.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

target_path = "/mood-detection/numpy_data_10_percent_size_overlapped"
if not os.path.exists(target_path):
        os.mkdir(target_path)
target_type = "ten_pc_"
target_dimensions = (30, 45, 1)

base_path = "/mood-detection/spectrogram_images_overlapped/concentration"

concentration_data_array = []
counter = 0
for foldername in os.listdir(base_path):
    logging.debug(foldername)
    spectrogram_dict = {}
    for filename in os.listdir(os.path.join(base_path, foldername)):
        example = foldername
        channel = filename.split('.')[0].split("_")[2]
        filepath = os.path.join(os.path.join(base_path, foldername), filename)
        img = load_img(filepath, target_size=(30, 45))
        img_array = img_to_array(img)
        spectrogram_dict[channel] = img_array
        array = np.zeros(target_dimensions)
    for channel in spectrogram_dict.keys():
        array = np.concatenate((array, spectrogram_dict[channel]), axis=2)
    concentration_data_array.append(np.delete(array,0,2))
    spectrogram_dict = None
    counter += 1
    if counter % 360 == 0:
        concentration_data_array = np.array(concentration_data_array)
        logging.debug(concentration_data_array.shape)
        filename = os.path.join(target_path, "concentration_array_" + target_type + str(counter // 360) +".npy")
        np.save(filename, concentration_data_array)
        concentration_data_array = []
concentration_data_array = np.array(concentration_data_array)
logging.debug(concentration_data_array.shape)
np.save(os.path.join(target_path, "concentration_array_" + target_type + "_final.npy"), concentration_data_array)

base_path = "/mood-detection/spectrogram_images_overlapped/relaxed"

relaxed_data_array = []
counter = 0
for foldername in os.listdir(base_path):
    logging.debug(foldername)
    spectrogram_dict = {}
    for filename in os.listdir(os.path.join(base_path, foldername)):
        example = foldername
        channel = filename.split('.')[0].split("_")[2]
        filepath = os.path.join(os.path.join(base_path, foldername), filename)
        img = load_img(filepath, target_size=(30, 45))
        img_array = img_to_array(img)
        spectrogram_dict[channel] = img_array
        array = np.zeros(target_dimensions)
    for channel in spectrogram_dict.keys():
        array = np.concatenate((array, spectrogram_dict[channel]), axis=2)
    relaxed_data_array.append(np.delete(array,0,2))
    spectrogram_dict = None
    counter += 1
    if counter % 360 == 0:
        relaxed_data_array = np.array(relaxed_data_array)
        logging.debug(relaxed_data_array.shape)
        filename = os.path.join(target_path, "relaxed_array_" + target_type + str(counter // 360) +".npy")
        np.save(filename, relaxed_data_array)
        relaxed_data_array = []
relaxed_data_array = np.array(relaxed_data_array)
logging.debug(relaxed_data_array.shape)
np.save(os.path.join(target_path, "relaxed_array_" + target_type + "_final.npy"), relaxed_data_array)
