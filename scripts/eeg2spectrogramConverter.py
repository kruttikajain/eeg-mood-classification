import argparse
import matplotlib.pyplot as plt # plotting
import mne
import numpy as np # linear algebra
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
plt.set_cmap('jet')

def create_spectrograms(file_name, file_type, data, channels):
    total_time_in_seconds = 180
    window_length_in_seconds = 5
    number_of_parts = int(total_time_in_seconds / window_length_in_seconds)
    overlap_datapoints = False

    for part in range(number_of_parts):
        for index, channel in enumerate(channels):
            fs = 256  # Assume 512 Hz sampling rate for now...   # SampFreq[0, 0] 
            folder_name = filename.split(".")[0] + "_" + str(part)
            start = part * fs * window_length_in_seconds
            end = (part+1) * fs * window_length_in_seconds
            folder_path = os.path.join('/mood-detection/spectrogram_images/relaxed',folder_name) if file_type == 0 else os.path.join('/mood-detection/spectrogram_images/concentration',folder_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            new_filename = "%s_%s.png" % (folder_name, channel)
            new_file_path = os.path.join(folder_path, new_filename)
            x = data[index, start:end]
            f, t, Sxx = signal.spectrogram(x, fs, window = signal.tukey(128), noverlap = 102)
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            fig.figsize= (400, 400)
            fig.gca().set_axis_off()
            fig.gca().xaxis.set_major_locator(plt.NullLocator())
            fig.gca().yaxis.set_major_locator(plt.NullLocator())
            ax.pcolormesh(t, f, np.log10(Sxx))
            fig.savefig(new_file_path, bbox_inches = 'tight', pad_inches = 0)   # save the figure to file
            plt.close(fig)
            x, f, t, Sxx = None, None, None, None


parser = argparse.ArgumentParser(description='Convert .edf file to spectrograms')
parser.add_argument('filename', help="file to convert")
args = parser.parse_args()
filename = args.filename

file_type = 0 if filename.split(".")[0].endswith("E01") or filename.split(".")[0].endswith("E03") else 1
data = mne.io.read_raw_edf(file_name)
raw_data = data.get_data()
channels = data.ch_names
print("Loaded file %s" % (filename))
create_spectrograms(file_name, file_type, raw_data, channels)
print("Completed spectrogram generation for file %s" % (filename))
