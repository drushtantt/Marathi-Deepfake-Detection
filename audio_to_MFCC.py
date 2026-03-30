import os
import librosa
import h5py
import numpy as np


def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def save_to_hdf5(data, filename, dataset_name='mfcc'):
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=data)


def process_audio_files(input_dir, output_dir, n_mfcc=13):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):  # Add other audio formats if needed
                audio_path = os.path.join(root, file)
                mfcc = extract_mfcc(audio_path, n_mfcc)

                # Create the output file path
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.h5")
                save_to_hdf5(mfcc, output_path)


input_directory = '/Users/atharvchoughule/Downloads/pythonProject/amazon_audio'
output_directory = '/Users/atharvchoughule/Downloads/pythonProject/MFCC/amazon_audio_MFCC'
n_mfcc = 13  # Number of MFCCs to extract

process_audio_files(input_directory, output_directory, n_mfcc)