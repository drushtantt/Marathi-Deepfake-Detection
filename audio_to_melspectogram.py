import os
import librosa
import h5py
import numpy as np


def extract_melspectrogram(audio_path, n_mels=128, hop_length=256, n_fft=1024):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_resized = librosa.util.fix_length(S_dB, size=n_mels)
    return S_resized


def save_to_hdf5(data, filename, dataset_name='melspectrogram'):
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=data)


def process_audio_files(input_dir, output_dir, n_mels=128, hop_length=256, n_fft=1024):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):  # Add other audio formats if needed
                audio_path = os.path.join(root, file)
                mel_spectrogram = extract_melspectrogram(audio_path, n_mels, hop_length, n_fft)

                # Resize to 128x128
                mel_spectrogram_resized = librosa.util.fix_length(mel_spectrogram, size=n_mels, axis=1)
                mel_spectrogram_resized = mel_spectrogram_resized[:n_mels, :n_mels]

                # Create the output file path
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.h5")
                save_to_hdf5(mel_spectrogram_resized, output_path)


input_directory = '/Users/atharvchoughule/Downloads/pythonProject/amazon_audio'
output_directory = '/Users/atharvchoughule/Downloads/pythonProject/MELSPECTRGRAMS/amazon_audio_melspectogram'
n_mels = 128  # Number of Mel bands to generate

process_audio_files(input_directory, output_directory, n_mels)