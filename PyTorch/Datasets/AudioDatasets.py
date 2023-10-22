import os
import torch
import librosa
import torchaudio
import pandas as pd
from torch.utils.data import Dataset


class UrbanSound8KDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the metadata csv file.
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on an audio sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir, 'fold' + str(self.metadata.iloc[idx, 5]), self.metadata.iloc[idx, 0])
        waveform, sample_rate = librosa.load(audio_name)

        label = self.metadata.iloc[idx, 6]
        label_encoded = int(label)

        sample = {'waveform': waveform, 'sample_rate': sample_rate, 'label': label_encoded}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    # Usage example
    dataset = UrbanSound8KDataset(csv_file='../../data/urban8k/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv',
                                  root_dir='../../data/urban8k/UrbanSound8K/UrbanSound8K/audio/')
    for i in range(20):
        print(len(dataset[i]['waveform']))
