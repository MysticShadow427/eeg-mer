import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import os
import mne

class EEGDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with EEG file paths, class labels, person labels, and arousal values.
            transform (callable, optional): Optional transform to be applied on an EEG signal.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        eeg_path = self.dataframe.iloc[idx]['eeg_path']

        eeg_signal, sample_rate = self.load_eeg_signal(eeg_path)
        eeg_signal = torch.tensor(eeg_signal, dtype=torch.float32)
        #mel_spectrogram = self.generate_mel_spectrogram(eeg_signal, sample_rate)

        class_label = self.dataframe.iloc[idx]['class_label']   # Target class
        person_label = self.dataframe.iloc[idx]['person_label'] # Person identifier
        arousal = self.dataframe.iloc[idx]['arousal']           # Arousal score

        if self.transform:
            eeg_signal = self.transform(eeg_signal)

        eeg_signal = torch.tensor(eeg_signal, dtype=torch.float32)
        class_label = torch.tensor(class_label, dtype=torch.long)
        person_label = torch.tensor(person_label, dtype=torch.long)
        arousal = torch.tensor(arousal, dtype=torch.float32)

        return {
            'eeg_signal': eeg_signal,
            'class_labels': class_label,
            'person_labels': person_label,
            'arousal': arousal
        }

    def load_eeg_signal(self, eeg_path):
        """ Load the EEG signal from a .fif file using MNE """
        if os.path.exists(eeg_path):
            raw = mne.io.read_raw_fif(eeg_path, preload=True)  # Load the raw EEG data
            eeg_signal = raw.get_data()  # Convert to numpy array
            sample_rate = raw.info['sfreq']  # Get the sample rate
        else:
            raise FileNotFoundError(f"EEG file not found: {eeg_path}")
        return eeg_signal, sample_rate

    def generate_mel_spectrogram(self, eeg_signal, sample_rate):
        """ Generate a mel spectrogram from the EEG signal """
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,          # Number of Mel bands
            n_fft=1024,          # Number of FFT components
            hop_length=512       # Hop length
        )
        mel_spectrogram = mel_spec_transform(eeg_signal)
        return mel_spectrogram

def collate_fn(batch):
    """ Collate function to pad EEG signals to the maximum length in the batch """
    signals = [item['eeg_signal'] for item in batch]
    class_labels = [item['class_label'] for item in batch]
    person_labels = [item['person_label'] for item in batch]
    arousals = [item['arousal'] for item in batch]

    signals_padded = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)

    return {
        'eeg_signal': signals_padded,
        'class_labels': torch.stack(class_labels),
        'person_labels': torch.stack(person_labels),
        'arousal': torch.stack(arousals)
    }