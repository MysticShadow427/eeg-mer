import torchaudio.transforms as T
import torchaudio.functional as F
import os
import mne
import numpy as np
import pandas as pd

def augment_waveform(waveform, sample_rate):
    # frequency masking
    frequency_masking = T.FrequencyMasking(freq_mask_param=30)
    augmented_waveform = frequency_masking(waveform)

    # time masking
    time_masking = T.TimeMasking(time_mask_param=50)
    augmented_waveform = time_masking(augmented_waveform)

    # time stretching
    time_stretch = T.TimeStretch()
    augmented_waveform, _ = time_stretch(augmented_waveform)

    # pitch shifting
    augmented_waveform = F.pitch_shift(augmented_waveform, sample_rate, n_steps=4)  # shifting pitch by 4 semitones

    return augmented_waveform

def generate_arousal_targets(eeg_dir, output_csv):
    def extract_arousal_score(eeg_file):
        raw = mne.io.read_raw_fif(eeg_file, preload=True)
        
        raw.filter(1., 40., fir_design='firwin')
        
        bands = {'alpha': (8, 13), 'beta': (13, 30)}
        
        psd, freqs = mne.time_frequency.psd_welch(raw, fmin=1, fmax=40, n_fft=2048)
        
        # extract band specific power
        band_power = {}
        for band, (low, high) in bands.items():
            band_idx = (freqs >= low) & (freqs <= high)
            band_power[band] = psd[:, band_idx].mean(axis=1)
        
        total_power = psd.sum(axis=1)
        relative_power = {band: power / total_power for band, power in band_power.items()}
        
        arousal_score = relative_power['beta'] - relative_power['alpha']
        
        mean_arousal_score = np.mean(arousal_score)
        
        return mean_arousal_score
    
    results = []

    for eeg_file in os.listdir(eeg_directory):
        if eeg_file.endswith('.fif'): 
            full_path = os.path.join(eeg_directory, eeg_file)
            
            arousal_score = extract_arousal_score(full_path)
           
            results.append({
                'filename': eeg_file,
                'arousal_score': arousal_score
            })

    df = pd.DataFrame(results)

    df.to_csv(output_csv, index=False)

    print(f"Arousal scores saved to {output_csv}")