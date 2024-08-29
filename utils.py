import torchaudio.transforms as T
import torchaudio.functional as F

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

