import numpy as np
import librosa
import mir_eval

original_bass, sr = librosa.load('DagstuhlChoirSet_V1.2.3/audio_wav_22050_mono/DCS_LI_FullChoir_Take01_B2_DYN.wav', sr=44100)
original_soprano, sr = librosa.load('DagstuhlChoirSet_V1.2.3/audio_wav_22050_mono/DCS_LI_FullChoir_Take01_S1_DYN.wav', sr=44100)

estimated_bass, sr = librosa.load('SINMF/b.wav', sr=44100)
estimated_soprano, sr = librosa.load('SINMF/s.wav', sr=44100)


min_len = min(len(original_bass), len(estimated_bass))
original_bass = original_bass[:min_len]
estimated_bass = estimated_bass[:min_len]

min_len = min(len(original_soprano), len(estimated_soprano))
original_soprano = original_soprano[:min_len]
estimated_soprano = estimated_soprano[:min_len]

original = np.vstack((original_soprano, original_bass))
estimated = np.vstack((estimated_soprano, estimated_bass))

sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(original, estimated)

print('SDR: ', sdr)
print('SIR: ', sir)
print('SAR: ', sar)

import matplotlib.pyplot as plt
import librosa.display

def plot_spectrogram(audio, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

plot_spectrogram(original_soprano, sr, "original_soprano")
plot_spectrogram(original_bass, sr, "original_bass")

plot_spectrogram(estimated_soprano, sr, "Estimated soprano")
plot_spectrogram(estimated_bass, sr, "Estimated bass")
