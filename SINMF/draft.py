import numpy as np
import librosa
import csv
from score_informed_nmf import pitch_to_component,freq_to_bin,resynthesize_sources
import score_informed_nmf as nmf
import itertools
from typing import List, Dict, Tuple, Optional, Union, Iterable,NamedTuple
import sklearn.decomposition
import os
import soundfile as sf

class NMFConfiguration(NamedTuple):
    config_name: str
    num_partials: Optional[int]
    phi: float
    tol_on: float
    tol_off: float
    n_fft: int
    mask: bool = True

def initialize_activations_from_score(signal, pitches, score, tol_on, tol_off):
    n_components = len(pitches)
    _n_features, n_samples = signal.S.shape
    H_init = np.zeros((n_components, n_samples), dtype='float32')
    print(H_init.dtype)
    for note in score:
        component = pitch_to_component(note[0], pitches)  # 找到该音符对应的声部
        start_frame, end_frame = librosa.time_to_frames([note[1] - tol_on, note[2] + tol_off],
                                                        sr=signal.sr, hop_length=signal.fft_hop_length)
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, n_samples - 1)
        H_init[component, start_frame:end_frame] = 1  # 激活该声部在时间范围内的时间帧
    print(H_init.dtype)
    return H_init

def initialize_components(signal, pitches, num_partials=15, phi=0.5):
    n_features, _n_samples = signal.S.shape
    fft_freqs = librosa.fft_frequencies(sr=signal.sr, n_fft=signal.n_fft)
    n_components = len(pitches)
    W_init = np.zeros((n_features, n_components),dtype='float32')
    phi_below = phi if isinstance(phi, (float, int)) else phi[0]
    phi_above = phi if isinstance(phi, (float, int)) else phi[1]
    for i, pitch in enumerate(pitches):
        if num_partials is None:
            partials: Iterable[int] = itertools.count(start=1)
        else:
            partials = range(1, num_partials + 1)
        for partial in partials:
            min_freq = librosa.midi_to_hz(pitch - phi_below) * partial
            if min_freq > fft_freqs[-1]:
                break
            max_freq = librosa.midi_to_hz(pitch + phi_above) * partial
            max_freq = min(fft_freqs[-1], max_freq)
            intensity = 1 / (partial ** 2)
            start_bin = freq_to_bin(min_freq, fft_freqs, round='down')
            end_bin = freq_to_bin(max_freq, fft_freqs, round='up')
            W_init[start_bin:end_bin + 1, i] = intensity
    return W_init

def load_score_from_csv(csv_filename, csv_filename2):
    score = []
    pitches=[]
    counter=0
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        #next(reader)  # 如果有表头可以跳过
        for row in reader:
            onset, offset, pitch = float(row[0]), float(row[1]), int(row[2])
            score.append([pitch, onset, offset])
            pitches.append(pitch)
            counter += 1
    print(counter)
    with open(csv_filename2, 'r') as csvfile2:
        reader = csv.reader(csvfile2)
        # next(reader)  # 如果有表头可以跳过
        for row in reader:
            onset, offset, pitch = float(row[0]), float(row[1]), int(row[2])
            score.append([pitch, onset, offset])
            pitches.append(pitch)
            counter += 1
    print(counter)
    print(len(set(pitches)))
    return score, sorted(set(pitches))

def load_audio(filename, sr=22050, n_fft=2048, hop_length=None):
    if hop_length is None:
        hop_length = n_fft // 4
    x, audio_sr = librosa.load(filename, sr=None, mono=False)
    assert sr == audio_sr, f'Expected sample rate {sr} but found {audio_sr} in file: {filename}'
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    X, X_phase = librosa.magphase(S)
    print(X.dtype)
    return nmf.Signal(x, sr, S, X, X_phase, n_fft, hop_length)


def separate(signal,config: NMFConfiguration, csv_filename, csv_filename2):
    output_dir = f'nmf_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    #mix_dir = f'datasets/{dataset}/mix'
    #mix_filename = f'{mix_dir}/chorale_{chorale}_mix.wav'
    #signal = load_audio(mix_filename, n_fft=config.n_fft)

    #midi_files, all_pitches = load_midi_files(chorale, dataset)
    midi_files, all_pitches = load_score_from_csv(csv_filename, csv_filename2)
    W_init = initialize_components(signal, all_pitches, num_partials=config.num_partials, phi=config.phi)
    H_init = initialize_activations_from_score(signal, all_pitches,midi_files, config.tol_on, config.tol_off)
    #signal, pitches, score, tol_on, tol_off
    n_components = H_init.shape[0]
    transformer = sklearn.decomposition.NMF(n_components=n_components, solver='mu', init='custom', max_iter=1000)
    W, H = nmf.decompose_custom(signal.X, n_components=n_components, sort=False, transformer=transformer, W=W_init, H=H_init)
    separated_sources = resynthesize_sources(W, H, pitches, signal)
    source_names = ['B', 'S']
    print(separated_sources)
    for source_name, separated_source in zip(source_names, separated_sources):
        filename = f'{output_dir}/{source_name}.wav'
        sf.write(filename, separated_source, signal.sr)

# 加载你的数据
csv_filename = '/Users/khador/PycharmProjects/DT2470Project/ChoirSeparation-Transcription/DagstuhlChoirSet_V1.2.3/score/DCS_LI_FullChoir_Take01_Stereo_STM_S.csv'
csv_filename2 = '/Users/khador/PycharmProjects/DT2470Project/ChoirSeparation-Transcription/DagstuhlChoirSet_V1.2.3/score/DCS_LI_FullChoir_Take01_Stereo_STM_B.csv'

score, pitches = load_score_from_csv(csv_filename,csv_filename2)
signal = load_audio(filename='/Users/khador/PycharmProjects/DT2470Project/ChoirSeparation-Transcription/combined-DYN.wav')
# 初始化激活矩阵
H_init = initialize_activations_from_score(signal, pitches, score, tol_on=0.1, tol_off=0.1)
print(H_init.shape)
#num_of_ones = np.sum(H_init == 1)
#print(f"Number of 1s in the array: {num_of_ones}")
#print(len(score))
#print([H_init[0][i] for i in range(6541)])

W_init = initialize_components(signal, pitches)
print(W_init.shape)
print(W_init)
config = NMFConfiguration(
        config_name='A',
        num_partials=None,
        phi=1,
        tol_on=0.2,
        tol_off=1,
        n_fft=2048,
    )

separate(signal, config, csv_filename, csv_filename2)