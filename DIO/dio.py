import numpy as np
import soundfile as sf
import pyworld as pw
import librosa
import os
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import mir_eval
from scipy.signal import butter, filtfilt
import librosa.display

# STFT计算函数 (保持原样)
def stft(data, window=np.hanning(1024), hopsize=256.0, nfft=1024.0, fs=44100.0):
    lengthWindow = window.size
    lengthData = data.size
    numberFrames = np.ceil(lengthData / np.double(hopsize)) + 2
    newLengthData = (numberFrames-1) * hopsize + lengthWindow
    data = np.concatenate((np.zeros(int(lengthWindow/2)), data))
    data = np.concatenate((data, np.zeros(int(newLengthData - data.size))))
    numberFrequencies = int(nfft / 2 + 1)
    STFT = np.zeros([int(numberFrames), int(numberFrequencies)], dtype=complex)
    for n in np.arange(int(numberFrames)):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * data[int(beginFrame):int(endFrame)]
        STFT[int(n), :] = np.fft.rfft(frameToProcess, int(nfft), norm="ortho")
    return STFT

# 反向STFT（重构信号）
def istft(mag, phase, window=np.hanning(1024), hopsize=256.0, nfft=1024.0):
    X = mag * np.exp(1j * phase)
    X = X.T
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = int(hopsize * (numberFrames - 1) + lengthWindow)
    normalisationSeq = np.zeros(lengthData)
    data = np.zeros(lengthData)
    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:, n], int(nfft), norm="ortho")
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] += window
        data[beginFrame:endFrame] += window * frameTMP
    normalisationSeq[normalisationSeq == 0] = 1
    data = data / normalisationSeq
    return data[int(lengthWindow / 2.0):]

def check_for_nan(f0, sp, ap):
    # 检查 f0 是否包含 NaN
    if np.isnan(f0).any():
        print("警告: f0 中包含 NaN 值！")
    # 检查频谱包络 sp 是否包含 NaN
    if np.isnan(sp).any():
        print("警告: 频谱包络 sp 中包含 NaN 值！")
    # 检查非周期成分 ap 是否包含 NaN
    if np.isnan(ap).any():
        print("警告: 非周期成分 ap 中包含 NaN 值！")

def handle_nan_values(data):
    # 用零替换 NaN 值
    nan_indices = np.isnan(data)
    if nan_indices.any():
        data[nan_indices] = 0
    return data

def apply_frequency_mask(stft_matrix, threshold, high_pass=True):
    mask = np.zeros_like(stft_matrix)
    if high_pass:
        mask[threshold:] = 1  # 保留高于阈值的频率
    else:
        mask[:threshold] = 1  # 保留低于阈值的频率
    return stft_matrix * mask

# 带通滤波器
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def plot_spectrogram(audio, sr, title):
    # 将音频中的非有限值（如 NaN, Inf）替换为 0
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 计算 STFT 并生成频谱图
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'STFT Spectrogram for {title}')
    plt.show()


"""
def extract_f0_and_separate(audio_file, fs=44100):
    audio, sr = librosa.load(audio_file, sr=fs)
    audio = audio.astype(np.float64)
    # 提取F0（基频）
    f0, t = pw.dio(audio, sr)    # 基于DIO算法提取F0
    f0 = pw.stonemask(audio, f0, t, sr)    # 基于STONE MASK优化F0
    sp = pw.cheaptrick(audio, f0, t, sr)   # 计算频谱包络
    ap = pw.d4c(audio, f0, t, sr)          # 计算非周期成分
    
    # 检查 f0, sp, ap 中是否有 NaN 值
    check_for_nan(f0, sp, ap)

    # 处理 NaN 值
    f0 = handle_nan_values(f0)
    sp = handle_nan_values(sp)
    ap = handle_nan_values(ap)

    # 动态设定男低音和女高音的F0阈值
    f0_mean = np.mean(f0[f0 > 0])  # 计算F0的平均值（只考虑非零值）
    f0_std = np.std(f0[f0 > 0])    # 计算F0的标准差
    #bass_threshold = max(f0_mean - 1.5 * f0_std, 80)    # 男低音阈值不能低于80Hz
    #soprano_threshold = min(f0_mean + 1.5 * f0_std, 350)  # 女高音阈值不应超过1000Hz
    bass_threshold = 512
    soprano_threshold = 512

    # 使用动态阈值分离男低音和女高音
    bass_f0 = np.where(f0 < bass_threshold, f0, 0)   # 男低音F0
    soprano_f0 = np.where(f0 >= soprano_threshold, f0, 0) # 女高音F0
    
    # 重构音频
    bass_audio = pw.synthesize(bass_f0, sp, ap, sr)
    soprano_audio = pw.synthesize(soprano_f0, sp, ap, sr)

    # 检查音频有效性
    check_audio_validity(bass_audio)
    check_audio_validity(soprano_audio)
    
    return bass_audio, soprano_audio, f0, bass_threshold, soprano_threshold

    """
def extract_f0_and_separate(audio_file, fs=44100):
    audio, sr = librosa.load(audio_file, sr=fs)
    audio = audio.astype(np.float64)
    
    # 提取 F0、频谱包络和非周期成分
    f0, t = pw.dio(audio, sr)
    f0 = pw.stonemask(audio, f0, t, sr)
    sp = pw.cheaptrick(audio, f0, t, sr)
    ap = pw.d4c(audio, f0, t, sr)
    
    # 动态设定男低音和女高音的F0阈值
    bass_threshold = 512  # 男低音阈值设定为 512Hz 以下
    soprano_threshold = 512  # 女高音设定为 512Hz 以上

    # 使用动态阈值分离男低音和女高音
    bass_f0 = np.where(f0 < bass_threshold, f0, 0)   # 男低音F0
    soprano_f0 = np.where(f0 >= soprano_threshold, f0, 0) # 女高音F0
    
    # 重构音频
    bass_audio = pw.synthesize(bass_f0, sp, ap, sr)
    soprano_audio = pw.synthesize(soprano_f0, sp, ap, sr)
    
    return bass_audio, soprano_audio


# 保存音频
def save_audio(audio, filename, fs=44100):
    # 获取目录路径
    directory = os.path.dirname(filename)
    
    # 如果目录不存在，则创建它
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存音频
    sf.write(filename, audio, fs)

# 音频分离效果评估函数
def evaluate_separation(reference_bass, reference_soprano, estimated_bass, estimated_soprano):
    # 计算男低音的SDR, SIR, SAR
    sdr_bass, sir_bass, sar_bass, _ = mir_eval.separation.bss_eval_sources(reference_bass, estimated_bass)
    
    # 计算女高音的SDR, SIR, SAR
    sdr_soprano, sir_soprano, sar_soprano, _ = mir_eval.separation.bss_eval_sources(reference_soprano, estimated_soprano)
    
    return sdr_bass, sir_bass, sar_bass, sdr_soprano, sir_soprano, sar_soprano

def load_reference_audio(reference_bass_file, reference_soprano_file, sr=44100):
    reference_bass, _ = librosa.load(reference_bass_file, sr=sr)
    reference_soprano, _ = librosa.load(reference_soprano_file, sr=sr)
    return reference_bass, reference_soprano

def match_audio_length(estimated, reference):
    # 如果长度不一致，取最小的长度
    min_length = min(len(estimated), len(reference))
    # 裁剪为相同长度
    estimated = estimated[:min_length]
    reference = reference[:min_length]
    return estimated, reference

def evaluate_separation(reference_bass, reference_soprano, estimated_bass, estimated_soprano):
    # 对齐音频的长度
    estimated_bass, reference_bass = match_audio_length(estimated_bass, reference_bass)
    estimated_soprano, reference_soprano = match_audio_length(estimated_soprano, reference_soprano)
    
    # 计算男低音的SDR, SIR, SAR
    sdr_bass, sir_bass, sar_bass, _ = mir_eval.separation.bss_eval_sources(reference_bass, estimated_bass)
    
    # 计算女高音的SDR, SIR, SAR
    sdr_soprano, sir_soprano, sar_soprano, _ = mir_eval.separation.bss_eval_sources(reference_soprano, estimated_soprano)
    
    return sdr_bass, sir_bass, sar_bass, sdr_soprano, sir_soprano, sar_soprano

def check_audio_validity(audio):
    if np.all(audio == 0):
        print("警告: 音频信号全为零！")
    elif np.isnan(audio).any():
        print("警告: 音频信号中包含NaN值！")
    else:
        print("音频信号有效。")

# 在提取男低音和女高音后检查音频的有效性


def main():
    """
    # 输入音频文件路径
    input_file = 'C:/Users/mrm/Desktop/test/DT2470/ChoirSeparation-Transcription/combined-DYN.wav'
    
    # 参考音频文件路径
    reference_bass_file = 'C:/Users/mrm/Desktop/test/DT2470/ChoirSeparation-Transcription/DagstuhlChoirSet_V1.2.3/audio_wav_22050_mono/DCS_LI_FullChoir_Take01_B2_DYN.wav'
    reference_soprano_file = 'C:/Users/mrm/Desktop/test/DT2470/ChoirSeparation-Transcription/DagstuhlChoirSet_V1.2.3/audio_wav_22050_mono/DCS_LI_FullChoir_Take01_S1_DYN.wav'

    # 加载参考音频
    reference_bass, reference_soprano = load_reference_audio(reference_bass_file, reference_soprano_file)
    
    # 分离男低音和女高音
    bass_audio, soprano_audio, f0, bass_threshold, soprano_threshold = extract_f0_and_separate(input_file)
    
    # 打印动态阈值
    print(f"动态男低音阈值: {bass_threshold}, 动态女高音阈值: {soprano_threshold}")
    
    # 评估分离效果
    sdr_bass, sir_bass, sar_bass, sdr_soprano, sir_soprano, sar_soprano = evaluate_separation(reference_bass, reference_soprano, bass_audio, soprano_audio)
    
    check_audio_validity(bass_audio)
    check_audio_validity(soprano_audio)

    print(f"男低音: SDR={sdr_bass}, SIR={sir_bass}, SAR={sar_bass}")
    print(f"女高音: SDR={sdr_soprano}, SIR={sir_soprano}, SAR={sar_soprano}")
    
    # 保存结果
    save_audio(bass_audio, 'C:/Users/mrm/Desktop/test/DT2470/ChoirSeparation-Transcription/bass_output.wav')
    save_audio(soprano_audio, 'C:/Users/mrm/Desktop/test/DT2470/ChoirSeparation-Transcription/soprano_output.wav')
    print("男低音和女高音音频分离完成！")
    """
    
    input_file = 'C:/Users/mrm/Desktop/test/DT2470/ChoirSeparation-Transcription/combined-DYN.wav'
    
    # 分离男低音和女高音
    bass_audio, soprano_audio = extract_f0_and_separate(input_file)
    
    # 绘制频谱图
    plot_spectrogram(bass_audio, 44100, "bass")
    plot_spectrogram(soprano_audio, 44100, "soprano")
    plot_spectrogram(librosa.load(input_file, sr=44100)[0], 44100, "combined_audio")
    
    # 保存结果
    save_audio(bass_audio, 'C:/Users/mrm/Desktop/test/DT2470/ChoirSeparation-Transcription/bass_output.wav')
    save_audio(soprano_audio, 'C:/Users/mrm/Desktop/test/DT2470/ChoirSeparation-Transcription/soprano_output.wav')
    
    print("男低音和女高音音频分离完成！")

if __name__ == '__main__':
    main()


