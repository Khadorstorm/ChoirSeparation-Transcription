import numpy as np
import soundfile as sf
import pyworld as pw
import librosa
import os
from scipy.ndimage import filters
import matplotlib.pyplot as plt

# STFT计算函数
def stft(data, window=np.hanning(1024), hopsize=256.0, nfft=1024.0, fs=44100.0):
    lengthWindow = window.size
    lengthData = data.size
    numberFrames = np.ceil(lengthData / np.double(hopsize)) + 2
    newLengthData = (numberFrames-1) * hopsize + lengthWindow
    data = np.concatenate((np.zeros(int(lengthWindow/2)), data))
    data = np.concatenate((data, np.zeros(int(newLengthData - data.size))))
    numberFrequencies = nfft / 2 + 1
    STFT = np.zeros([int(numberFrames), int(numberFrequencies)], dtype=complex)
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * data[int(beginFrame):int(endFrame)]
        STFT[int(n), :] = np.fft.rfft(frameToProcess, np.int32(nfft), norm="ortho")
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
        frameTMP = np.fft.irfft(X[:, n], np.int32(nfft), norm="ortho")
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = normalisationSeq[beginFrame:endFrame] + window
        data[beginFrame:endFrame] = data[beginFrame:endFrame] + window * frameTMP
    data = data[int(lengthWindow / 2.0):]
    normalisationSeq = normalisationSeq[int(lengthWindow / 2.0):]
    normalisationSeq[normalisationSeq == 0] = 1.
    data = data / normalisationSeq
    return data

# F0提取和音频分离
def extract_f0_and_separate(audio_file, fs=44100, gender_threshold=200):
    audio, sr = librosa.load(audio_file, sr=fs)
    
    audio = audio.astype(np.float64)
    # 提取F0（基频）
    f0, t = pw.dio(audio, sr)    # 基于DIO算法提取F0
    f0 = pw.stonemask(audio, f0, t, sr)    # 基于STONE MASK优化F0
    sp = pw.cheaptrick(audio, f0, t, sr)   # 计算频谱包络
    ap = pw.d4c(audio, f0, t, sr)          # 计算非周期成分
    
    # 男低音和女高音分离
    bass_f0 = np.where(f0 < gender_threshold, f0, 0)   # 男低音F0
    soprano_f0 = np.where(f0 >= 250, f0, 0) # 女高音F0
    
    # 重构音频
    bass_audio = pw.synthesize(bass_f0, sp, ap, sr)
    soprano_audio = pw.synthesize(soprano_f0, sp, ap, sr)
    
    return bass_audio, soprano_audio

# 保存分离的音频
# 保存音频的函数，确保目录存在
def save_audio(audio, filename, fs=44100):
    # 获取目录路径
    directory = os.path.dirname(filename)
    
    # 如果目录不存在，则创建它
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存音频
    sf.write(filename, audio, fs)

# 主函数
def main():
    # 输入音频文件路径
    input_file = 'C:/Users/mrm/Desktop/test/DT2470/WGANSing-mtg/combined-DYN.wav'
    
    # 分离男低音和女高音
    bass_audio, soprano_audio = extract_f0_and_separate(input_file)
    
    # 保存结果
    save_audio(bass_audio, './bass_output.wav')
    save_audio(soprano_audio, './soprano_output.wav')
    print("男低音和女高音音频分离完成！")

if __name__ == '__main__':
    main()