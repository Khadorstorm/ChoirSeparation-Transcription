import crepe
import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_f0(audio_path):
    # 读取音频
    audio, sr = librosa.load(audio_path, sr=16000)
    # 使用 CREPE 进行基频预测
    time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=True)
    return time, frequency, confidence

def hz_to_note_name(frequency):
    """将频率转换为简谱音符，返回音符名称。"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # MIDI 69 是 A4，对应于 440 Hz
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    note_index = int(round(midi_note)) % 12
    octave = int((round(midi_note) // 12) - 1)
    return f"{note_names[note_index]}{octave}"

def save_notes_to_txt(notes_with_time, output_path):
    with open(output_path, "w") as file:
        for note, time in notes_with_time:
            file.write(f"{time:.2f}s: {note}\n")
    print(f"简谱已保存到 {output_path}")

def plot_pitch_curve(time, frequency, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(time, frequency, label="基频 (Hz)", color='b', alpha=0.7)
    plt.xlabel("time(s)")
    plt.ylabel("fluency(Hz)")
    plt.title("curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"音阶曲线图已保存到 {output_path}")

if __name__ == "__main__":
    audio_path = "flute-c-major.wav"
    output_txt_path = "simplified_notes.txt"
    output_plot_path = "pitch_curve.png"

    # 提取基频
    time, frequency, confidence = extract_f0(audio_path)

    # 转换频率到音符名称并过滤置信度低的值
    notes_with_time = [
        (hz_to_note_name(f), t) for t, f, c in zip(time, frequency, confidence) if c > 0.8 and f > 0
    ]

    # 保存音符到文本文件
    save_notes_to_txt(notes_with_time, output_txt_path)

    # 绘制基频变化曲线
    plot_pitch_curve(time, frequency, output_plot_path)
