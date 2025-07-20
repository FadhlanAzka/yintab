import librosa
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import Counter

# === 1. Pilih file audio ===
Tk().withdraw()
audio_path = askopenfilename(title="Pilih file audio (.wav)", filetypes=[("WAV files", "*.wav")])
if not audio_path:
    print("Tidak ada file yang dipilih.")
    exit()

# === 2. Load audio ===
y, sr = librosa.load(audio_path, sr=None, mono=True)
duration_sec = len(y) / sr
print(f"\n🎧 Durasi Audio: {duration_sec:.2f} detik")

# === 3. Estimasi BPM ===
tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
tempo = tempo_array.item()
print(f"🎵 Estimasi BPM: {tempo:.2f}")

# === 4. Estimasi key/tonalitas menggunakan chroma
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
chroma_mean = chroma.mean(axis=1)
pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
key_index = np.argmax(chroma_mean)
key_estimated = pitch_classes[key_index]
print(f"🎼 Estimasi Nada Dasar: {key_estimated} (root note)\n")

# === 5. Deteksi pitch dengan YIN ===
hop_length = 512
f0 = librosa.yin(y, 
                 fmin=librosa.note_to_hz('E2'), 
                 fmax=librosa.note_to_hz('E6'), 
                 sr=sr, 
                 frame_length=2048, 
                 hop_length=hop_length)

# === 6. Hitung waktu per frame & konversi ke note ===
times_ms = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length) * 1000
notes = [librosa.midi_to_note(librosa.hz_to_midi(f)) if not np.isnan(f) else 'N/A' for f in f0]

# === 6.1: Hitung RMS (volume/energi per frame)
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

# === 6.2: Tentukan ambang batas volume
volume_threshold = np.percentile(rms, 75)  # top 25% paling keras

# === 7. Filtering: buang pitch yang hanya muncul < threshold
threshold_frame = 5  # pitch muncul minimal 5 frame (~0.12s kalau hop 512)
note_counts = Counter(notes)
valid_notes = {note for note, count in note_counts.items() if count >= threshold_frame}
filtered_f0 = []
filtered_times = []
filtered_notes = []

for i in range(len(f0)):
    if (notes[i] in valid_notes and notes[i] != 'N/A' and rms[i] >= volume_threshold):
        filtered_f0.append(f0[i])
        filtered_times.append(times_ms[i])
        filtered_notes.append(notes[i])

# === 8. Plot scatter
plt.figure(figsize=(14, 6))
colors = [f'C{hash(n)%10}' for n in filtered_notes]
plt.scatter(filtered_times, filtered_f0, c=colors, s=12, label='Pitch')

plt.title(f"Pitch Scatter (YIN) | Durasi: {duration_sec:.2f}s | BPM: {tempo:.1f} | Key: {key_estimated}")
plt.xlabel("Waktu (ms)")
plt.ylabel("Frekuensi F₀ (Hz)")
plt.grid(True)
plt.tight_layout()

# === 9. Legenda manual per nada ===
unique_notes = sorted(set(filtered_notes))
for note in unique_notes:
    plt.scatter([], [], c=f'C{hash(note)%10}', label=note)
plt.legend(title="Nada Terdeteksi (Valid)")

plt.show()