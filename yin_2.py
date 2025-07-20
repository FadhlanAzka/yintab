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

# === 3. Pitch detection ===
hop_length = 512
f0 = librosa.yin(
    y,
    fmin=librosa.note_to_hz('E2'),
    fmax=librosa.note_to_hz('E6'),
    sr=sr,
    frame_length=2048,
    hop_length=hop_length,
)

# === 4. Time axis & note mapping ===
times_ms = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length) * 1000
notes = [librosa.midi_to_note(librosa.hz_to_midi(f)) if not np.isnan(f) else 'N/A' for f in f0]

# === 4.5. Cari frekuensi & nada yang paling sering muncul ===
valid_f0 = [f for f in f0 if not np.isnan(f)]
if valid_f0:
    # Round ke 2 desimal (opsional untuk menghindari noise)
    rounded_f0 = [round(f, 1) for f in valid_f0]
    f0_count = Counter(rounded_f0)
    most_common_f0, freq_count = f0_count.most_common(1)[0]

    # Konversi ke notasi
    most_common_note = librosa.midi_to_note(librosa.hz_to_midi(most_common_f0))

    print(f"🎯 Frekuensi F₀ paling dominan: {most_common_f0} Hz")
    print(f"🎶 Nada paling dominan: {most_common_note} (muncul {freq_count} kali)")
else:
    print("Tidak ada frekuensi valid untuk dianalisis.")

# === 5. Plot with color segmentation ===
plt.figure(figsize=(14, 6))
last_note = None
for i in range(1, len(f0)):
    f_prev, f_curr = f0[i - 1], f0[i]
    t_prev, t_curr = times_ms[i - 1], times_ms[i]

    if np.isnan(f_prev) or np.isnan(f_curr):
        continue

    # Jika nada berubah, ubah warna
    note_now = notes[i]
    color = f'C{hash(note_now) % 10}'  # hash warna per note stabil

    plt.plot([t_prev, t_curr], [f_prev, f_curr], color=color, linewidth=2)

# === 6. Merge dominant text into the title ===
dominant_text = f" | F₀ Dominan: {most_common_f0} Hz, Nada Dominan: {most_common_note} ({freq_count}x)"
plt.title(f"Pitch Contour (YIN) | Durasi Audio: {duration_sec:.2f} detik{dominant_text}")
plt.xlabel("Waktu (ms)")
plt.ylabel("Frekuensi Fundamental F₀ (Hz)")
plt.grid(True)
plt.tight_layout()

# === 7. Tambahkan legend manual untuk note unik ===
unique_notes = sorted(set([n for n in notes if n != 'N/A']))
for i, note in enumerate(unique_notes):
    plt.plot([], [], color=f'C{hash(note) % 10}', label=note)
plt.legend(title="Nada")

plt.show()