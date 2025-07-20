import librosa
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import namedtuple

# === Parameter ===
MIN_NOTE_MS = 500          # minimal durasi note valid
GAP_BRIDGE_MS = 120        # gap kecil yang boleh dijembatani
SMOOTH_WINDOW = 5          # ukuran median smoothing (frame)
USE_PITCH_CLASS_ONLY = False  # False => Tampilkan octave (A3, A4, dll)

# === 1. Pilih file ===
Tk().withdraw()
audio_path = askopenfilename(title="Pilih file audio (.wav)", filetypes=[("WAV files", "*.wav")])
if not audio_path:
    print("Tidak ada file yang dipilih.")
    exit()

# === 2. Load audio ===
y, sr = librosa.load(audio_path, sr=None, mono=True)
duration_sec = len(y) / sr
print(f"🎧 Durasi Audio: {duration_sec:.2f} detik")

# === 3. Pitch detection (YIN) ===
hop_length = 512
frame_ms = librosa.frames_to_time(1, sr=sr, hop_length=hop_length) * 1000
f0 = librosa.yin(
    y,
    fmin=librosa.note_to_hz('E2'),
    fmax=librosa.note_to_hz('E6'),
    sr=sr,
    frame_length=2048,
    hop_length=hop_length,
)

# === 4. Smoothing f0 (median) ===
f0_smooth = f0.copy()
for i in range(len(f0)):
    start = max(0, i - SMOOTH_WINDOW // 2)
    end   = min(len(f0), i + SMOOTH_WINDOW // 2 + 1)
    window_vals = f0[start:end]
    window_vals = window_vals[~np.isnan(window_vals)]
    if len(window_vals) > 0:
        f0_smooth[i] = np.median(window_vals)

# === 5. Konversi ke note (dengan atau tanpa octave) ===
raw_notes = []
for f in f0_smooth:
    if np.isnan(f):
        raw_notes.append('N/A')
    else:
        note_full = librosa.midi_to_note(librosa.hz_to_midi(f))
        if USE_PITCH_CLASS_ONLY:
            pc = ''.join([c for c in note_full if not c.isdigit()])
            raw_notes.append(pc)
        else:
            raw_notes.append(note_full)  # lengkap dengan octave

# === 6. Bangun segmen awal ===
Segment = namedtuple('Segment', ['note', 'start_idx', 'end_idx'])
segments = []
if raw_notes:
    cur_note = raw_notes[0]
    seg_start = 0
    for i in range(1, len(raw_notes)+1):
        if i == len(raw_notes) or raw_notes[i] != cur_note:
            segments.append(Segment(cur_note, seg_start, i))
            if i < len(raw_notes):
                cur_note = raw_notes[i]
                seg_start = i

# === 7. Filter segmen < MIN_NOTE_MS atau 'N/A' ===
def seg_duration_ms(seg: Segment):
    return (seg.end_idx - seg.start_idx) * frame_ms

filtered = []
for seg in segments:
    if seg.note == 'N/A':
        continue
    if seg_duration_ms(seg) >= MIN_NOTE_MS:
        filtered.append(seg)

# === 8. Bridging gap kecil antar segmen note yang sama ===
bridged = []
i = 0
while i < len(filtered):
    current = filtered[i]
    j = i + 1
    while j < len(filtered):
        next_seg = filtered[j]
        if next_seg.note != current.note:
            break
        gap_ms = (next_seg.start_idx - current.end_idx) * frame_ms
        if gap_ms <= GAP_BRIDGE_MS:
            current = Segment(current.note, current.start_idx, next_seg.end_idx)
            j += 1
        else:
            break
    bridged.append(current)
    i = j

# === 9. Rekonstruksi frame-level note ===
final_notes = ['N/A'] * len(raw_notes)
for seg in bridged:
    for k in range(seg.start_idx, seg.end_idx):
        final_notes[k] = seg.note

# === 10. Plot scatter ===
times_ms = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length) * 1000
plot_f0 = [f if n != 'N/A' else np.nan for f, n in zip(f0_smooth, final_notes)]

plt.figure(figsize=(14, 8))
colors = [f'C{hash(n) % 10}' if n != 'N/A' else 'gray' for n in final_notes]
plt.scatter(times_ms, plot_f0, c=colors, s=10)

plt.title(
    f"Pitch Scatter (YIN) | Durasi: {duration_sec:.2f}s | "
    f"MinDur: {MIN_NOTE_MS} ms | GapBridge: {GAP_BRIDGE_MS} ms"
)
plt.xlabel("Waktu (ms)")
plt.ylabel("Frekuensi Fundamental F₀ (Hz)")
plt.grid(True)
plt.tight_layout()

# === 11. Legend berdasarkan urutan muncul ===
seen = set()
ordered = []
for n in final_notes:
    if n != 'N/A' and n not in seen:
        seen.add(n)
        ordered.append(n)
for n in ordered:
    plt.scatter([], [], color=f'C{hash(n)%10}', label=n)
plt.legend(title="Nada (urut muncul)")

# === 12. Timeline ringkas ===
timeline = []
last = None
for n in final_notes:
    if n != 'N/A' and n != last:
        timeline.append(n)
        last = n
timeline_str = " – ".join(timeline[:25])
plt.figtext(0.5, -0.05, f"Timeline: {timeline_str}", ha='center', fontsize=10, color='blue')

plt.show()