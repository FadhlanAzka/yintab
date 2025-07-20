import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import time
import scipy.signal as signal
from tkinter import Tk, filedialog, messagebox

# ================== PARAMETER ==================
FRAME_INTERVAL_MS = 50          # resolusi waktu sampling (ms)
SMOOTH_WINDOW = 5               # ukuran median window (frame YIN)
CENT_NOTE_LOCK = 12             # variasi < ini (cent) tetap dianggap nada yang sama
CENT_BOUNDARY_CORRECTION = 18   # jika label berubah tapi selisih < ini (cent), koreksi balik
SHOW_DEBUG = False              # True untuk print koreksi di terminal
HOP_LENGTH = 256                # resolusi frame (lebih kecil)

# --- Noise Filtering Parameter ---
RMS_NOISE_PERCENTILE = 20       # frame di bawah persentil ini dianggap noise/senyap
# ===============================================

# --- Band-pass Filter Function ---
def bandpass_filter(data, sr, low_hz=80, high_hz=2000):
    sos = signal.butter(10, [low_hz, high_hz], btype='bandpass', fs=sr, output='sos')
    return signal.sosfilt(sos, data)

# --- 0. UI: Pilih mode (vocal/guitar) ---
root = Tk()
root.withdraw()
mode_choice = messagebox.askquestion("Pilih Mode", "Apakah ingin mendeteksi **vokal**?\n\nKlik 'Yes' untuk Vokal, 'No' untuk Gitar.")
SOURCE_MODE = 'vocal' if mode_choice == 'yes' else 'guitar'
print(f"🔍 Mode Deteksi: {SOURCE_MODE.upper()}")

# --- 1. File picker ---
audio_path = filedialog.askopenfilename(title="Pilih file audio (.wav)", filetypes=[("WAV files", "*.wav")])
if not audio_path:
    print("Tidak ada file yang dipilih.")
    raise SystemExit

# --- 2. Load audio ---
y, sr = librosa.load(audio_path, sr=None, mono=True)
duration_sec = len(y) / sr
print(f"🎧 Durasi Audio: {duration_sec:.2f} detik")

# --- 2b. Source filtering ---
y_harm, y_perc = librosa.effects.hpss(y)
if SOURCE_MODE == 'vocal':
    y = bandpass_filter(y_harm, sr, 80, 1200)
else:
    y = bandpass_filter(y_harm, sr, 80, 2000)

# --- 3. Pitch detection (YIN) ---
f0 = librosa.yin(
    y,
    fmin=librosa.note_to_hz('E2'),
    fmax=librosa.note_to_hz('E6'),
    sr=sr,
    frame_length=2048,
    hop_length=HOP_LENGTH
)

# --- 3b. Noise Filtering (range + energy) ---
valid = ~np.isnan(f0)
f0[(valid) & ((f0 < 50) | (f0 > 2000))] = np.nan
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP_LENGTH)[0]
rms = librosa.util.fix_length(rms, size=len(f0))
rms_threshold = np.percentile(rms, RMS_NOISE_PERCENTILE)
f0[rms < rms_threshold] = np.nan

# --- 4. Median smoothing (frame-level) ---
f0_smooth = f0.copy()
for i in range(len(f0)):
    start = max(0, i - SMOOTH_WINDOW // 2)
    end = min(len(f0), i + SMOOTH_WINDOW // 2 + 1)
    win = f0[start:end]
    win = win[~np.isnan(win)]
    if len(win) > 0:
        f0_smooth[i] = np.median(win)

# --- 5. Sampling tiap FRAME_INTERVAL_MS ---
times_frame = librosa.frames_to_time(np.arange(len(f0_smooth)), sr=sr, hop_length=HOP_LENGTH)
block_len_s = FRAME_INTERVAL_MS / 1000.0
sample_block_starts = np.arange(0, duration_sec, block_len_s)

sample_times, sample_f0, sample_midi_exact, sample_note_raw = [], [], [], []
for t0 in sample_block_starts:
    t1 = t0 + block_len_s
    idx = np.where((times_frame >= t0) & (times_frame < t1))[0]
    if len(idx) == 0:
        sample_times.append(t0)
        sample_f0.append(np.nan)
        sample_midi_exact.append(np.nan)
        sample_note_raw.append('N/A')
        continue
    block_vals = f0_smooth[idx]
    block_vals = block_vals[~np.isnan(block_vals)]
    if len(block_vals) == 0:
        sample_times.append(t0)
        sample_f0.append(np.nan)
        sample_midi_exact.append(np.nan)
        sample_note_raw.append('N/A')
        continue
    median_f = np.median(block_vals)
    midi_exact = librosa.hz_to_midi(median_f)
    note_name = librosa.midi_to_note(midi_exact)
    sample_times.append(t0)
    sample_f0.append(median_f)
    sample_midi_exact.append(midi_exact)
    sample_note_raw.append(note_name)

sample_times = np.array(sample_times)
sample_f0 = np.array(sample_f0)
sample_midi_exact = np.array(sample_midi_exact)
sample_note_raw = np.array(sample_note_raw, dtype=object)

# --- 6. Stabilization & cent-based correction ---
stable_notes, stable_f0, stable_midi = [], [], []
prev_note, prev_midi = None, None
for t, f_hz, midi_exact, note_label in zip(sample_times, sample_f0, sample_midi_exact, sample_note_raw):
    if np.isnan(f_hz) or note_label == 'N/A':
        stable_notes.append('N/A')
        stable_f0.append(np.nan)
        stable_midi.append(np.nan)
        continue
    if prev_note is None:
        stable_notes.append(note_label)
        stable_f0.append(f_hz)
        stable_midi.append(midi_exact)
        prev_note, prev_midi = note_label, midi_exact
        continue
    diff_cents = (midi_exact - prev_midi) * 100.0
    if note_label == prev_note:
        stable_notes.append(note_label)
        stable_f0.append(f_hz)
        stable_midi.append(midi_exact)
        prev_midi = midi_exact
    else:
        if abs(diff_cents) <= CENT_BOUNDARY_CORRECTION:
            if SHOW_DEBUG:
                print(f"[CORRECT] {note_label} -> {prev_note} (Δ {diff_cents:.1f} cent)")
            stable_notes.append(prev_note)
            stable_f0.append(f_hz)
            weight = 0.3
            prev_midi = (1 - weight) * prev_midi + weight * midi_exact
            stable_midi.append(prev_midi)
        else:
            stable_notes.append(note_label)
            stable_f0.append(f_hz)
            stable_midi.append(midi_exact)
            prev_note, prev_midi = note_label, midi_exact

stable_notes = np.array(stable_notes, dtype=object)
stable_f0 = np.array(stable_f0)

# --- 7. Onset Detection ---
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
print("Onsets terdeteksi pada (detik):", onset_times)

# --- 8. Plot Setup ---
fig, ax = plt.subplots(figsize=(14, 6))
legend_order = []
for i in range(len(stable_notes)):
    note_now = stable_notes[i]
    if note_now == 'N/A' or np.isnan(stable_f0[i]):
        continue
    color = f'C{hash(note_now) % 10}'
    if note_now not in legend_order:
        legend_order.append(note_now)
    if i > 0 and stable_notes[i] == stable_notes[i - 1]:
        ax.plot(
            [sample_times[i - 1] * 1000, sample_times[i] * 1000],
            [stable_f0[i - 1], stable_f0[i]],
            color=color, linewidth=2
        )
    else:
        ax.scatter(sample_times[i] * 1000, stable_f0[i], c=color, s=40)

for onset in onset_times:
    ax.axvline(onset * 1000, color='red', linestyle='--', alpha=0.5, linewidth=1)
for n in legend_order:
    ax.plot([], [], color=f'C{hash(n)%10}', label=n)
ax.legend(title="Nada (urut kemunculan)")
ax.set_title(
    f"Pitch Timeline (YIN, {FRAME_INTERVAL_MS} ms) + Onsets | Durasi: {duration_sec:.2f}s\n"
    f"Mode: {SOURCE_MODE.upper()} | Stabil: lock≤{CENT_NOTE_LOCK}c, boundary≤{CENT_BOUNDARY_CORRECTION}c"
)
ax.set_xlabel("Waktu (ms)")
ax.set_ylabel("Frekuensi Fundamental F₀ (Hz)")
ax.grid(True)

# --- 9. Real-time Pointer + Audio Playback ---
pointer_line = ax.axvline(0, color='black', linewidth=2)
start_time = [None]

def update_pointer(frame):
    if start_time[0] is None:
        start_time[0] = time.time()
    elapsed = time.time() - start_time[0]
    pointer_line.set_xdata(elapsed * 1000)  # ms
    return pointer_line,

ani = animation.FuncAnimation(fig, update_pointer, interval=20, blit=True)

# Mulai mainkan audio
sd.play(y, sr)

plt.tight_layout()
plt.show()

# Hentikan audio jika plot ditutup
sd.stop()
