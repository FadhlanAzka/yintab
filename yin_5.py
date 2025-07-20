import librosa
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ================== PARAMETER ==================
FRAME_INTERVAL_MS = 500         # resolusi waktu sampling (ms)
SMOOTH_WINDOW = 5               # ukuran median window (frame YIN)
CENT_NOTE_LOCK = 12             # variasi < ini (cent) tetap dianggap nada yang sama
CENT_BOUNDARY_CORRECTION = 18   # jika label berubah tapi selisih < ini (cent), koreksi balik
SHOW_DEBUG = False              # True untuk print koreksi di terminal
# ===============================================

# --- 1. File picker ---
Tk().withdraw()
audio_path = askopenfilename(title="Pilih file audio (.wav)", filetypes=[("WAV files", "*.wav")])
if not audio_path:
    print("Tidak ada file yang dipilih.")
    raise SystemExit

# --- 2. Load audio ---
y, sr = librosa.load(audio_path, sr=None, mono=True)
duration_sec = len(y) / sr
print(f"🎧 Durasi Audio: {duration_sec:.2f} detik")

# --- 3. Pitch detection (YIN) ---
hop_length = 512
f0 = librosa.yin(
    y,
    fmin=librosa.note_to_hz('E2'),
    fmax=librosa.note_to_hz('E6'),
    sr=sr,
    frame_length=2048,
    hop_length=hop_length
)

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
times_frame = librosa.frames_to_time(np.arange(len(f0_smooth)), sr=sr, hop_length=hop_length)
block_len_s = FRAME_INTERVAL_MS / 1000.0
sample_block_starts = np.arange(0, duration_sec, block_len_s)

sample_times = []
sample_f0 = []
sample_midi_exact = []
sample_note_raw = []

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
    note_name = librosa.midi_to_note(midi_exact)  # internal: akan di-round
    sample_times.append(t0)
    sample_f0.append(median_f)
    sample_midi_exact.append(midi_exact)
    sample_note_raw.append(note_name)

sample_times = np.array(sample_times)
sample_f0 = np.array(sample_f0)
sample_midi_exact = np.array(sample_midi_exact)
sample_note_raw = np.array(sample_note_raw, dtype=object)

# --- 6. Stabilization & cent-based correction ---
stable_notes = []
stable_f0 = []
stable_midi = []

prev_note = None
prev_midi = None

for t, f_hz, midi_exact, note_label in zip(sample_times, sample_f0, sample_midi_exact, sample_note_raw):
    if np.isnan(f_hz) or note_label == 'N/A':
        # tidak menambah segmen; tetap kosong
        stable_notes.append('N/A')
        stable_f0.append(np.nan)
        stable_midi.append(np.nan)
        continue

    # Jika belum ada note sebelumnya -> inisialisasi
    if prev_note is None:
        stable_notes.append(note_label)
        stable_f0.append(f_hz)
        stable_midi.append(midi_exact)
        prev_note = note_label
        prev_midi = midi_exact
        continue

    # Hitung selisih dalam cent relatif terhadap note sebelumnya
    diff_cents = (midi_exact - prev_midi) * 100.0  # 1 semitone = 100 cent

    if note_label == prev_note:
        # Sama persis label -> selalu lanjut (garis)
        stable_notes.append(note_label)
        stable_f0.append(f_hz)
        stable_midi.append(midi_exact)
        prev_midi = midi_exact  # update pusat
    else:
        # Label beda. Cek apakah mendekati pitch sebelumnya (cent)
        if abs(diff_cents) <= CENT_BOUNDARY_CORRECTION:
            # Koreksi balik ke prev_note
            if SHOW_DEBUG:
                print(f"[CORRECT] {note_label} dikoreksi -> {prev_note} (Δ {diff_cents:.1f} cent)")
            stable_notes.append(prev_note)
            stable_f0.append(f_hz)
            # Tetap update prev_midi perlahan (low-pass) agar tidak 'ngambang'
            weight = 0.3
            prev_midi = (1 - weight) * prev_midi + weight * midi_exact
            stable_midi.append(prev_midi)
        else:
            # Perubahan nyata -> note baru (mulai dot)
            stable_notes.append(note_label)
            stable_f0.append(f_hz)
            stable_midi.append(midi_exact)
            prev_note = note_label
            prev_midi = midi_exact

stable_notes = np.array(stable_notes, dtype=object)
stable_f0 = np.array(stable_f0)

# --- 7. Plot: line untuk note berlanjut, dot untuk transisi ---
plt.figure(figsize=(14, 6))
legend_order = []

for i in range(len(stable_notes)):
    note_now = stable_notes[i]
    if note_now == 'N/A' or np.isnan(stable_f0[i]):
        continue

    color = f'C{hash(note_now) % 10}'
    if note_now not in legend_order:
        legend_order.append(note_now)

    if i > 0 and stable_notes[i] == stable_notes[i - 1]:
        # lanjut (line) meskipun f0 bergeser kecil
        plt.plot(
            [sample_times[i - 1] * 1000, sample_times[i] * 1000],
            [stable_f0[i - 1], stable_f0[i]],
            color=color,
            linewidth=2
        )
    else:
        # note baru -> dot
        plt.scatter(sample_times[i] * 1000, stable_f0[i], c=color, s=40)

# --- 8. Legend urut kemunculan ---
for n in legend_order:
    plt.plot([], [], color=f'C{hash(n)%10}', label=n)
plt.legend(title="Nada (urut kemunculan)")

plt.title(
    f"Pitch Timeline (YIN, {FRAME_INTERVAL_MS} ms) | Durasi: {duration_sec:.2f}s\n"
    f"Stabil: lock≤{CENT_NOTE_LOCK}c, boundary≤{CENT_BOUNDARY_CORRECTION}c"
)
plt.xlabel("Waktu (ms)")
plt.ylabel("Frekuensi Fundamental F₀ (Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()