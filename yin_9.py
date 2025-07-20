import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import sounddevice as sd
import time
import scipy.signal as signal
from tkinter import Tk, Toplevel, Button as TkButton, Label, filedialog

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


# ------------------------------------------------
# UI: dialog pemilihan mode (Vokal / Gitar)
# ------------------------------------------------
def select_source_mode():
    root = Tk()
    root.withdraw()  # sembunyikan root utama

    mode_holder = {'mode': None}

    def choose_vokal():
        mode_holder['mode'] = 'vocal'
        sel.destroy()

    def choose_gitar():
        mode_holder['mode'] = 'guitar'
        sel.destroy()

    sel = Toplevel(root)
    sel.title("Pilih Deteksi")
    Label(sel, text="Pilih Deteksi", font=("Helvetica", 12, "bold")).pack(padx=20, pady=(20, 10))
    TkButton(sel, text="Vokal", width=15, command=choose_vokal).pack(pady=5)
    TkButton(sel, text="Gitar", width=15, command=choose_gitar).pack(pady=(0, 20))

    # buat modal-ish
    sel.grab_set()
    root.wait_window(sel)
    root.destroy()

    if mode_holder['mode'] is None:
        print("Tidak ada mode dipilih. Keluar.")
        raise SystemExit
    return mode_holder['mode']


# ------------------------------------------------
# Band-pass Filter Function
# ------------------------------------------------
def bandpass_filter(data, sr, low_hz=80, high_hz=2000, order=10):
    sos = signal.butter(order, [low_hz, high_hz], btype='bandpass', fs=sr, output='sos')
    return signal.sosfilt(sos, data)


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":
    SOURCE_MODE = select_source_mode()
    print(f"🔍 Mode Deteksi: {SOURCE_MODE.upper()}")

    # --- File picker ---
    audio_path = filedialog.askopenfilename(
        title="Pilih file audio (.wav)",
        filetypes=[("WAV files", "*.wav")]
    )
    if not audio_path:
        print("Tidak ada file yang dipilih.")
        raise SystemExit

    # --- Load audio (mono) ---
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration_sec = len(y) / sr
    print(f"🎧 Durasi Audio: {duration_sec:.2f} detik")

    # --- Source filtering (HPSS + Bandpass) ---
    y_harm, y_perc = librosa.effects.hpss(y)
    if SOURCE_MODE == 'vocal':
        y_proc = bandpass_filter(y_harm, sr, 80, 1200)
        fmin_hz, fmax_hz = librosa.note_to_hz('C2'), librosa.note_to_hz('C6')
    else:  # guitar
        y_proc = bandpass_filter(y_harm, sr, 80, 2000)
        fmin_hz, fmax_hz = librosa.note_to_hz('E2'), librosa.note_to_hz('E6')

    # --- Pitch detection (YIN) ---
    f0 = librosa.yin(
        y_proc,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        frame_length=2048,
        hop_length=HOP_LENGTH
    )

    # --- Noise Filtering (range + energy) ---
    valid = ~np.isnan(f0)
    f0[(valid) & ((f0 < 50) | (f0 > 2000))] = np.nan  # jaga range umum

    rms = librosa.feature.rms(y=y_proc, frame_length=2048, hop_length=HOP_LENGTH)[0]
    rms = librosa.util.fix_length(rms, size=len(f0))
    rms_threshold = np.percentile(rms, RMS_NOISE_PERCENTILE)
    f0[rms < rms_threshold] = np.nan

    if SHOW_DEBUG:
        print(f"[NOISE] RMS threshold @ p{RMS_NOISE_PERCENTILE}: {rms_threshold:.6f}")

    # --- Median smoothing (frame-level) ---
    f0_smooth = f0.copy()
    for i in range(len(f0)):
        start = max(0, i - SMOOTH_WINDOW // 2)
        end = min(len(f0), i + SMOOTH_WINDOW // 2 + 1)
        win = f0[start:end]
        win = win[~np.isnan(win)]
        if len(win) > 0:
            f0_smooth[i] = np.median(win)

    # --- Sampling tiap FRAME_INTERVAL_MS ---
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

    # --- Stabilization & cent-based correction ---
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

    # --- Onset Detection ---
    onset_frames = librosa.onset.onset_detect(y=y_proc, sr=sr, hop_length=HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    print("Onsets terdeteksi pada (detik):", onset_times)

    # ------------------------------------------------
    # Plot Setup
    # ------------------------------------------------
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

    # pointer line (vertical)
    pointer_line = ax.axvline(0, color='black', linewidth=2)

    # ------------------------------------------------
    # Playback State Machine
    # ------------------------------------------------
    play_state = {
        'started': False,
        'is_paused': False,
        'play_start_global_sec': 0.0,  # posisi awal playback relatif ke file
        'start_time_perf': None,       # perf_counter saat playback terakhir dimulai
        'pause_pos_sec': 0.0,          # posisi saat pause
        'sr': sr,
        'audio': y_proc,               # audio yang diputar
        'duration': duration_sec,
    }

    def current_play_pos_sec():
        """Return posisi playback (detik) relatif ke awal file."""
        if play_state['is_paused']:
            return play_state['pause_pos_sec']
        if not play_state['started'] or play_state['start_time_perf'] is None:
            return 0.0
        elapsed_local = time.perf_counter() - play_state['start_time_perf']
        return play_state['play_start_global_sec'] + elapsed_local

    def start_play_from(pos_sec=0.0):
        """Mulai playback dari posisi tertentu."""
        if pos_sec < 0:
            pos_sec = 0
        if pos_sec >= play_state['duration']:
            return
        start_idx = int(pos_sec * play_state['sr'])
        sd.stop()
        sd.play(play_state['audio'][start_idx:], play_state['sr'])
        play_state['play_start_global_sec'] = pos_sec
        play_state['start_time_perf'] = time.perf_counter()
        play_state['is_paused'] = False
        play_state['started'] = True

    def pause_playback(event=None):
        if not play_state['started'] or play_state['is_paused']:
            return
        # capture current pos
        pos = current_play_pos_sec()
        play_state['pause_pos_sec'] = min(pos, play_state['duration'])
        sd.stop()
        play_state['is_paused'] = True
        print(f"⏸️ Pause @ {play_state['pause_pos_sec']:.3f}s")

    def resume_playback(event=None):
        if not play_state['is_paused']:
            return
        print(f"▶️ Resume @ {play_state['pause_pos_sec']:.3f}s")
        start_play_from(play_state['pause_pos_sec'])

    # ------------------------------------------------
    # Matplotlib Buttons: Pause / Resume
    # ------------------------------------------------
    # Add small axes above main plot (normalized figure coords)
    axpause = fig.add_axes([0.80, 0.92, 0.08, 0.05])
    axresume = fig.add_axes([0.89, 0.92, 0.08, 0.05])
    btn_pause = Button(axpause, 'Pause')
    btn_resume = Button(axresume, 'Resume')
    btn_pause.on_clicked(pause_playback)
    btn_resume.on_clicked(resume_playback)

    # ------------------------------------------------
    # Animation update (sync pointer)
    # ------------------------------------------------
    def update_pointer(_frame):
        # start playback at first animation call
        if not play_state['started'] and not play_state['is_paused']:
            start_play_from(0.0)

        pos_sec = current_play_pos_sec()
        if pos_sec >= play_state['duration']:
            # clamp at end & stop audio
            pos_sec = play_state['duration']
            sd.stop()
            play_state['is_paused'] = True  # freeze pointer at end

        x_ms = pos_sec * 1000.0
        # update pointer line xdata as 2 points (avoid deprecation)
        pointer_line.set_xdata([x_ms, x_ms])
        return (pointer_line,)

    # Use cache_frame_data=False to suppress warning
    ani = animation.FuncAnimation(
        fig,
        update_pointer,
        interval=20,
        blit=True,
        cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()

    # stop audio when window closed
    sd.stop()
