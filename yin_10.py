import argparse
import os
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import sounddevice as sd
import scipy.signal as signal
from tkinter import Tk, filedialog, Toplevel, Label, Button as TkButton

"""
solo_full_monophonic.py
---------------------------------
Tujuan:
  * Deteksi pitch dominan (monophonic) dari sinyal audio uji (misal solo gitar / vokal tunggal).
  * Pipeline: Load -> (opsional HPSS) -> Bandpass -> YIN -> Noise Gate -> Smoothing -> Enforce Monophonic -> Sampling -> Plot.
  * Opsi: mode vokal atau gitar (range frekuensi + bandpass berbeda).
  * Interaktif: tombol Pause / Resume playback; pointer vertikal sinkron audio.
  * Ekspor CSV hasil (time_s, f0_hz, midi_float, midi_int, note).

Catatan:
  - Script dapat dipanggil via CLI atau dijalankan langsung (akan munculkan dialog file).
  - File default (--audio) bisa diisi path ke /mnt/data/solo_full.wav jika ingin auto-run.
  - YIN tetap digunakan sebagai inti pitch detection.
"""

# =============================================================
# --------------------- PARAMETER GLOBAL ---------------------
# =============================================================
FRAME_INTERVAL_MS = 50          # resolusi waktu sampling output (ms)
SMOOTH_WINDOW = 5               # median smoothing di domain frame YIN
CENT_NOTE_LOCK = 12             # < ini (cent) tetap sama
CENT_BOUNDARY_CORRECTION = 18   # koreksi balik jika loncatan kecil
GAP_FILL_MS = 100               # isi gap pendek (np.nan) dengan nada sebelumnya (ms)
MIN_NOTE_MS = 60                # durasi minimum agar nada baru dianggap sah (ms)
SHOW_DEBUG = False
HOP_LENGTH = 256                # hop untuk STFT/YIN
RMS_NOISE_PERCENTILE = 20       # gating energi rendah
EXPORT_CSV = True               # simpan hasil csv
PLAY_AUDIO = True               # mainkan audio + pointer
PLOT_ONSETS = True              # tampilkan onset
AUDIO_LATENCY_COMP_MS = 0       # kompensasi pointer vs audio (ms)


# =============================================================
# ----------------------- UTIL FUNGSION -----------------------
# =============================================================

def ui_select_mode():
    """Modal Tkinter popup: pilih deteksi Vokal / Gitar. Return 'vocal' atau 'guitar'."""
    root = Tk(); root.withdraw()
    holder = {'mode': None}

    def _v():
        holder['mode'] = 'vocal'; win.destroy()
    def _g():
        holder['mode'] = 'guitar'; win.destroy()

    win = Toplevel(root)
    win.title("Pilih Deteksi")
    Label(win, text="Pilih Deteksi", font=("Helvetica", 12, "bold")).pack(padx=20, pady=(20, 10))
    TkButton(win, text="Vokal", width=15, command=_v).pack(pady=5)
    TkButton(win, text="Gitar", width=15, command=_g).pack(pady=(0, 20))
    win.grab_set(); root.wait_window(win); root.destroy()

    if holder['mode'] is None:
        raise SystemExit("Tidak ada mode dipilih.")
    return holder['mode']


def ui_select_file():
    """Dialog pilih file wav."""
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(title="Pilih file audio (.wav)", filetypes=[("WAV files", "*.wav")])
    root.destroy()
    return path


def bandpass_filter(data, sr, low_hz=80, high_hz=2000, order=10):
    sos = signal.butter(order, [low_hz, high_hz], btype='bandpass', fs=sr, output='sos')
    return signal.sosfilt(sos, data)


def noise_gate_f0(f0, y_proc, sr, rms_percentile=20, f0_min=50, f0_max=2000, hop_length=HOP_LENGTH):
    """Apply simple range + RMS energy gating to f0 array."""
    f0 = f0.copy()
    valid = ~np.isnan(f0)
    f0[(valid) & ((f0 < f0_min) | (f0 > f0_max))] = np.nan

    rms = librosa.feature.rms(y=y_proc, frame_length=2048, hop_length=hop_length)[0]
    rms = librosa.util.fix_length(rms, size=len(f0))
    thr = np.percentile(rms, rms_percentile)
    f0[rms < thr] = np.nan
    return f0, thr


def median_smooth_nan(arr, win):
    if win <= 1: return arr
    out = arr.copy()
    n = len(arr)
    for i in range(n):
        s = max(0, i - win // 2); e = min(n, i + win // 2 + 1)
        ww = arr[s:e]
        ww = ww[~np.isnan(ww)]
        if len(ww) > 0:
            out[i] = np.median(ww)
    return out


def enforce_monophonic(f0, sr, hop_length, cent_lock=CENT_NOTE_LOCK, cent_boundary=CENT_BOUNDARY_CORRECTION,
                        gap_fill_ms=GAP_FILL_MS, min_note_ms=MIN_NOTE_MS, show_debug=False):
    """
    Enforce monophonic continuity pada vektor f0 (Hz) per frame YIN.
    - Gap pendek (<=gap_fill_ms) diisi dengan nilai sebelumnya.
    - Lompatan kecil (<=cent_boundary) dikoreksi balik.
    - Nada baru harus bertahan >=min_note_ms; jika tidak, revert ke nada lama.
    Return f0_stable.
    """
    f0 = f0.copy()
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    frame_dur = times[1] - times[0] if len(times) > 1 else hop_length / sr

    gap_fill_frames = int(round(gap_fill_ms / (frame_dur * 1000.0))) or 1
    min_note_frames = int(round(min_note_ms / (frame_dur * 1000.0))) or 1

    # ---- 1) Gap fill sederhana ----
    last_valid = np.nan
    nan_run = 0
    for i in range(len(f0)):
        if np.isnan(f0[i]):
            nan_run += 1
        else:
            if nan_run > 0 and nan_run <= gap_fill_frames and not np.isnan(last_valid):
                f0[i-nan_run:i] = last_valid
            nan_run = 0
            last_valid = f0[i]

    # ---- 2) Stabil cent lock & boundary ----
    out = np.copy(f0)
    prev_hz = np.nan
    hold_start = 0
    current_note_hz = np.nan

    def hz_to_cents(hz):
        # relative to C0 = 16.35? Actually we only need diff; use log2 ratio * 1200
        return 1200.0 * np.log2(hz)

    for i, hz in enumerate(f0):
        if np.isnan(hz):
            out[i] = np.nan
            continue
        if np.isnan(prev_hz):
            out[i] = hz; prev_hz = hz; current_note_hz = hz; hold_start = i; continue
        # diff in cents (log ratio vs prev)
        diff_c = (hz_to_cents(hz) - hz_to_cents(prev_hz))

        if abs(diff_c) <= cent_lock:
            # treat as same & update slow average
            out[i] = (prev_hz * 0.7 + hz * 0.3)
            prev_hz = out[i]
            continue
        elif abs(diff_c) <= cent_boundary:
            # boundary correction -> keep prev
            if show_debug:
                print(f"[BOUNDARY]{i}: {hz:.1f}Hz -> lock {prev_hz:.1f}Hz Δ{diff_c:.1f}c")
            out[i] = prev_hz
            continue
        else:
            # candidate new note; check future duration
            # look ahead until stable or nan
            j = i + 1
            cnt = 1
            while j < len(f0) and not np.isnan(f0[j]):
                # if revert within lock -> break
                dd = (hz_to_cents(f0[j]) - hz_to_cents(hz))
                if abs(dd) > cent_boundary:
                    break
                cnt += 1; j += 1
            if cnt < min_note_frames:
                # too short -> ignore change
                if show_debug:
                    print(f"[REJECT]{i}: short {cnt}f < {min_note_frames}f")
                out[i] = prev_hz
            else:
                # accept new
                if show_debug:
                    print(f"[ACCEPT]{i}: new note {hz:.1f}Hz (dur >= {cnt}f)")
                out[i] = hz; prev_hz = hz; current_note_hz = hz; hold_start = i
    return out


def sample_blocks(f0_frame, sr, hop_length, interval_ms=FRAME_INTERVAL_MS):
    """Aggregate frame-level f0 ke blok waktu seragam (interval_ms). Return arrays."""
    times_frame = librosa.frames_to_time(np.arange(len(f0_frame)), sr=sr, hop_length=hop_length)
    duration_sec = times_frame[-1] if len(times_frame) else 0
    block_len_s = interval_ms / 1000.0
    sample_starts = np.arange(0, duration_sec + 1e-9, block_len_s)

    sample_times, sample_f0, sample_midi_exact, sample_note_raw = [], [], [], []
    for t0 in sample_starts:
        t1 = t0 + block_len_s
        idx = np.where((times_frame >= t0) & (times_frame < t1))[0]
        if len(idx) == 0:
            sample_times.append(t0); sample_f0.append(np.nan); sample_midi_exact.append(np.nan); sample_note_raw.append('N/A'); continue
        vals = f0_frame[idx]; vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            sample_times.append(t0); sample_f0.append(np.nan); sample_midi_exact.append(np.nan); sample_note_raw.append('N/A'); continue
        med = np.median(vals)
        midi_exact = librosa.hz_to_midi(med)
        note_name = librosa.midi_to_note(midi_exact)
        sample_times.append(t0); sample_f0.append(med); sample_midi_exact.append(midi_exact); sample_note_raw.append(note_name)

    return (np.array(sample_times), np.array(sample_f0), np.array(sample_midi_exact), np.array(sample_note_raw, dtype=object))


def build_csv(out_path, sample_times, sample_f0, sample_midi_exact, sample_note_raw):
    import csv
    midi_int = np.round(sample_midi_exact).astype(float)  # keep nan
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'f0_hz', 'midi_float', 'midi_int', 'note'])
        for t, hz, midf, note, mi in zip(sample_times, sample_f0, sample_midi_exact, sample_note_raw, midi_int):
            if np.isnan(hz):
                w.writerow([f"{t:.6f}", '', '', '', 'N/A'])
            else:
                if np.isnan(midf):
                    w.writerow([f"{t:.6f}", f"{hz:.6f}", '', '', note])
                else:
                    w.writerow([f"{t:.6f}", f"{hz:.6f}", f"{midf:.6f}", f"{int(mi)}", note])
    return out_path


# =============================================================
# ----------------------- PLAYBACK CLASS ----------------------
# =============================================================
class PlaybackController:
    """Manage audio playback + sync pointer."""
    def __init__(self, audio, sr, latency_comp_ms=0.0):
        self.audio = audio
        self.sr = sr
        self.dur = len(audio) / sr
        self.started = False
        self.is_paused = False
        self.play_start_global = 0.0
        self.start_perf = None
        self.pause_pos = 0.0
        self.latency_comp = latency_comp_ms / 1000.0

    def _start_from(self, pos_sec):
        if pos_sec < 0: pos_sec = 0
        if pos_sec >= self.dur: return
        start_idx = int(pos_sec * self.sr)
        sd.stop()
        sd.play(self.audio[start_idx:], self.sr)
        self.play_start_global = pos_sec
        self.start_perf = time.perf_counter()
        self.is_paused = False
        self.started = True

    def pause(self, event=None):
        if not self.started or self.is_paused: return
        pos = self.position()
        self.pause_pos = min(pos, self.dur)
        sd.stop(); self.is_paused = True
        print(f"⏸ Pause @ {self.pause_pos:.3f}s")

    def resume(self, event=None):
        if not self.is_paused: return
        print(f"▶ Resume @ {self.pause_pos:.3f}s")
        self._start_from(self.pause_pos)

    def position(self):
        if self.is_paused: return self.pause_pos
        if not self.started or self.start_perf is None: return 0.0
        elapsed = time.perf_counter() - self.start_perf
        return max(0.0, self.play_start_global + elapsed - self.latency_comp)

    def stop(self):
        sd.stop(); self.is_paused = True


# =============================================================
# ------------------------- MAIN RUN -------------------------
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="Monophonic pitch detection (YIN) for solo/vocal audio.")
    parser.add_argument('--audio', type=str, default='', help='Path audio WAV. Jika kosong akan popup.')
    parser.add_argument('--mode', type=str, choices=['vocal','guitar','ask'], default='ask', help='Sumber: vocal / guitar / ask (popup).')
    parser.add_argument('--no-play', action='store_true', help='Jangan putar audio.')
    parser.add_argument('--csv', action='store_true', help='Ekspor CSV.')
    parser.add_argument('--debug', action='store_true', help='Print debug info.')
    args = parser.parse_args()

    if args.mode == 'ask':
        mode = ui_select_mode()
    else:
        mode = args.mode
    print(f"🔍 Mode Deteksi: {mode.upper()}")

    audio_path = args.audio
    if not audio_path:
        audio_path = ui_select_file()
        if not audio_path:
            raise SystemExit("Tidak ada file dipilih.")

    if not os.path.exists(audio_path):
        raise SystemExit(f"File tidak ditemukan: {audio_path}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration_sec = len(y)/sr
    print(f"🎧 Durasi Audio: {duration_sec:.2f} detik")

    # Source filtering
    y_harm, y_perc = librosa.effects.hpss(y)
    if mode == 'vocal':
        y_proc = bandpass_filter(y_harm, sr, 80, 1200)
        fmin_hz, fmax_hz = librosa.note_to_hz('C2'), librosa.note_to_hz('C6')
    else:
        y_proc = bandpass_filter(y_harm, sr, 80, 2000)
        fmin_hz, fmax_hz = librosa.note_to_hz('E2'), librosa.note_to_hz('E6')

    # YIN
    f0 = librosa.yin(y_proc, fmin=fmin_hz, fmax=fmax_hz, sr=sr, frame_length=4096, win_length=2048, hop_length=HOP_LENGTH)

    # Noise gate
    f0, thr = noise_gate_f0(f0, y_proc, sr, rms_percentile=RMS_NOISE_PERCENTILE)
    if args.debug:
        print(f"[NOISE] RMS thr p{RMS_NOISE_PERCENTILE}: {thr:.6f}")

    # Median smoothing
    f0_sm = median_smooth_nan(f0, SMOOTH_WINDOW)

    # Enforce monophonic continuity
    f0_mono = enforce_monophonic(f0_sm, sr, HOP_LENGTH, show_debug=args.debug)

    # Sample to uniform intervals
    sample_times, sample_f0, sample_midi_exact, sample_note_raw = sample_blocks(f0_mono, sr, HOP_LENGTH, FRAME_INTERVAL_MS)

    # CSV export
    csv_path = ''
    if args.csv or EXPORT_CSV:
        base = os.path.splitext(audio_path)[0]
        csv_path = base + '_monophonic.csv'
        build_csv(csv_path, sample_times, sample_f0, sample_midi_exact, sample_note_raw)
        print(f"💾 CSV tersimpan: {csv_path}")

    # Onset
    onset_times = np.array([])
    if PLOT_ONSETS:
        onset_frames = librosa.onset.onset_detect(y=y_proc, sr=sr, hop_length=HOP_LENGTH)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot main monophonic line (connect successive non-nan)
    valid_idx = ~np.isnan(sample_f0)
    ax.plot(sample_times[valid_idx]*1000.0, sample_f0[valid_idx], '-', linewidth=2, label='F0 Mono')
    ax.scatter(sample_times[valid_idx]*1000.0, sample_f0[valid_idx], s=25)

    # Onsets
    if onset_times.size > 0:
        for o in onset_times:
            ax.axvline(o*1000.0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Label hints: show every Nth label
    note_labels = []
    for i,(t,hz,n) in enumerate(zip(sample_times, sample_f0, sample_note_raw)):
        if np.isnan(hz) or n=='N/A':
            continue
        if i % 10 == 0:  # reduce clutter
            ax.text(t*1000.0, hz, n, fontsize=8, ha='center', va='bottom')
            note_labels.append(n)

    ax.set_title(f"Monophonic Pitch (YIN) | Mode: {mode.upper()} | Durasi: {duration_sec:.2f}s")
    ax.set_xlabel("Waktu (ms)")
    ax.set_ylabel("Frekuensi Fundamental F₀ (Hz)")
    ax.grid(True)
    if note_labels:
        ax.legend()

    # Pointer
    pointer_line = ax.axvline(0, color='black', linewidth=2)

    # Playback controller
    pc = PlaybackController(y_proc, sr, latency_comp_ms=AUDIO_LATENCY_COMP_MS)

    def _pause(event):
        pc.pause()
    def _resume(event):
        pc.resume()

    # Buttons
    axpause = fig.add_axes([0.80, 0.92, 0.08, 0.05])
    axresume = fig.add_axes([0.89, 0.92, 0.08, 0.05])
    bpause = Button(axpause, 'Pause'); bpause.on_clicked(_pause)
    bresume = Button(axresume, 'Resume'); bresume.on_clicked(_resume)

    # Animation update
    def upd(_):
        if PLAY_AUDIO and not pc.started and not pc.is_paused:
            pc._start_from(0.0)
        pos = pc.position()
        if pos >= pc.dur:
            pos = pc.dur
            pc.stop()
        x_ms = pos * 1000.0
        pointer_line.set_xdata([x_ms, x_ms])  # 2-point list avoids deprecation warning
        return (pointer_line,)

    ani = animation.FuncAnimation(fig, upd, interval=20, blit=True, cache_frame_data=False)

    plt.tight_layout(); plt.show()
    pc.stop()


# =============================================================
# ------------------------- ENTRYPOINT -----------------------
# =============================================================
if __name__ == '__main__':
    main()