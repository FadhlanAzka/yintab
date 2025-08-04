import argparse
import os
import time
import csv
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
Deteksi pitch dominan (monophonic) dari sinyal audio (solo gitar / vokal) menggunakan YIN.

Fitur utama:
  • Popup pemilihan mode (Vokal / Gitar) → beda band-pass + range YIN.
  • HPSS + Band-pass + Noise Gate RMS.
  • Smoothing + Enforce Monophonic (cent lock, gap fill, min dur).
  • (Opsional) Optimasi fingering antar event agar posisi fret berdekatan & playable.
  • Sampling seragam (FRAME_INTERVAL_MS) untuk dataset frame-level.
  • Plot interaktif + Playback audio + Pointer sinkron + tombol Pause/Resume.
  • Ekspor CSV siap LSTM (frame-level) → time_sec, midi_note, string_num, fret_num, token_idx.
  • (Opsional) Ekspor CSV event-level (start_s, end_s, dur_s, midi_note, string_num, fret_num, token_idx) untuk model berbasis event.

Catatan Tablature Mapping:
  Tuning standar gitar (low→high):
    6:E2=40, 5:A2=45, 4:D3=50, 3:G3=55, 2:B3=59, 1:E4=64 (MIDI numbers).
  token_idx dihitung 0..(6*TOT_FRETS-1) dengan urutan dari senar 6 (low E) naik ke senar 1.
    token_idx = (6 - string_num) * TOT_FRETS + fret_num
    Contoh: senar 6 fret 0 → idx=0; senar 6 fret 3 → 3; senar 5 fret 0 → TOT_FRETS; dst.

Gunakan argumen --debug untuk cetak info tambahan.
"""

# =============================================================
# --------------------- PARAMETER GLOBAL ---------------------
# =============================================================
FRAME_INTERVAL_MS = 50           # resolusi waktu sampling output (ms)
SMOOTH_WINDOW = 5                # median smoothing di domain frame YIN
CENT_NOTE_LOCK = 12              # < ini (cent) tetap sama
CENT_BOUNDARY_CORRECTION = 18    # koreksi balik jika loncatan kecil
GAP_FILL_MS = 100                # isi gap pendek (np.nan) dengan nada sebelumnya (ms)
MIN_NOTE_MS = 60                 # durasi minimum agar nada baru dianggap sah (ms)
SHOW_DEBUG = False
HOP_LENGTH = 256                 # hop untuk STFT/YIN
RMS_NOISE_PERCENTILE = 20        # gating energi rendah
EXPORT_CSV = True                # simpan hasil csv frame-level
EXPORT_EVENT_CSV = False         # set True utk simpan event-level
PLAY_AUDIO = True                # mainkan audio + pointer
PLOT_ONSETS = True               # tampilkan onset
AUDIO_LATENCY_COMP_MS = 0        # kompensasi pointer vs audio (ms)
MAX_FRET = 20                    # batas fret yang dimodelkan (0..MAX_FRET)
HAND_SPAN = 5                    # jangkauan nyaman (fret)
# =============================================================

# TOT_FRETS = jumlah posisi per senar
TOT_FRETS = MAX_FRET + 1  # misal 21 jika MAX_FRET=20

# =============================================================
# -------------------- TUNING & MAPPING ----------------------
# =============================================================
# (low→high)   6:E2  5:A2  4:D3  3:G3  2:B3  1:E4
GUITAR_TUNING_ORDER = [(6,40),(5,45),(4,50),(3,55),(2,59),(1,64)]

# Fast lookups
_OPEN_MIDI_BY_STRING = {s:m for s,m in GUITAR_TUNING_ORDER}


# Prefer open string pada pemetaan posisi gitar
PREFER_OPEN_STRING = True

# (fungsi midi_to_guitar_position diubah agar prefer fret 0 bila ada)
def midi_to_guitar_position(midi_note):
    if np.isnan(midi_note):
        return (0, -1)
    positions = []
    for s, open_midi in GUITAR_TUNING_ORDER:
        fret = int(round(midi_note - open_midi))
        if 0 <= fret <= MAX_FRET:
            positions.append((s, fret))
    if not positions:
        return (0, -1)
    if PREFER_OPEN_STRING:
        for s, f in positions:
            if f == 0:
                return (s, f)
    return min(positions, key=lambda x: x[1])


def all_positions_for_midi(midi_note):
    """Semua posisi valid (string,fret) utk midi_note."""
    out = []
    if np.isnan(midi_note):
        return out
    for s, open_midi in GUITAR_TUNING_ORDER:
        fret = int(round(midi_note - open_midi))
        if 0 <= fret <= MAX_FRET:
            out.append((s,fret))
    return out


def tablature_token_idx(string_num, fret_num):
    """Map (string,fret) ke token_idx 0..(6*TOT_FRETS-1).
    Urutan index: semua fret senar 6, lalu senar 5, ... hingga senar 1.
    Rumus: idx = (6 - string_num) * TOT_FRETS + fret_num.
    Return -1 jika tidak valid.
    """
    if string_num < 1 or string_num > 6 or fret_num < 0 or fret_num > MAX_FRET:
        return -1
    return (6 - string_num) * TOT_FRETS + fret_num


# =============================================================
# ------------------- FINGERING OPTIMIZER --------------------
# =============================================================

def fingering_optimize(midi_seq, hand_span=HAND_SPAN,
                       w_fret=1.0, w_string=0.8, w_shift=4.0,
                       w_high=0.05, open_penalty=0.1):
    """Dynamic-programming pilih jalur posisi (string,fret) ter-playable.
    midi_seq: list midi ints (NaN di-skip).
    Return list posisi panjang sama (s,f) per not.
    """
    # Kumpulkan kandidat utk tiap nada
    cand_list = []
    valid_idx_map = []  # simpan indeks midi yg valid
    for i,m in enumerate(midi_seq):
        if np.isnan(m):
            cand_list.append([])
        else:
            cand_list.append(all_positions_for_midi(m))
            valid_idx_map.append(i)

    if not valid_idx_map:
        return [(0,-1) for _ in midi_seq]

    # DP hanya atas yg valid; nanti isi kembali ke full seq
    valid_midi = [midi_seq[i] for i in valid_idx_map]
    valid_cand = [cand_list[i] for i in valid_idx_map]

    # dp arrays
    N = len(valid_midi)
    dp = [[np.inf]*len(c) for c in valid_cand]
    back = [[-1]*len(c) for c in valid_cand]

    # init
    for k,(s,f) in enumerate(valid_cand[0]):
        dp[0][k] = 0.0

    # recurrence
    for i in range(1,N):
        for k,(s1,f1) in enumerate(valid_cand[i]):
            best_cost = np.inf; best_prev=-1
            for j,(s0,f0) in enumerate(valid_cand[i-1]):
                cost = 0.0
                cost += w_fret * abs(f1-f0)
                cost += w_string * abs(s1-s0)
                if abs(f1-f0) > hand_span:
                    cost += w_shift * (abs(f1-f0) - hand_span)
                if f1 >= 12:
                    cost += w_high * (f1-11)
                if f1 == 0:
                    cost += open_penalty
                ctot = dp[i-1][j] + cost
                if ctot < best_cost:
                    best_cost = ctot; best_prev=j
            dp[i][k] = best_cost
            back[i][k] = best_prev

    # backtrack
    end_k = int(np.argmin(dp[-1]))
    path_valid = []
    for i in reversed(range(N)):
        s,f = valid_cand[i][end_k]
        path_valid.append((s,f))
        end_k = back[i][end_k]
    path_valid.reverse()

    # expand ke full panjang midi_seq (na -> (0,-1))
    full_path = [(0,-1) for _ in midi_seq]
    for seq_i, (s,f) in zip(valid_idx_map, path_valid):
        full_path[seq_i] = (s,f)
    return full_path


# =============================================================
# ----------------------- UI FUNCTIONS -----------------------
# =============================================================

def ui_select_mode():
    """Modal Tkinter popup: pilih deteksi Vokal / Gitar."""
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
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(title="Pilih file audio (.wav)", filetypes=[("WAV files", "*.wav")])
    root.destroy(); return path


# =============================================================
# ---------------------- DSP UTILITIES -----------------------
# =============================================================

def bandpass_filter(data, sr, low_hz=80, high_hz=2000, order=10):
    sos = signal.butter(order, [low_hz, high_hz], btype='bandpass', fs=sr, output='sos')
    return signal.sosfilt(sos, data)


def noise_gate_f0(f0, y_proc, sr, rms_percentile=RMS_NOISE_PERCENTILE, f0_min=50, f0_max=2000, hop_length=HOP_LENGTH):
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
    out = arr.copy(); n=len(arr)
    for i in range(n):
        s = max(0, i - win // 2); e = min(n, i + win // 2 + 1)
        ww = arr[s:e]; ww = ww[~np.isnan(ww)]
        if len(ww) > 0: out[i] = np.median(ww)
    return out


def enforce_monophonic(f0, sr, hop_length, cent_lock=CENT_NOTE_LOCK, cent_boundary=CENT_BOUNDARY_CORRECTION,
                        gap_fill_ms=GAP_FILL_MS, min_note_ms=MIN_NOTE_MS, show_debug=False):
    """Stabilisasi f0 agar monophonic: gap fill, cent lock, min dur note."""
    f0 = f0.copy()
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    frame_dur = times[1] - times[0] if len(times) > 1 else hop_length / sr
    gap_fill_frames = int(round(gap_fill_ms / (frame_dur * 1000.0))) or 1
    min_note_frames = int(round(min_note_ms / (frame_dur * 1000.0))) or 1

    # gap fill
    last_valid = np.nan; nan_run = 0
    for i in range(len(f0)):
        if np.isnan(f0[i]):
            nan_run += 1
        else:
            if nan_run > 0 and nan_run <= gap_fill_frames and not np.isnan(last_valid):
                f0[i-nan_run:i] = last_valid
            nan_run = 0; last_valid = f0[i]

    # cent/boundary stabilizer
    out = np.copy(f0); prev_hz = np.nan
    def hz2c(h): return 1200.0*np.log2(h)
    for i,hz in enumerate(f0):
        if np.isnan(hz): out[i]=np.nan; continue
        if np.isnan(prev_hz): out[i]=hz; prev_hz=hz; continue
        diff_c = hz2c(hz) - hz2c(prev_hz)
        if abs(diff_c) <= cent_lock:
            out[i] = (prev_hz*0.7 + hz*0.3); prev_hz=out[i]; continue
        elif abs(diff_c) <= cent_boundary:
            if show_debug: print(f"[BOUNDARY]{i}: {hz:.1f}Hz -> {prev_hz:.1f}Hz Δ{diff_c:.1f}c")
            out[i]=prev_hz; continue
        else:
            # calon nada baru; cek durasi minimal
            j=i+1; cnt=1
            while j<len(f0) and not np.isnan(f0[j]):
                dd = hz2c(f0[j]) - hz2c(hz)
                if abs(dd) > cent_boundary: break
                cnt+=1; j+=1
            if cnt < min_note_frames:
                if show_debug: print(f"[REJECT]{i}: short {cnt}f")
                out[i]=prev_hz
            else:
                if show_debug: print(f"[ACCEPT]{i}: new {hz:.1f}Hz dur>={cnt}")
                out[i]=hz; prev_hz=hz
    return out


def sample_blocks(f0_frame, sr, hop_length, interval_ms=FRAME_INTERVAL_MS):
    times_frame = librosa.frames_to_time(np.arange(len(f0_frame)), sr=sr, hop_length=hop_length)
    duration_sec = times_frame[-1] if len(times_frame) else 0
    block_len_s = interval_ms / 1000.0
    sample_starts = np.arange(0, duration_sec + 1e-9, block_len_s)

    sample_times, sample_f0, sample_midi_exact = [], [], []
    for t0 in sample_starts:
        t1 = t0 + block_len_s
        idx = np.where((times_frame >= t0) & (times_frame < t1))[0]
        if len(idx) == 0:
            sample_times.append(t0); sample_f0.append(np.nan); sample_midi_exact.append(np.nan); continue
        vals = f0_frame[idx]; vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            sample_times.append(t0); sample_f0.append(np.nan); sample_midi_exact.append(np.nan); continue
        med = np.median(vals)
        midi_exact = librosa.hz_to_midi(med)
        sample_times.append(t0); sample_f0.append(med); sample_midi_exact.append(midi_exact)
    return (np.array(sample_times), np.array(sample_f0), np.array(sample_midi_exact))


# =============================================================
# --------------------- CSV BUILDERS -------------------------
# =============================================================

def build_csv_lstm(out_path, sample_times, sample_midi_exact, optimize=True):
    """Simpan CSV minimal untuk training LSTM tablature (frame-level).
    Kolom: time_sec, midi_note, string_num, fret_num, token_idx
    Jika optimize=True → jalankan fingering_optimize secara global.
    """
    midi_seq = sample_midi_exact.copy()

    # jalankan fingering optimizer (abaikan NaN)
    if optimize:
        best_positions = fingering_optimize(midi_seq)
    else:
        best_positions = [midi_to_guitar_position(m) for m in midi_seq]

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['time_sec','midi_note','string_num','fret_num','token_idx'])
        for t, midf, (s,fret) in zip(sample_times, midi_seq, best_positions):
            if np.isnan(midf) or s<=0 or fret<0:
                w.writerow([f"{t:.6f}", '', '', '', ''])
                continue
            token = tablature_token_idx(s, fret)
            w.writerow([f"{t:.6f}", int(round(midf)), s, fret, token])
    return out_path


def build_csv_events(out_path, sample_times, sample_midi_exact, optimize=True):
    """Bangun CSV event-level dari data frame-level.
    Segmentasi ketika pitch berubah atau menjadi NaN.
    Kolom: start_s, end_s, duration_s, midi_note, note_name, string_num, fret_num, token_idx
    """
    times = sample_times; midi = sample_midi_exact
    events = []
    note_start = None; note_val = None
    for i,(t,m) in enumerate(zip(times, midi)):
        if np.isnan(m):
            if note_val is not None:
                events.append((note_start, t, note_val))
                note_start=None; note_val=None
            continue
        m_int = int(round(m))
        if note_val is None:
            note_start = t; note_val = m_int
        elif m_int != note_val:
            events.append((note_start, t, note_val))
            note_start = t; note_val = m_int
    if note_val is not None:
        events.append((note_start, times[-1]+FRAME_INTERVAL_MS/1000.0, note_val))

    # optimasi fingering di level event
    midi_seq = [ev[2] for ev in events]
    if optimize:
        best_positions = fingering_optimize(midi_seq)
    else:
        best_positions = [midi_to_guitar_position(m) for m in midi_seq]

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['start_s','end_s','duration_s','midi_note','note_name','string_num','fret_num','token_idx'])
        for (st,et,m), (s,fret) in zip(events, best_positions):
            dur = et - st
            note_name = librosa.midi_to_note(m)
            if s<=0 or fret<0:
                w.writerow([f"{st:.3f}", f"{et:.3f}", f"{dur:.3f}", m, note_name, '', '', ''])
                continue
            token = tablature_token_idx(s, fret)
            w.writerow([f"{st:.3f}", f"{et:.3f}", f"{dur:.3f}", m, note_name, s, fret, token])
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
        sd.stop(); sd.play(self.audio[start_idx:], self.sr)
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
    parser = argparse.ArgumentParser(description="Monophonic pitch detection (YIN) for solo/vocal audio → CSV LSTM tablature.")
    parser.add_argument('--audio', type=str, default='', help='Path audio WAV. Jika kosong akan popup.')
    parser.add_argument('--mode', type=str, choices=['vocal','guitar','ask'], default='ask', help='Sumber: vocal / guitar / ask (popup).')
    parser.add_argument('--no-play', action='store_true', help='Jangan putar audio.')
    parser.add_argument('--no-csv', action='store_true', help='Jangan ekspor CSV frame-level.')
    parser.add_argument('--event-csv', action='store_true', help='Ekspor CSV event-level.')
    parser.add_argument('--no-opt', action='store_true', help='Nonaktifkan fingering optimizer (pakai fret terendah).')
    parser.add_argument('--debug', action='store_true', help='Print debug info.')
    args = parser.parse_args()

    if args.mode == 'ask':
        mode = ui_select_mode()
    else:
        mode = args.mode
    print(f"🔍 Mode Deteksi: {mode.upper()}")

    audio_path = args.audio or ui_select_file()
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
    f0, thr = noise_gate_f0(f0, y_proc, sr, rms_percentile=RMS_NOISE_PERCENTILE, hop_length=HOP_LENGTH)
    if args.debug:
        print(f"[NOISE] RMS thr p{RMS_NOISE_PERCENTILE}: {thr:.6f}")

    # Median smoothing
    f0_sm = median_smooth_nan(f0, SMOOTH_WINDOW)

    # Enforce monophonic continuity
    f0_mono = enforce_monophonic(f0_sm, sr, HOP_LENGTH, show_debug=args.debug)

    # Sample to uniform intervals (frame-level dataset)
    sample_times, sample_f0, sample_midi_exact = sample_blocks(f0_mono, sr, HOP_LENGTH, FRAME_INTERVAL_MS)

    # Build CSV (frame-level LSTM)
    if not args.no_csv and EXPORT_CSV:
        base = os.path.splitext(audio_path)[0]
        csv_path = base + '_lstm_tab.csv'
        build_csv_lstm(csv_path, sample_times, sample_midi_exact, optimize=not args.no_opt)
        print(f"💾 CSV frame-level tersimpan: {csv_path}")

    # Build CSV (event-level)
    if args.event_csv or EXPORT_EVENT_CSV:
        base = os.path.splitext(audio_path)[0]
        event_csv_path = base + '_events.csv'
        build_csv_events(event_csv_path, sample_times, sample_midi_exact, optimize=not args.no_opt)
        print(f"💾 CSV event-level tersimpan: {event_csv_path}")

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
    for i,(t,hz) in enumerate(zip(sample_times, sample_f0)):
        if np.isnan(hz):
            continue
        if i % 10 == 0:  # reduce clutter
            note_name = librosa.midi_to_note(sample_midi_exact[i])
            ax.text(t*1000.0, hz, note_name, fontsize=8, ha='center', va='bottom')

    ax.set_title(f"Monophonic Pitch (YIN) | Mode: {mode.upper()} | Durasi: {duration_sec:.2f}s")
    ax.set_xlabel("Waktu (ms)")
    ax.set_ylabel("Frekuensi Fundamental F₀ (Hz)")
    ax.grid(True)
    ax.legend()

    # Pointer
    pointer_line = ax.axvline(0, color='black', linewidth=2)

    # Playback controller
    pc = PlaybackController(y_proc, sr, latency_comp_ms=AUDIO_LATENCY_COMP_MS)

    def _pause(event): pc.pause()
    def _resume(event): pc.resume()

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
            pos = pc.dur; pc.stop()
        x_ms = pos * 1000.0
        pointer_line.set_xdata([x_ms, x_ms])  # 2-point list avoids deprecation warning
        return (pointer_line,)

    animation.FuncAnimation(fig, upd, interval=20, blit=True, cache_frame_data=False)
    plt.tight_layout(); plt.show()
    pc.stop()


# =============================================================
# ------------------------- ENTRYPOINT -----------------------
# =============================================================
if __name__ == '__main__':
    main()