import argparse
import os
import time
import csv
import numpy as np
import librosa
import sounddevice as sd
import scipy.signal as signal
from tkinter import Tk, filedialog, Toplevel, Label, Button as TkButton

# =============================================================
# --------------------- PARAMETER GLOBAL ---------------------
# =============================================================
FRAME_INTERVAL_MS = 50
SMOOTH_WINDOW = 5
CENT_NOTE_LOCK = 12
CENT_BOUNDARY_CORRECTION = 18
GAP_FILL_MS = 100
MIN_NOTE_MS = 60
SHOW_DEBUG = False
HOP_LENGTH = 256
RMS_NOISE_PERCENTILE = 20
MAX_FRET = 20
HAND_SPAN = 5
TOT_FRETS = MAX_FRET + 1
GUITAR_TUNING_ORDER = [(6,40),(5,45),(4,50),(3,55),(2,59),(1,64)]
PREFER_OPEN_STRING = True

_OPEN_MIDI_BY_STRING = {s:m for s,m in GUITAR_TUNING_ORDER}

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
    out = []
    if np.isnan(midi_note):
        return out
    for s, open_midi in GUITAR_TUNING_ORDER:
        fret = int(round(midi_note - open_midi))
        if 0 <= fret <= MAX_FRET:
            out.append((s,fret))
    return out

def tablature_token_idx(string_num, fret_num):
    if string_num < 1 or string_num > 6 or fret_num < 0 or fret_num > MAX_FRET:
        return -1
    return (6 - string_num) * TOT_FRETS + fret_num

def fingering_optimize(midi_seq, fret_span=5):
    """Optimalkan pemilihan fret agar perpindahan antar nada minimal, dalam jangkauan fret tertentu."""
    n = len(midi_seq)
    valid_cand = []
    for m in midi_seq:
        if np.isnan(m):
            valid_cand.append([(0, -1)])  # kosong
            continue
        positions = []
        for s, open_midi in GUITAR_TUNING_ORDER:
            fret = int(round(m - open_midi))
            if 0 <= fret <= MAX_FRET:
                positions.append((s, fret))
        if not positions:
            valid_cand.append([(0, -1)])  # tidak bisa dimainkan
        else:
            valid_cand.append(positions)

    dp = [{} for _ in range(n)]
    back = [{} for _ in range(n)]

    # Init
    for k, (s, f) in enumerate(valid_cand[0]):
        dp[0][k] = 0

    for i in range(1, n):
        for k, (s, f) in enumerate(valid_cand[i]):
            best_cost = float('inf')
            best_prev = None
            for j, (ps, pf) in enumerate(valid_cand[i - 1]):
                if dp[i - 1].get(j) is None:
                    continue
                cost = abs(f - pf)
                if cost <= fret_span and dp[i - 1][j] + cost < best_cost:
                    best_cost = dp[i - 1][j] + cost
                    best_prev = j
            if best_prev is not None:
                dp[i][k] = best_cost
                back[i][k] = best_prev

    # Ambil posisi akhir terbaik
    end_k = min(dp[-1], key=lambda k: dp[-1][k]) if dp[-1] else 0
    result = [None] * n
    k = end_k
    for i in reversed(range(n)):
        try:
            result[i] = valid_cand[i][k]
            k = back[i].get(k, 0)
        except IndexError:
            result[i] = midi_to_guitar_position(midi_seq[i])
    return result


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
    f0 = f0.copy()
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    frame_dur = times[1] - times[0] if len(times) > 1 else hop_length / sr
    gap_fill_frames = int(round(gap_fill_ms / (frame_dur * 1000.0))) or 1
    min_note_frames = int(round(min_note_ms / (frame_dur * 1000.0))) or 1

    last_valid = np.nan; nan_run = 0
    for i in range(len(f0)):
        if np.isnan(f0[i]):
            nan_run += 1
        else:
            if nan_run > 0 and nan_run <= gap_fill_frames and not np.isnan(last_valid):
                f0[i-nan_run:i] = last_valid
            nan_run = 0; last_valid = f0[i]

    out = np.copy(f0); prev_hz = np.nan
    def hz2c(h): return 1200.0*np.log2(h)
    for i,hz in enumerate(f0):
        if np.isnan(hz): out[i]=np.nan; continue
        if np.isnan(prev_hz): out[i]=hz; prev_hz=hz; continue
        diff_c = hz2c(hz) - hz2c(prev_hz)
        if abs(diff_c) <= cent_lock:
            out[i] = (prev_hz*0.7 + hz*0.3); prev_hz=out[i]; continue
        elif abs(diff_c) <= cent_boundary:
            out[i]=prev_hz; continue
        else:
            j=i+1; cnt=1
            while j<len(f0) and not np.isnan(f0[j]):
                dd = hz2c(f0[j]) - hz2c(hz)
                if abs(dd) > cent_boundary: break
                cnt+=1; j+=1
            if cnt < min_note_frames:
                out[i]=prev_hz
            else:
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

def ui_select_mode():
    root = Tk(); root.withdraw()
    holder = {'mode': None}
    def _v(): holder['mode'] = 'vocal'; win.destroy()
    def _g(): holder['mode'] = 'guitar'; win.destroy()
    win = Toplevel(root)
    win.title("Pilih Deteksi")
    Label(win, text="Pilih Deteksi", font=("Helvetica", 12, "bold")).pack(padx=20, pady=(20, 10))
    TkButton(win, text="Vokal", width=15, command=_v).pack(pady=5)
    TkButton(win, text="Gitar", width=15, command=_g).pack(pady=(0, 20))
    win.grab_set(); root.wait_window(win); root.destroy()
    if holder['mode'] is None:
        raise SystemExit("Tidak ada mode dipilih.")
    return holder['mode']

def process_single_audio(audio_path, mode, optimize=True, debug=False):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration_sec = len(y)/sr
    print(f"🎧 {os.path.basename(audio_path)} | Durasi: {duration_sec:.2f}s")

    y_harm, y_perc = librosa.effects.hpss(y)
    if mode == 'vocal':
        y_proc = bandpass_filter(y_harm, sr, 80, 1200)
        fmin_hz, fmax_hz = librosa.note_to_hz('C2'), librosa.note_to_hz('C6')
    else:
        y_proc = bandpass_filter(y_harm, sr, 80, 2000)
        fmin_hz, fmax_hz = librosa.note_to_hz('E2'), librosa.note_to_hz('E6')

    f0 = librosa.yin(y_proc, fmin=fmin_hz, fmax=fmax_hz, sr=sr, frame_length=4096, win_length=2048, hop_length=HOP_LENGTH)
    f0, thr = noise_gate_f0(f0, y_proc, sr, rms_percentile=RMS_NOISE_PERCENTILE, hop_length=HOP_LENGTH)
    if debug:
        print(f"[NOISE] RMS thr p{RMS_NOISE_PERCENTILE}: {thr:.6f}")

    f0_sm = median_smooth_nan(f0, SMOOTH_WINDOW)
    f0_mono = enforce_monophonic(f0_sm, sr, HOP_LENGTH, show_debug=debug)
    sample_times, sample_f0, sample_midi_exact = sample_blocks(f0_mono, sr, HOP_LENGTH, FRAME_INTERVAL_MS)

    return sample_times, sample_midi_exact

def main():
    parser = argparse.ArgumentParser(description="Monophonic pitch detection (YIN) batch processing → unified CSV for LSTM tablature.")
    parser.add_argument('--dir', type=str, default='', help='Path folder berisi file WAV.')
    parser.add_argument('--mode', type=str, choices=['vocal','guitar','ask'], default='ask', help='Sumber: vocal / guitar / ask (popup).')
    parser.add_argument('--out', type=str, default='output_lstm_tab.csv', help='Nama file output CSV gabungan.')
    parser.add_argument('--no-opt', action='store_true', help='Nonaktifkan fingering optimizer (pakai fret terendah).')
    parser.add_argument('--debug', action='store_true', help='Print debug info.')
    args = parser.parse_args()

    if args.mode == 'ask':
        mode = ui_select_mode()
    else:
        mode = args.mode
    print(f"🔍 Mode Deteksi: {mode.upper()}")

    if not args.dir:
        root = Tk(); root.withdraw()
        args.dir = filedialog.askdirectory(title="Pilih folder audio")
        root.destroy()
    if not os.path.isdir(args.dir):
        raise SystemExit("Folder tidak valid.")

    files = sorted([f for f in os.listdir(args.dir) if f.lower().endswith('.wav')])
    if not files:
        raise SystemExit("Tidak ada file .wav ditemukan di folder tersebut.")

    all_rows = [['file','time_sec','midi_note','string_num','fret_num','token_idx']]
    for fname in files:
        fpath = os.path.join(args.dir, fname)
        sample_times, sample_midi_exact = process_single_audio(fpath, mode, optimize=not args.no_opt, debug=args.debug)
        midi_seq = sample_midi_exact.copy()
        best_positions = fingering_optimize(midi_seq) if not args.no_opt else [midi_to_guitar_position(m) for m in midi_seq]

        for t, m, (s,f) in zip(sample_times, midi_seq, best_positions):
            if np.isnan(m) or s<=0 or f<0:
                all_rows.append([fname, f"{t:.6f}", '', '', '', ''])
            else:
                token = tablature_token_idx(s, f)
                all_rows.append([fname, f"{t:.6f}", int(round(m)), s, f, token])

    out_path = os.path.join(args.dir, args.out)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
    print(f"✅ CSV gabungan tersimpan: {out_path}")

if __name__ == '__main__':
    main()