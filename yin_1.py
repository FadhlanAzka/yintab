import librosa
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
audio_path = askopenfilename(title="Pilih file audio (.wav)", filetypes=[("WAV files", "*.wav")])

if not audio_path:
    print("Tidak ada file yang dipilih.")
    exit()

y, sr = librosa.load(audio_path, sr=None, mono=True)

f0 = librosa.yin(
    y=y,
    fmin=librosa.note_to_hz('E2'),
    fmax=librosa.note_to_hz('E6'),
    sr=sr,
    frame_length=2048,
    hop_length=512,
)

valid_f0 = f0[~np.isnan(f0)]
if len(valid_f0) > 0:
    average_pitch = np.median(valid_f0)
    print(f"Pitch terdeteksi: {average_pitch:.2f} Hz")

    midi_number = librosa.hz_to_midi(average_pitch)
    note_name = librosa.midi_to_note(midi_number)
    print(f"Note: {note_name}")
else:
    print("Tidak ada pitch yang terdeteksi.")