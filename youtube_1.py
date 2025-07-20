import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from yt_dlp import YoutubeDL

guitar_notes = {
    'E': ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3'],
    'A': ['A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3'],
    'D': ['D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4'],
    'G': ['G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4'],
    'B': ['B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'],
    'e': ['E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5']
}

def download_audio_from_youtube(url, output_path='audio.wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'outtmpl': 'temp.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    os.rename('temp.wav', output_path)
    return output_path

def hz_to_note_name(hz):
    if hz <= 0:
        return None
    note_num = int(np.round(12 * np.log2(hz / 440.0))) + 69
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note_num // 12 - 1
    return f"{note_names[note_num % 12]}{octave}"

def find_fret(note_name):
    for string, frets in guitar_notes.items():
        if note_name in frets:
            return string, frets.index(note_name)
    return None, None

def detect_yin_to_tab(audio_path):
    y, sr = librosa.load(audio_path)
    pitches = librosa.yin(y, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('E6'), sr=sr)
    times = librosa.times_like(pitches, sr=sr)

    tablature = []
    for t, pitch in zip(times, pitches):
        note = hz_to_note_name(pitch)
        if note:
            string, fret = find_fret(note)
            if string:
                tablature.append((t, string, fret))
    return tablature

def plot_tablature(tablature):
    string_order = ['e', 'B', 'G', 'D', 'A', 'E']
    fig, ax = plt.subplots(figsize=(12, 4))
    
    for t, string, fret in tablature:
        y = string_order.index(string)
        ax.text(t, y, str(fret), va='center', ha='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_yticks(range(len(string_order)))
    ax.set_yticklabels(string_order)
    ax.set_xlabel("Time (s)")
    ax.set_title("Guitar Tablature (YIN Pitch Detection)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

youtube_link = 'https://www.youtube.com/watch?v=YOyQgtcmK9s'
audio_file = download_audio_from_youtube(youtube_link)
tablature = detect_yin_to_tab(audio_file)
plot_tablature(tablature)