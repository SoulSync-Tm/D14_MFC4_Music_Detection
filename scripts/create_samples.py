"""
Create short audio samples (clean + noisy) from songs for query testing.

Output structure:
    songs/
    ├── samples/           <- clean clips
    │   ├── Neelothi_sample_1.wav
    │   └── ...
    └── samples_noisy/     <- same clips with added noise/distortion
        ├── Neelothi_sample_1_noisy.wav
        └── ...
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import librosa
import soundfile as sf
import numpy as np

SONGS_DIR    = os.path.join(os.path.dirname(__file__), "..", "songs")
SAMPLES_DIR  = os.path.join(SONGS_DIR, "samples")
NOISY_DIR    = os.path.join(SONGS_DIR, "samples_noisy")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}

# Clip settings
CLIP_DURATION  = 4       # seconds per sample
TARGET_SR      = 22050   # output sample rate

# Start offsets (in seconds) for each sample — spreads across the song
CLIP_OFFSETS = [15, 30, 45, 60, 90, 120, 150]

# Noise profiles applied to each clip (name, SNR dB)
# Lower SNR = more noise = harder match
NOISE_PROFILES = [
    ("light",  20),   # barely audible noise
    ("medium", 10),   # clearly noisy, like a phone recording
    ("heavy",   5),   # heavy background noise
]


def add_noise(y, snr_db):
    """Add white Gaussian noise at the given SNR (dB)."""
    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), size=y.shape)
    return np.clip(y + noise, -1.0, 1.0)


def load_clip(audio_path, start, duration):
    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
        offset=start,
        duration=duration
    )
    return librosa.util.normalize(y)


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(NOISY_DIR,   exist_ok=True)

    song_files = [
        f for f in os.listdir(SONGS_DIR)
        if os.path.isfile(os.path.join(SONGS_DIR, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not song_files:
        print("No audio files found in the songs folder.")
        return

    total_clean = 0
    total_noisy = 0

    print(f"Found {len(song_files)} song(s).")
    print(f"Offsets : {CLIP_OFFSETS}")
    print(f"Noise   : {[p[0] for p in NOISE_PROFILES]}\n")

    for filename in sorted(song_files):
        song_name  = os.path.splitext(filename)[0]
        audio_path = os.path.join(SONGS_DIR, filename)

        print(f"  {filename}")

        try:
            duration = librosa.get_duration(path=audio_path)

            for i, start in enumerate(CLIP_OFFSETS, start=1):
                if start >= duration:
                    print(f"    [{i}] offset {start}s exceeds duration ({duration:.1f}s), skipping.")
                    continue

                actual_dur = min(CLIP_DURATION, duration - start)

                # --- Clean clip ---
                clean_path = os.path.join(SAMPLES_DIR, f"{song_name}_sample_{i}.wav")
                if os.path.isfile(clean_path):
                    print(f"    [{i}] clean already exists, skipping.")
                    y_clean = load_clip(audio_path, start, actual_dur)
                else:
                    y_clean = load_clip(audio_path, start, actual_dur)
                    sf.write(clean_path, y_clean, TARGET_SR)
                    total_clean += 1
                    print(f"    [{i}] clean saved ({actual_dur:.1f}s from {start}s).")

                # --- Noisy clips ---
                for profile_name, snr_db in NOISE_PROFILES:
                    noisy_path = os.path.join(
                        NOISY_DIR,
                        f"{song_name}_sample_{i}_{profile_name}.wav"
                    )
                    if os.path.isfile(noisy_path):
                        print(f"    [{i}] {profile_name} noise already exists, skipping.")
                        continue

                    y_noisy = add_noise(y_clean, snr_db)
                    sf.write(noisy_path, y_noisy, TARGET_SR)
                    total_noisy += 1
                    print(f"    [{i}] {profile_name} noise saved (SNR {snr_db} dB).")

        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nDone.")
    print(f"  Clean  samples : {total_clean}  →  {SAMPLES_DIR}")
    print(f"  Noisy  samples : {total_noisy}  →  {NOISY_DIR}")


if __name__ == "__main__":
    main()
