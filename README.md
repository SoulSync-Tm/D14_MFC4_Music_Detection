# Shazam-Style Audio Fingerprinting System

A Python implementation of Shazam-style audio fingerprinting for song identification from short clips (3–5 seconds). Uses spectral peak detection, anchor-target constellation mapping, and offset-alignment voting for robust matching — even under noise.

## 📋 Table of Contents

- [Overview](#overview)
- [Why Preprocessing?](#-why-does-audio-need-preprocessing)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Pipeline](#-pipeline)
- [Scripts Reference](#-scripts-reference)
- [Configuration](#-configuration)
- [Evaluation](#-evaluation)
- [Robustness](#-robustness-features)
- [Docker Setup](#-docker-setup)
- [References](#-references)

## 🎯 Overview

This system implements industry-standard Shazam-style audio fingerprinting featuring:

- **Audio Preprocessing**: Mono conversion, resampling to 22050 Hz, DC removal, amplitude normalisation
- **Spectral Analysis**: STFT with Hann window and configurable FFT size
- **Peak Detection**: Local maxima via `maximum_filter` with amplitude thresholding
- **Constellation Mapping**: Time-frequency landmark visualisation (full song + zoomed view)
- **64-bit Fingerprint Hashing**: Anchor-target pairing packed as `[f1·16][f2·16][Δt·16]`
- **SQLite Database**: Songs table + fingerprints table with `idx_hash` index for O(1) lookups
- **Offset-Alignment Voting**: Time-invariant matching robust to recording offset
- **Score Normalisation & Thresholding**: Confidence scores (0–1) with rejection below threshold

---

## 🔥 Why Does Audio Need Preprocessing?

Real-world audio recordings are rarely clean. Noise and distortion are introduced at multiple stages of the recording chain, making raw audio unreliable for direct fingerprinting. Common sources include:

| Source | Description |
|---|---|
| **Cheap Microphones** | Low-quality capsules introduce self-noise and frequency coloration, adding a constant noise floor to recordings |
| **ADC Imperfections** | Analog-to-Digital Converters introduce quantization noise, clipping, and non-linearity errors during the analog → digital conversion |
| **Recording Devices** | Interference from device electronics, power supply hum (50/60 Hz), and poor shielding bleed into the audio signal |
| **Compression Artifacts** | Lossy codecs (MP3, AAC) discard high-frequency detail and introduce ringing/pre-echo artifacts, altering the spectral content |

> This is common in all recorded audio — even professional recordings contain low-level noise from the environment, equipment, and encoding pipeline.

### 🛠️ How Preprocessing Fixes This

Before fingerprinting, the audio pipeline applies a series of normalisation and conditioning steps to strip away noise and produce a consistent, comparable signal:

1. **Mono Conversion** — Collapses stereo channels into a single channel. Eliminates channel imbalance and halves data size without losing tonal information relevant to fingerprinting.

2. **Resampling to 22050 Hz** — Standardises the sample rate across all recordings. This ensures fingerprints generated from different devices (phones, studio mics, web streams) are directly comparable. 22050 Hz captures frequencies up to 11025 Hz — well above the perceptually relevant range.

3. **Amplitude Normalisation** — Scales the waveform so the peak amplitude is ±1. Removes volume differences caused by recording level, gain staging, or playback volume.

4. **Spectral Peak Thresholding** — During STFT analysis, an amplitude threshold (`amp_min = -40 dB`) discards low-energy spectral bins that are more likely to contain noise than musical content. Only the strongest, most stable peaks are retained as landmarks.

Together, these steps ensure the fingerprinting system focuses on the **stable spectral structure** of a song rather than the noise layered on top of it.

---

## 📁 Project Structure

```
music-detection/
├── fingerprint.py             # AudioFingerprinter class (preprocess, spectrogram, peaks, hashes)
├── database.py                # SQLite helpers (create, insert, bulk match)
├── experiment.ipynb           # Interactive exploration notebook
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── README.md                  # This file
├── scripts/
│   ├── insert_songs.py        # Register songs from songs/ into the DB
│   ├── fingerprint_songs.py   # Compute & store fingerprints for all registered songs
│   ├── create_samples.py      # Generate clean + noisy test clips
│   ├── match_song.py          # Match a single audio file against the DB
│   ├── evaluate.py            # Batch evaluation across clean and noisy samples
│   └── drop_tables.py         # Drop all DB tables (reset)
├── songs/                     # Full-length audio files
│   ├── SongName.mp3
│   ├── samples/               # Clean 4s clips
│   │   ├── SongName_sample_1.wav
│   │   └── ...
│   └── samples_noisy/         # Noisy variants (light / medium / heavy)
│       ├── SongName_sample_1_light.wav
│       ├── SongName_sample_1_medium.wav
│       ├── SongName_sample_1_heavy.wav
│       └── ...
└── database/
    └── fingerprints.db        # SQLite database (auto-created)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (or Docker)
- Audio files in `songs/` (MP3, WAV, FLAC, OGG)

### Local Setup

```bash
# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Full Pipeline (run in order)

```bash
# 1. Register song names in the database
python scripts/insert_songs.py

# 2. Compute and store fingerprints for every registered song
python scripts/fingerprint_songs.py

# 3. Generate test clips (clean + noisy)
python scripts/create_samples.py

# 4. Match a single clip
python scripts/match_song.py songs/samples/SongName_sample_2.wav

# 5. Evaluate the whole system
python scripts/evaluate.py
```

---

## 🔄 Pipeline

```
Audio File
    │
    ▼
preprocess()          mono · resample 22050 Hz · normalize · DC remove
    │
    ▼
generate_spectrogram() STFT → |magnitude| → log dB
    │
    ▼
find_peaks()          maximum_filter · amplitude threshold → (freq_idx, time_idx)
    │
    ▼
generate_hashes()     anchor-target pairing → 64-bit hash · time_offset
    │
    ▼
SQLite DB             fingerprints(song_id, hash_value BIGINT, time_offset INTEGER)
    │                 idx_hash index for O(1) lookup
    ▼
match_fingerprints_bulk()  batch IN(...) query
    │
    ▼
Offset-Alignment Voting    votes[song_id][db_t − query_t] += 1
    │
    ▼
Score Normalisation        confidence = max_votes / n_hashes
    │
    ▼
Thresholding               discard if confidence < MIN_CONFIDENCE
    │
    ▼
Ranked Results             [(song_id, confidence), ...]
```

---

## 📜 Scripts Reference

### `scripts/insert_songs.py`
Scans `songs/` for audio files and inserts each song name into the `songs` table. Safe to re-run — skips duplicates.

```bash
python scripts/insert_songs.py
```

---

### `scripts/fingerprint_songs.py`
For each song in the DB, fingerprints it and stores all hashes. Skips songs already fingerprinted.

```bash
python scripts/fingerprint_songs.py
```

---

### `scripts/create_samples.py`
Generates short test clips from each song in `songs/`:

- **Clean clips** → `songs/samples/SongName_sample_N.wav`
- **Noisy clips** → `songs/samples_noisy/SongName_sample_N_<level>.wav`

| Setting | Value |
|---|---|
| Clip duration | 4 seconds |
| Offsets | 15, 30, 45, 60, 90, 120, 150 s |
| Noise levels | light (SNR 20 dB), medium (SNR 10 dB), heavy (SNR 5 dB) |

```bash
python scripts/create_samples.py
```

---

### `scripts/match_song.py`
Match a single audio file against the fingerprint database. Works with any audio format supported by librosa.

```bash
python scripts/match_song.py <path_to_audio>

# Examples
python scripts/match_song.py songs/samples/Neelothi_sample_2.wav
python scripts/match_song.py songs/samples_noisy/Neelothi_sample_3_heavy.wav
python scripts/match_song.py songs/UnknownSong.mp3
```

Output:
```
Query : songs/samples/Neelothi_sample_2.wav

Hashes generated : 2847
Matches found    : 3

  Rank   Song                                Confidence
  -------------------------------------------------------
  1      Neelothi                                0.1823  <-- BEST MATCH
  2      Cheenikkallu                            0.0031
  3      Mannichiru                              0.0021
```

---

### `scripts/evaluate.py`
Batch evaluation across all clean and noisy samples. Reports Top-1 accuracy, Top-3 accuracy, and no-match rate per noise level with a final comparison table.

```bash
python scripts/evaluate.py
```

Output sections:
- **CLEAN SAMPLES** — baseline accuracy
- **NOISY SAMPLES — LIGHT** — SNR 20 dB
- **NOISY SAMPLES — MEDIUM** — SNR 10 dB
- **NOISY SAMPLES — HEAVY** — SNR 5 dB
- **OVERALL COMPARISON** — side-by-side accuracy table

---

### `scripts/drop_tables.py`
Drops all tables in the database (requires manual confirmation). Use to reset and re-run the pipeline from scratch.

```bash
python scripts/drop_tables.py
# → Type 'yes' to confirm
```

---

## 🔧 Configuration

### `AudioFingerprinter` defaults (`fingerprint.py`)

| Parameter | Default | Description |
|---|---|---|
| `sample_rate` | 22050 Hz | Target sample rate |
| `n_fft` | 2048 | STFT window size |
| `hop_length` | 512 | STFT hop (frame step) |
| `fan_value` | 7 | Target peaks per anchor |
| `delta_t_min` | 1 | Min frame gap anchor→target |
| `delta_t_max` | 200 | Max frame gap (~4.7 s) |
| `freq_bin_size` | 5 | Frequency quantisation |
| `amp_min` | -40 dB | Peak amplitude threshold |

### Matching defaults (`evaluate.py`, `match_song.py`)

| Parameter | Default | Description |
|---|---|---|
| `MIN_CONFIDENCE` | 0.02 | Reject matches below this normalised score |
| `TOP_N` | 5 | Number of candidates returned |

---

## 📊 Evaluation

### Matching Algorithm

1. **Batch DB query** — all query hashes fetched in a single `WHERE hash_value IN (...)` call, chunked at 900 values to respect SQLite limits
2. **Offset-alignment voting** — `votes[song_id][db_offset − query_offset] += 1`; robust to where in the song the clip was taken
3. **Score normalisation** — `confidence = peak_votes / total_query_hashes` → comparable 0–1 scale
4. **Thresholding** — results below `MIN_CONFIDENCE` are discarded as no-match

### Expected Performance

| Condition | Top-1 Accuracy |
|---|---|
| Clean samples | ~95–100% |
| Light noise (SNR 20 dB) | ~90–98% |
| Medium noise (SNR 10 dB) | ~75–90% |
| Heavy noise (SNR 5 dB) | ~50–75% |

---

## 🔐 Robustness Features

- ✅ Volume changes — amplitude normalisation before hashing
- ✅ Recording offset — offset-alignment voting is position-independent
- ✅ Compression artifacts — high fan-out creates overlapping, redundant hashes
- ✅ Background noise — amplitude thresholding retains only dominant spectral peaks
- ✅ Device/sample-rate variation — all audio resampled to 22050 Hz

---

## 🐳 Docker Setup

```bash
# Build
docker build -t music-detection .

# Run Jupyter Lab
docker run -p 8888:8888 -v $(pwd):/workspace music-detection
```

Access at: `http://localhost:8888`

---

## 📚 References

- Wang, A. (2003). *An Industrial-Strength Audio Search Algorithm.* Shazam Entertainment.
- Librosa documentation: https://librosa.org/
- STFT: https://en.wikipedia.org/wiki/Short-time_Fourier_transform

---

## 📄 License

Educational project — S4 MFC Course.

---

**Last Updated**: March 2026 · **Python**: 3.10+ · **Sample Rate**: 22050 Hz · **FFT Size**: 2048


## 🎯 Overview

This system implements industry-standard Shazam-style audio fingerprinting featuring:

- **Audio Loading & Preprocessing**: Mono conversion, resampling to 22050 Hz, amplitude normalization
- **Spectral Analysis**: STFT computation with Hann window and configurable FFT size
- **Peak Detection**: Local maxima extraction with adaptive thresholding
- **Constellation Mapping**: Time-frequency landmark extraction
- **Fingerprint Generation**: Anchor-target pairing with deterministic SHA-256 hashing
- **Temporal Locality**: Configurable fan-out and time delta windows

---

## 🔥 Why Does Audio Need Preprocessing?

Real-world audio recordings are rarely clean. Noise and distortion are introduced at multiple stages of the recording chain, making raw audio unreliable for direct fingerprinting. Common sources include:

| Source | Description |
|---|---|
| **Cheap Microphones** | Low-quality capsules introduce self-noise and frequency coloration, adding a constant noise floor to recordings |
| **ADC Imperfections** | Analog-to-Digital Converters introduce quantization noise, clipping, and non-linearity errors during the analog → digital conversion |
| **Recording Devices** | Interference from device electronics, power supply hum (50/60 Hz), and poor shielding bleed into the audio signal |
| **Compression Artifacts** | Lossy codecs (MP3, AAC) discard high-frequency detail and introduce ringing/pre-echo artifacts, altering the spectral content |

> This is common in all recorded audio — even professional recordings contain low-level noise from the environment, equipment, and encoding pipeline.

### 🛠️ How Preprocessing Fixes This

Before fingerprinting, the audio pipeline applies a series of normalisation and conditioning steps to strip away noise and produce a consistent, comparable signal:

1. **Mono Conversion** — Collapses stereo channels into a single channel. Eliminates channel imbalance and halves data size without losing tonal information relevant to fingerprinting.

2. **Resampling to 22050 Hz** — Standardises the sample rate across all recordings. This ensures fingerprints generated from different devices (phones, studio mics, web streams) are directly comparable. 22050 Hz is sufficient to capture frequencies up to 11025 Hz — well above the perceptually relevant range.

3. **Amplitude Normalisation** — Scales the waveform so the peak amplitude is ±1. Removes volume differences caused by recording level, gain staging, or playback volume — so a quiet recording and a loud recording of the same song produce the same fingerprint.

4. **Spectral Peak Thresholding** — During STFT analysis, a percentile threshold (default: 75th percentile) discards low-energy spectral bins that are more likely to contain noise than musical content. Only the strongest, most stable peaks are retained as landmarks.

Together, these steps ensure the fingerprinting system focuses on the **stable spectral structure** of a song rather than the noise layered on top of it — making identification robust even when the query audio is low-quality or recorded in a noisy environment.
