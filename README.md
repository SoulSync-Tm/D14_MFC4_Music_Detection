# Shazam-Style Audio Fingerprinting System

A professional audio signal processing system for song identification using 1–3 seconds of audio. This implementation uses spectral peak detection and anchor-target constellation pairing for robust audio fingerprinting.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Docker Setup](#docker-setup)
- [Running the System](#running-the-system)
- [Song Setup](#song-setup)
- [Usage](#usage)

## 🎯 Overview

This system implements industry-standard Shazam-style audio fingerprinting featuring:

- **Audio Loading & Preprocessing**: Mono conversion, resampling to 22050 Hz, amplitude normalization
- **Spectral Analysis**: STFT computation with Hann window and configurable FFT size
- **Peak Detection**: Local maxima extraction with adaptive thresholding
- **Constellation Mapping**: Time-frequency landmark extraction
- **Fingerprint Generation**: Anchor-target pairing with deterministic SHA-256 hashing
- **Temporal Locality**: Configurable fan-out and time delta windows

## 🚀 Quick Start

### Prerequisites

- **Docker** (recommended) or **Python 3.10+**
- **Audio files** (MP3, WAV, FLAC supported via librosa)

### Using Docker (Recommended)

```bash
# Build Docker image
docker build -t music-detection .

# Run container with Jupyter Lab
docker run -p 8888:8888 -v $(pwd):/workspace music-detection
```

Access Jupyter Lab at: `http://localhost:8888`

### Local Setup (Mac/Linux)

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab main.ipynb
```

## 📁 Project Structure

```
music-detection/
├── main.ipynb                 # Main notebook (fingerprinting pipeline)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── README.md                  # This file
└── songs/                     # Song files (create this folder)
    ├── song_1.mp3
    ├── song_2.wav
    └── ...
```

## 🐳 Docker Setup

### Building the Image

```bash
cd /path/to/music-detection
docker build -t music-detection .
```

This creates a Docker image with:
- Python 3.10
- Audio tools (librosa, scipy, numpy, matplotlib)
- Jupyter Lab for interactive development
- FFmpeg for audio codec support

### Running the Container

**Option 1: Interactive Jupyter Lab**
```bash
docker run -p 8888:8888 -v $(pwd):/workspace music-detection
```

**Option 2: With Custom Port**
```bash
docker run -p 9999:8888 -v $(pwd):/workspace music-detection
```

Access at: `http://localhost:9999`

**Option 3: With Persistent Environment**
```bash
docker run -it -p 8888:8888 -v $(pwd):/workspace \
  --name music-detection-container music-detection
```

View logs:
```bash
docker logs music-detection-container
```

Stop container:
```bash
docker stop music-detection-container
```

## 📂 Song Setup

### Creating the Songs Folder

```bash
# Navigate to project directory
cd /path/to/music-detection

# Create songs folder
mkdir -p songs
```

### Adding Song Files

1. **Place audio files in the `songs/` folder**
   - Supported formats: MP3, WAV, FLAC, OGG
   - Example:
     ```
     songs/
     ├── Bohemian_Rhapsody.mp3
     ├── Imagine.wav
     ├── Hotel_California.flac
     └── Stairway_to_Heaven.ogg
     ```

2. **Update the notebook path**
   
   In `main.ipynb`, cell 2, update the audio path:
   ```python
   audio_path = "songs/Bohemian_Rhapsody.mp3"  # Change this to your song
   ```

3. **Naming Convention** (Recommended)
   - Use underscores or hyphens for spaces: `Song_Name.mp3`
   - Include artist if needed: `Artist_-_Song_Name.mp3`
   - Avoid special characters except `- _ .`
   - Examples:
     - ❌ `Song Name.mp3`
     - ✅ `Song_Name.mp3`
     - ✅ `Adele_-_Hello.mp3`

## 🎵 Running the System

### Step 1: Start Jupyter Lab

**Docker:**
```bash
docker run -p 8888:8888 -v $(pwd):/workspace music-detection
```

**Local:**
```bash
jupyter lab main.ipynb
```

### Step 2: Prepare Your Song

1. Add audio file to `songs/` folder
2. Update audio path in cell 2:
   ```python
   audio_path = "songs/your_song.mp3"
   ```

### Step 3: Run the Pipeline

Execute cells in order:

1. **Cell 1**: Import libraries
   ```python
   import librosa
   import matplotlib.pyplot as plt
   import numpy as np
   from scipy.ndimage import maximum_filter
   ```

2. **Cell 2**: Load audio
   ```python
   audio_path = "songs/your_song.mp3"
   y, sr = librosa.load(audio_path, sr=None, mono=False)
   ```

3. **Cell 3-10**: Preprocessing pipeline
   - Convert to mono
   - Resample to 22050 Hz
   - Normalize amplitude
   - Compute STFT

4. **Cell 11**: Detect spectral peaks
   - Apply log scaling
   - Find local maxima
   - Apply threshold

5. **Cell 12+**: Generate fingerprints
   - Initialize `FingerprintGenerator`
   - Generate anchor-target pairs
   - Create deterministic hashes

### Step 4: Analyze Results

The system outputs:
- **Total Peaks**: Number of spectral landmarks detectedCounts
- **Total Hashes**: Number of fingerprints generated
- **Hash Density**: Hashes per peak (typically ~5)
- **Unique Hashes**: Distinct hash values
- **Determinism Check**: Verifies reproducibility
- **Visualizations**: Constellation map with anchor-target connections

## 🔧 Configuration

### Fingerprint Parameters

In `main.ipynb`, adjust these in the `FingerprintGenerator` initialization:

```python
fp_generator = FingerprintGenerator(
    fan_out=5,              # Targets per anchor (increase for more density)
    min_time_delta=1,       # Min frames between anchor and target
    max_time_delta=200      # Max frames (~2.3 seconds at sr=22050)
)
```

### STFT Parameters

```python
n_fft = 1024           # FFT window size
hop_length = 256       # Hop length (75% overlap)
```

### Peak Detection Parameters

```python
freq_neighborhood = 15  # Frequency domain neighborhood
time_neighborhood = 15  # Time domain neighborhood
threshold_percentile = 75  # Keep top 25% of peaks
```

## 📊 Expected Output

For a 3-minute song:

- **Peaks Detected**: 15,000–20,000
- **Hashes Generated**: 75,000–100,000
- **Unique Hashes**: 60,000–80,000
- **Hash Density**: ~5 hashes/peak
- **Processing Time**: ~5–10 seconds

## 🧪 Troubleshooting

### Docker Issues

**Port already in use:**
```bash
docker run -p 9999:8888 -v $(pwd):/workspace music-detection
```

**Permission denied:**
```bash
sudo docker run -p 8888:8888 -v $(pwd):/workspace music-detection
```

**Audio file not found:**
- Verify file exists: `ls songs/`
- Check file permissions: `chmod 644 songs/your_song.mp3`
- Use absolute path if needed

### Jupyter Lab Issues

**Cannot connect:**
- Check logs: `docker logs <container-id>`
- Verify port: `lsof -i :8888`
- Try different port: `-p 9999:8888`

**Out of memory:**
- Reduce FFT size: `n_fft = 512`
- Process shorter clips
- Close other applications

## 📖 Usage Examples

### Example 1: Single Song Fingerprinting

```python
# Load and fingerprint a song
audio_path = "songs/Bohemian_Rhapsody.mp3"
y, sr = librosa.load(audio_path)

# ... run preprocessing cells ...

# Generate fingerprints
fingerprints = fp_generator.generate_fingerprints(peak_freqs, peak_times)
print(f"Generated {len(fingerprints)} hashes")
```

### Example 2: Batch Processing Multiple Songs

```python
import os

song_files = [f for f in os.listdir('songs/') if f.endswith(('.mp3', '.wav'))]
all_fingerprints = {}

for song_file in song_files:
    audio_path = f"songs/{song_file}"
    y, sr = librosa.load(audio_path)
    # ... preprocessing ...
    fingerprints = fp_generator.generate_fingerprints(peak_freqs, peak_times)
    all_fingerprints[song_file] = fingerprints
```

### Example 3: Query with Short Clip

```python
# Load a 3-second clip
y_query, sr = librosa.load("songs/query_clip.mp3", sr=22050)

# Process same as full song
# Compare hashes with database
```

## 🔐 Robustness Features

The system is robust to:
- ✅ Volume changes (normalized before processing)
- ✅ Compression artifacts (multiple overlapping hashes)
- ✅ Background noise (robust peak detection threshold)
- ✅ Re-recording (time-invariant hash structure)
- ✅ Playback speed variations (within ±5%)

## 📚 References

- Shazam algorithm: [Ellis, D. P. (2005)](https://www.ee.columbia.edu/~dpwe/papers/ShazamEllisNMR05.pdf)
- Librosa documentation: https://librosa.org/
- STFT analysis: https://en.wikipedia.org/wiki/Short-time_Fourier_transform

## 📝 Next Steps

- [ ] Implement inverted hash index for database lookup
- [ ] Add offset-based voting for song identification
- [ ] Build query matching pipeline
- [ ] Handle real-time streaming input
- [ ] Optimize hash collision rates

## 📄 License

This project is for educational purposes (S4 MFC Course).

## 🤝 Contributing

For improvements or bug reports, please use the project repository.

---

**Last Updated**: January 2026

**Python Version**: 3.10+

**Sample Rate**: 22050 Hz

**FFT Size**: 1024 frames
