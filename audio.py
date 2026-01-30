import numpy as np
import librosa
import hashlib
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter

def generate_hash(f1, f2, dt):
    return hashlib.sha1(f"{f1}|{f2}|{dt}".encode()).hexdigest()[:20]

class StreamingFingerprinter:
    def __init__(
        self,
        sr=22050,
        buffer_seconds=3,
        n_fft=1024,
        hop_length=256,
        fan_out=5,
        min_dt=1,
        max_dt=50,
        peak_percentile=75,
        freq_neighborhood=15,
        time_neighborhood=15,
    ):
        self.sr = sr
        self.buffer_size = int(buffer_seconds * sr)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fan_out = fan_out
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.peak_percentile = peak_percentile
        self.freq_neighborhood = freq_neighborhood
        self.time_neighborhood = time_neighborhood
        self.peaks = []

    def reset(self):
        self.peaks = []
        self.audio_buffer[:] = 0

    def add_audio(self, chunk):
        self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
        self.audio_buffer[-len(chunk):] = chunk

    def compute_stft(self):
        return np.abs(
            librosa.stft(
                self.audio_buffer,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window="hann",
            )
        )

    def detect_peaks(self, S_mag):
        S_log = np.log1p(S_mag)
        local_max = maximum_filter(
            S_log, size=(self.freq_neighborhood, self.time_neighborhood)
        )
        mask = (S_log == local_max)
        threshold = np.percentile(S_log, self.peak_percentile)
        mask &= (S_log >= threshold)
        freqs, times = np.where(mask)
        return list(zip(freqs, times))

    def generate_hashes(self, peaks):
        hashes = []
        peaks.sort(key=lambda x: x[1])
        for i in range(len(peaks)):
            f1, t1 = peaks[i]
            for j in range(1, self.fan_out + 1):
                if i + j >= len(peaks):
                    break
                f2, t2 = peaks[i + j]
                dt = t2 - t1
                if self.min_dt <= dt <= self.max_dt:
                    hashes.append((generate_hash(f1, f2, dt), t1))
        return hashes

    def fingerprint_window(self, audio):
        self.reset()
        self.add_audio(audio)
        S_mag = self.compute_stft()
        peaks = self.detect_peaks(S_mag)
        return self.generate_hashes(peaks)

def load_audio(path, sr=22050):
    y, s = librosa.load(path, sr=None, mono=False)
    if y.ndim == 2:
        y = np.mean(y, axis=0)
    if s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr)
    return y / np.max(np.abs(y)), sr

# -------------------------------
# BUILD DATABASE
# -------------------------------
db_audio, sr = load_audio("songs/EppadiVandhaayo.mp3")

# Use consistent fingerprinting parameters
db_fp = StreamingFingerprinter(
    sr=sr,
    buffer_seconds=3,
    fan_out=3,              # Reduced for more discriminative hashes
    min_dt=5,               # Increased minimum time gap
    max_dt=50,
    peak_percentile=92,     # Very selective peaks (top 8%)
    freq_neighborhood=25,   # Even larger neighborhood
    time_neighborhood=25,
)

hash_db = defaultdict(list)
hop = int(0.5 * sr)

for i in range(0, len(db_audio), hop):
    win = db_audio[i:i + 3*sr]
    if len(win) < 3*sr:
        break
    for h, t in db_fp.fingerprint_window(win):
        # Store absolute time position in the song
        absolute_t = i // db_fp.hop_length + t
        hash_db[h].append(("EppadiVandhaayo", absolute_t))

print("DB hashes:", sum(len(v) for v in hash_db.values()))
print(f"DB audio duration: {len(db_audio)/sr:.2f} sec")

# -------------------------------
# QUERY (2s sliding windows)
# -------------------------------
query_audio, _ = load_audio("songs/test2.mp3")
print(f"Query audio duration: {len(query_audio)/sr:.2f} sec")

# Use SAME fingerprinting parameters as database
query_fp = StreamingFingerprinter(
    sr=sr,
    buffer_seconds=0.5,     # Shorter window for faster detection
    fan_out=3,              # Reduced for more discriminative hashes
    min_dt=5,               # Increased minimum time gap
    max_dt=50,
    peak_percentile=92,     # Very selective peaks (top 8%)
    freq_neighborhood=25,   # Even larger neighborhood
    time_neighborhood=25,
)

# Shorter windows for faster, more granular detection
WINDOW = int(0.5 * sr)      # 0.5 seconds (reduced from 2s)
HOP = int(0.25 * sr)        # 0.25 seconds (reduced from 0.5s)

# Adjusted thresholds for shorter windows
MIN_MATCH_COUNT = 6            # Fewer hashes in shorter window
MIN_OFFSET_RATIO = 0.25        # Require 25% consensus per window
MIN_ABSOLUTE_VOTES = 10        # Adjusted for shorter windows
MIN_WINDOW_CONSENSUS = 0.10    # At least 10% of windows (more windows now)
MAX_NEGATIVE_OFFSET = -5       # Allow small negative offsets for alignment

best = None
all_matches = []
offset_votes = Counter()  # Track offset consistency across windows

for i in range(0, len(query_audio) - WINDOW, HOP):
    window = query_audio[i:i + WINDOW]
    hashes = query_fp.fingerprint_window(window)
    
    # Calculate absolute time for query hashes
    query_time_offset = i // query_fp.hop_length

    votes = defaultdict(list)
    for h, tq in hashes:
        if h in hash_db:
            for sid, td in hash_db[h]:
                absolute_tq = query_time_offset + tq
                votes[sid].append(td - absolute_tq)

    for sid, offsets in votes.items():
        if len(offsets) < MIN_MATCH_COUNT:
            continue
        cnt = Counter(offsets)
        off, c = cnt.most_common(1)[0]
        if c / len(offsets) >= MIN_OFFSET_RATIO:
            all_matches.append((sid, off, c, len(offsets), i/sr))
            offset_votes[off] += 1  # Count how many windows vote for this offset
            if not best:  # Take first match
                best = (sid, off, c)
            break

# Use the most voted offset across all windows for better accuracy
if offset_votes:
    best_offset, window_count = offset_votes.most_common(1)[0]
    total_windows = len(range(0, len(query_audio) - WINDOW, HOP))
    window_consensus_ratio = window_count / total_windows
    
    # Find the match with this offset that has highest votes
    candidate = max([m for m in all_matches if m[1] == best_offset], 
                    key=lambda x: x[2], default=best)
    
    # Strict validation to prevent false positives
    if candidate:
        # Calculate average vote percentage across all matching windows
        matching_windows = [m for m in all_matches if m[1] == best_offset]
        avg_vote_ratio = np.mean([m[2]/m[3] for m in matching_windows])
        
        # Validation checks
        offset_time = best_offset * query_fp.hop_length / sr
        checks_passed = True
        rejection_reason = None
        
        if candidate[2] < MIN_ABSOLUTE_VOTES:
            checks_passed = False
            rejection_reason = f"Insufficient votes ({candidate[2]} < {MIN_ABSOLUTE_VOTES})"
        elif window_consensus_ratio < MIN_WINDOW_CONSENSUS:
            checks_passed = False
            rejection_reason = f"Low window consensus ({window_consensus_ratio:.1%} < {MIN_WINDOW_CONSENSUS:.0%})"
        elif offset_time < MAX_NEGATIVE_OFFSET:
            checks_passed = False
            rejection_reason = f"Invalid negative offset ({offset_time:.1f}s < {MAX_NEGATIVE_OFFSET}s)"
        elif avg_vote_ratio < 0.30:  # Average vote ratio should be at least 30%
            checks_passed = False
            rejection_reason = f"Low average vote ratio ({avg_vote_ratio:.1%} < 30%)"
        
        if checks_passed:
            best = candidate
        else:
            print(f"\n⚠️  Match rejected: {rejection_reason}")
            best = None

print(f"\nQuery windows scanned: {len(range(0, len(query_audio) - WINDOW, HOP))}")
print(f"Potential matches found: {len(all_matches)}")

if offset_votes and len(offset_votes) > 0:
    print(f"\nTop offset candidates:")
    for off, count in offset_votes.most_common(3):
        vote_pct = count / len(range(0, len(query_audio) - WINDOW, HOP)) * 100
        print(f"  {off*query_fp.hop_length/sr:6.1f}s: {count} windows ({vote_pct:.1f}%)")

if best:
    print("\n✅ STRONG MATCH FOUND")
    print("Song:", best[0])
    print(f"Offset: {best[1]*query_fp.hop_length/sr:.2f} sec")
    print(f"Confidence: {best[2]} votes")
    window_pct = offset_votes[best[1]] / len(range(0, len(query_audio) - WINDOW, HOP)) * 100
    print(f"Consensus: {offset_votes[best[1]]} of {len(range(0, len(query_audio) - WINDOW, HOP))} windows ({window_pct:.1f}%)")
    
    # Calculate and show average vote ratio
    matching_windows = [m for m in all_matches if m[1] == best[1]]
    avg_ratio = np.mean([m[2]/m[3] for m in matching_windows])
    print(f"Avg vote ratio: {avg_ratio:.1%}")
    
    if len(matching_windows) > 1:
        print(f"\nDetailed results (showing offset {best[1]*query_fp.hop_length/sr:.1f}s matches only):")
        for sid, off, votes, total, qtime in matching_windows[:10]:
            print(f"  {qtime:5.1f}s → {votes:3d}/{total:3d} votes ({votes/total*100:5.1f}%)")
else:
    print("\n❌ NO RELIABLE MATCH FOUND")
