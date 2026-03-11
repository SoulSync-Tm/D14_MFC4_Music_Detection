"""app/utils/audio.py — Audio file utilities."""

import os
from collections import defaultdict, deque

import numpy as np

from app.core.fingerprint import AudioFingerprinter

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def find_audio_file(directory: str, song_name: str) -> str | None:
    """Return the full path to *song_name* inside *directory*, or None if not found."""
    for ext in SUPPORTED_EXTENSIONS:
        path = os.path.join(directory, song_name + ext)
        if os.path.isfile(path):
            return path
    return None


def list_audio_files(directory: str) -> list[str]:
    """Return sorted list of audio filenames (not full paths) in *directory*."""
    return sorted(
        f
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )


def bytes_to_pcm(data: bytes) -> np.ndarray:
    """Convert raw int16 little-endian PCM bytes to float32 in the range [-1, 1]."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def build_hashes(
    fingerprinter: AudioFingerprinter,
    peaks: list[tuple[int, int]],
    peak_buffer: deque[tuple[int, int]],
    last_processed_frame: int,
    delta_t_max: int,
) -> list[tuple[int, int]]:
    """Generate landmark hashes by combining cross-window pairs (peak_buffer x
    new_peaks) with within-window pairs from new_peaks alone.

    Returns a list of (hash_value, time_frame) tuples.
    """
    new_peaks = [p for p in peaks if p[1] > last_processed_frame]
    hashes: list[tuple[int, int]] = []

    for f1, t1 in peak_buffer:
        for f2, t2 in new_peaks:
            delta = t2 - t1
            if 1 <= delta <= delta_t_max:
                h = fingerprinter._make_hash(f1, f2, delta)
                if h:
                    hashes.append((h, t1))
    hashes.extend(fingerprinter.generate_hashes(new_peaks))
    return hashes


def offset_vote(
    matches: list[tuple[int, int, int]], query_hashes: list[tuple[int, int]]
) -> dict[int, dict[int, int]]:
    """
    matches: [(hash, song_id, db_time)]
    query_hashes: [(hash, query_time)]
    """

    query_time_map = {h: t for h, t in query_hashes}

    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for hv, song_id, db_time in matches:
        if hv not in query_time_map:
            continue

        query_time = query_time_map[hv]
        offset = db_time - query_time

        votes[song_id][offset] += 1

    return votes


def find_best_match(
    votes: dict[int, dict[int, int]],
) -> tuple[int | None, int | None, int]:
    best_song = None
    best_offset = None
    best_votes = 0

    for song_id, offsets in votes.items():
        offset, count = max(offsets.items(), key=lambda x: x[1])

        if count > best_votes:
            best_votes = count
            best_song = song_id
            best_offset = offset

    return best_song, best_offset, best_votes
