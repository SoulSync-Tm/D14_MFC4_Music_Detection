"""
app/core/matcher.py — Matching primitives, consensus voting, and worker threads
===============================================================================

Public API
----------
bandpass(y, sr, low=300, high=3400) -> np.ndarray
    4th-order Butterworth bandpass filter (300–3400 Hz).  Used by
    ``AudioFingerprinter`` when ``is_phone_mode=True`` to simulate telephone
    / GSM channel conditions.

fingerprint_only(fp, y) -> (hashes, hash_to_query_times)
    Pure-CPU step: STFT → peaks → 64-bit landmark hashes.
    No Redis I/O — safe to run in a thread executor alongside async code.

score_matches(hashes, hash_to_query_times, db_rows) -> (best_id, confidence, offset_bins)
    Pure-CPU voting step.  Aggregates offset-alignment votes from Redis rows,
    returns the highest-confidence song id, its normalised confidence score,
    and the winning time-delta bucket (spectrogram frames) as the playback
    offset.  Convert to seconds: ``offset_bins * hop_length / sample_rate``.

match_audio(r, fp, y) -> (song_id | None, confidence, offset_bins)
    Full pipeline wrapper: fingerprint_only → match_fingerprints_bulk → score_matches.
    Kept for non-async callers (e.g. matcher_worker).


matcher_worker(audio_queue, stop_flag, on_match, on_no_match) -> None
    Thread consumer — dequeues PCM chunks, calls match_audio, fires callbacks.

start_matcher_worker(...) -> threading.Thread
    Convenience wrapper — starts matcher_worker in a daemon thread.
"""

import queue
import threading
import time
from collections import defaultdict
from typing import Any, Callable

import numpy as np
from scipy.signal import butter, lfilter


def bandpass(y: np.ndarray, sr: int, low: int = 300, high: int = 3400) -> np.ndarray:
    """4th-order Butterworth bandpass filter (300–3400 Hz by default)."""
    nyq = sr / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return np.asarray(lfilter(b, a, y))


def fingerprint_only(
    fp: Any, y: np.ndarray
) -> tuple[list[tuple[int, int]], dict[int, list[int]]]:
    """
    Pure-CPU step: spectrogram → peaks → hashes.

    Returns:
        (hashes, hash_to_query_times)
        hashes               : list of (hash_value: int, time_offset: int)
        hash_to_query_times  : dict[int, list[int]]
    """

    S_db = fp.generate_spectrogram(y)
    peaks = fp.find_peaks(S_db)
    hashes = fp.generate_hashes(peaks)

    if not hashes:
        return [], {}

    hash_to_query_times = defaultdict(list)
    for h, t in hashes:
        hash_to_query_times[int(h)].append(t)

    return hashes, hash_to_query_times


def score_matches(
    hashes: list[tuple[int, int]],
    hash_to_query_times: dict[int, list[int]],
    db_rows: list[tuple[int, int, int]],
) -> tuple[int | None, float, int]:
    """
    Vote-and-score step (pure CPU, no I/O).

    Parameters:
        hashes               : list of (hash_value, time_offset)
        hash_to_query_times  : dict[int, list[int]]
        db_rows              : list of (hash_value, song_id, db_time)

    Returns:
        (best_id: int | None, confidence: float, best_offset_bins: int)
        best_offset_bins is the winning delta-time bucket (db_t − query_t) in
        spectrogram frames.  Convert to wall-clock seconds with:
            offset_s = best_offset_bins * hop_length / sample_rate
        e.g. hop_length=256, sample_rate=8000 → multiply by 0.032.
    """
    from app.config import MIN_CONFIDENCE

    if not hashes or not db_rows:
        return None, 0.0, 0

    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for hash_value, song_id, db_t in db_rows:
        for query_t in hash_to_query_times.get(hash_value, []):
            votes[song_id][db_t - query_t] += 1

    scores = {
        sid: max(buckets.values()) / len(hashes) for sid, buckets in votes.items()
    }
    scores = {sid: s for sid, s in scores.items() if s >= MIN_CONFIDENCE}

    if not scores:
        return None, 0.0, 0

    best_id = max(scores, key=lambda k: scores[k])
    best_offset_bins = max(votes[best_id], key=lambda k: votes[best_id][k])
    return best_id, scores[best_id], best_offset_bins


def match_audio(r: Any, fp: Any, y: np.ndarray) -> tuple[int | None, float, int]:
    """
    Fingerprint one audio chunk and return the best matching song.

    Returns:
        (song_id: int | None, confidence: float, best_offset_bins: int)
    """
    from app.db.fingerprint_repo import match_fingerprints_bulk

    hashes, hash_to_query_times = fingerprint_only(fp, y)
    if not hashes:
        return None, 0.0, 0

    hash_values = [int(h) for h, _ in hashes]
    db_rows = match_fingerprints_bulk(r, hash_values)
    return score_matches(hashes, hash_to_query_times, db_rows)


def matcher_worker(
    audio_queue: queue.Queue[Any],
    stop_flag: threading.Event,
    on_match: Callable[[dict[str, Any]], None] | None = None,
    on_no_match: Callable[[], None] | None = None,
) -> None:
    """
    Consume audio chunks, run match_audio, and invoke result callbacks.

    Parameters:
        audio_queue  : queue holding (pcm_array, timestamp) tuples
        stop_flag    : threading.Event — worker exits when set
        on_match     : callable(result: dict) — called on every successful match
                       result keys: name, confidence, start_time, end_time, duration_s
        on_no_match  : callable() — called when a chunk yields no match
    """
    from app.core.fingerprint import AudioFingerprinter
    from app.db.redis import get_connection

    r = get_connection()
    fp = AudioFingerprinter()

    current_song_id: int | None = None
    song_start_time: float | None = None

    while not stop_flag.is_set():
        try:
            y, chunk_start = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        _t0 = time.time()
        best_id, conf, offset_bins = match_audio(r, fp, y)
        match_time = time.time() - _t0
        print(f"match_time = {match_time:.3f}s")
        offset_s = round(offset_bins * fp.hop_length / fp.sample_rate, 2)

        if best_id is not None:
            name = r.hget(f"song:{best_id}", "name") or "Unknown"

            if best_id != current_song_id:
                current_song_id = best_id
                song_start_time = chunk_start

            end_wall = chunk_start + offset_s
            duration_s = end_wall - song_start_time

            if on_match:
                on_match(
                    {
                        "name": name,
                        "confidence": conf,
                        "offset_s": offset_s,
                        "start_time": song_start_time,
                        "end_time": end_wall,
                        "duration_s": duration_s,
                    }
                )
        else:
            current_song_id = None
            song_start_time = None
            if on_no_match:
                on_no_match()

    if hasattr(r, "close"):
        r.close()


def start_matcher_worker(
    audio_queue: queue.Queue[Any],
    stop_flag: threading.Event,
    on_match: Callable[[dict[str, Any]], None] | None = None,
    on_no_match: Callable[[], None] | None = None,
) -> threading.Thread:
    """
    Start matcher_worker in a daemon thread and return the Thread object.

    Example::

        stop = threading.Event()
        t = start_matcher_worker(q, stop, on_match=handle_match)
        # ... later ...
        stop.set()
        t.join()
    """
    t = threading.Thread(
        target=matcher_worker,
        args=(audio_queue, stop_flag, on_match, on_no_match),
        daemon=True,
    )
    t.start()
    return t
