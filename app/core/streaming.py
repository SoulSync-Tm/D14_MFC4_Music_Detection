"""
app/core/streaming.py — Live audio producer / consumer engine
=============================================================

Provides three building blocks used by the stream_audio CLI script and the
WebSocket API endpoint:

  audio_producer(audio_queue, stop_flag)
      Captures microphone input in ``CHUNK_DURATION``-second blocks and puts
      ``(pcm_array, absolute_timestamp)`` tuples onto *audio_queue*.

  match_audio(r, fp, y) -> (song_id | None, confidence: float)
      Fingerprints one audio chunk and returns the best matching song_id and
      a normalised confidence score.

  matcher_worker(audio_queue, stop_flag, on_match, on_no_match)
      Consumes chunks from *audio_queue*, calls match_audio, and invokes
      callbacks so the caller decides how to present results.
"""

import threading
import queue
import time
from collections import defaultdict

import numpy as np
import sounddevice as sd

from app.config import SAMPLE_RATE, CHUNK_DURATION, MIN_CONFIDENCE
from app.core.fingerprint import AudioFingerprinter
from app.db.redis import get_connection
from app.db.fingerprint_repo import match_fingerprints_bulk


# ---------------------------------------------------------------------------
# Audio producer
# ---------------------------------------------------------------------------

def audio_producer(audio_queue: queue.Queue, stop_flag: threading.Event) -> None:
    """
    Capture microphone input and push ``(chunk, timestamp)`` onto *audio_queue*.

    The timestamp is an absolute Unix epoch time computed by adding the ADC
    relative time (``time_info.inputBufferAdcTime``) to the wall-clock instant
    captured just before the InputStream opens.  This gives true audio-hardware
    timing rather than Python scheduling delays.
    """
    blocksize = int(SAMPLE_RATE * CHUNK_DURATION)

    def callback(indata, frames, time_info, status):
        if stop_flag.is_set():
            raise sd.CallbackStop()
        chunk       = indata[:, 0].copy()
        chunk_start = stream_open_wall_time + time_info.inputBufferAdcTime
        try:
            audio_queue.put_nowait((chunk, chunk_start))
        except queue.Full:
            pass  # drop oldest chunk if consumer is lagging

    stream_open_wall_time = time.time()
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=blocksize,
        callback=callback,
    ):
        while not stop_flag.is_set():
            sd.sleep(100)


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def match_audio(r, fp: AudioFingerprinter, y: np.ndarray) -> tuple:
    """
    Fingerprint one audio chunk and return the best matching song.

    Returns:
        (song_id: int | None, confidence: float)
    """
    S_db   = fp.generate_spectrogram(y)
    peaks  = fp.find_peaks(S_db)
    hashes = fp.generate_hashes(peaks)

    if not hashes:
        return None, 0.0

    hash_values         = [int(h) for h, _ in hashes]
    hash_to_query_times = defaultdict(list)
    for h, t in hashes:
        hash_to_query_times[int(h)].append(t)

    db_rows = match_fingerprints_bulk(r, hash_values)

    votes: dict = defaultdict(lambda: defaultdict(int))
    for hash_value, song_id, db_t in db_rows:
        for query_t in hash_to_query_times[hash_value]:
            votes[song_id][db_t - query_t] += 1

    scores = {
        sid: max(buckets.values()) / len(hashes)
        for sid, buckets in votes.items()
    }
    scores = {sid: s for sid, s in scores.items() if s >= MIN_CONFIDENCE}

    if not scores:
        return None, 0.0

    best_id = max(scores, key=scores.get)
    return best_id, scores[best_id]


# ---------------------------------------------------------------------------
# Matching consumer
# ---------------------------------------------------------------------------

def matcher_worker(
    audio_queue: queue.Queue,
    stop_flag:   threading.Event,
    on_match=None,
    on_no_match=None,
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
    r  = get_connection()
    fp = AudioFingerprinter()

    current_song_id: int | None   = None
    song_start_time: float | None = None

    while not stop_flag.is_set():
        try:
            y, chunk_start = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        best_id, conf = match_audio(r, fp, y)

        if best_id is not None:
            name = r.hget(f"song:{best_id}", "name") or "Unknown"

            if best_id != current_song_id:
                current_song_id = best_id
                song_start_time = chunk_start

            end_wall   = time.time()
            duration_s = end_wall - song_start_time

            if on_match:
                on_match({
                    "name":       name,
                    "confidence": conf,
                    "start_time": song_start_time,
                    "end_time":   end_wall,
                    "duration_s": duration_s,
                })
        else:
            current_song_id = None
            song_start_time = None
            if on_no_match:
                on_no_match()

    if hasattr(r, "close"):
        r.close()
