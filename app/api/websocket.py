"""
app/api/websocket.py — WebSocket streaming endpoint
=====================================================

WebSocket /ws/stream

Protocol
--------
Client → Server : raw PCM bytes, 16-bit signed little-endian, mono, 8 kHz.
                  Each message = one audio chunk (CHUNK_DURATION seconds worth).

Server → Client : JSON messages:
    {"matched": true,  "name": "...", "confidence": 0.23,
     "start_time": "14:05:01", "end_time": "14:05:03", "duration": "00:02"}
    {"matched": false}
"""

import time
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import MIN_CONFIDENCE, SAMPLE_RATE
from app.core.fingerprint import AudioFingerprinter
from app.db.fingerprint_repo import match_fingerprints_bulk
from app.db.redis import get_connection

ws_router = APIRouter()

_fp = AudioFingerprinter()   # shared, stateless


def _match_chunk(r, y: np.ndarray) -> dict:
    S_db   = _fp.generate_spectrogram(y)
    peaks  = _fp.find_peaks(S_db)
    hashes = _fp.generate_hashes(peaks)

    if not hashes:
        return {"matched": False}

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
        return {"matched": False}

    best_id   = max(scores, key=scores.get)
    best_conf = scores[best_id]
    name      = r.hget(f"song:{best_id}", "name") or "Unknown"

    return {"matched": True, "song_id": best_id, "name": name, "confidence": round(best_conf, 4)}


@ws_router.websocket("/ws/stream")
async def stream_audio(websocket: WebSocket):
    """
    Accept raw PCM chunks over WebSocket and return real-time match results.
    """
    await websocket.accept()
    r = get_connection()

    current_song_id: int | None   = None
    song_start_time: float | None = None

    try:
        while True:
            raw = await websocket.receive_bytes()

            # Convert raw bytes (int16 LE) to float32 in [-1, 1]
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            chunk_received = time.time()
            result = _match_chunk(r, pcm)

            if result["matched"]:
                song_id = result["song_id"]
                if song_id != current_song_id:
                    current_song_id = song_id
                    song_start_time = chunk_received

                end_wall   = time.time()
                duration_s = end_wall - song_start_time
                await websocket.send_json({
                    **result,
                    "start_time": time.strftime("%H:%M:%S", time.localtime(song_start_time)),
                    "end_time":   time.strftime("%H:%M:%S", time.localtime(end_wall)),
                    "duration":   f"{int(duration_s // 60):02d}:{int(duration_s % 60):02d}",
                })
            else:
                current_song_id = None
                song_start_time = None
                await websocket.send_json({"matched": False})

    except WebSocketDisconnect:
        pass
