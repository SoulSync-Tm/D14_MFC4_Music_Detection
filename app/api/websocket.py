"""
app/api/websocket.py — Real-time WebSocket streaming endpoint
=============================================================

Endpoint
--------
WS  /ws/stream

Wire protocol
-------------
Client → Server : Raw PCM binary frames, int16 little-endian, mono, 8 kHz.
                  Each frame covers PACKET_DURATION seconds (default 0.5 s,
                  i.e. 4 000 samples = 8 000 bytes).

Server → Client : UTF-8 JSON, one message per processed window.

Pipeline
--------
1. ``bytes_to_pcm`` converts each incoming binary frame to float32 PCM.
2. ``RingBuffer`` accumulates samples and emits sliding windows.
"""

from collections import deque
import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import SAMPLE_RATE, MIN_CONFIDENCE
from app.core.fingerprint import AudioFingerprinter
from app.core.buffer import RingBuffer
from app.utils.audio import bytes_to_pcm, build_hashes, find_best_match, offset_vote
from app.utils.logging import get_logger
from app.db.fingerprint_repo import match_fingerprints_bulk, song_name_from_id
from app.db.redis import get_connection

ws_router = APIRouter()

logger = get_logger(__name__)
logger.setLevel(10)  # DEBUG


@ws_router.websocket("/ws/stream")
async def stream_audio(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket connection accepted from %s", websocket.client)
    buf: RingBuffer = RingBuffer()
    audioFingerprinter: AudioFingerprinter = AudioFingerprinter(sample_rate=SAMPLE_RATE)
    r = get_connection()

    chunk_count = 0
    window_count = 0
    sample_counter = 0
    STEP_SAMPLES = int(0.3 * 8000)
    peak_buffer: deque[tuple[int, int]] = deque()  # stores (freq, time_frame)
    last_processed_frame = -1
    DELTA_T_MAX = 80

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk_count += 1
            pcm = bytes_to_pcm(data)
            buf.append(pcm)
            sample_counter += len(pcm)
            logger.info(
                "chunk=%d samples=%d buffer_index=%d", chunk_count, len(pcm), buf.index
            )

            if buf.is_full:
                logger.info("Buffer is full. Ready to emit windows.")

            if sample_counter >= STEP_SAMPLES:
                # making sure that we emit a window every 0.3 seconds, even if the buffer is not full yet
                sample_counter = 0

                # getting the array buffered in the ring buffer and processing it to generate hashes
                window = buf.get_window()
                # Ignore silence
                # if np.mean(np.abs(window)) < 0.01:
                #     logger.info("Skipping silent window.")
                #     continue
                S_db = audioFingerprinter.generate_spectrogram(window)
                peaks = audioFingerprinter.find_peaks(S_db)

                new_peaks = [p for p in peaks if p[1] > last_processed_frame]
                hashes = build_hashes(
                    audioFingerprinter,
                    peaks,
                    peak_buffer,
                    last_processed_frame,
                    DELTA_T_MAX,
                )
                if peaks:
                    last_processed_frame = peaks[-1][1]
                peak_buffer.extend(new_peaks)
                while peak_buffer and (
                    last_processed_frame - peak_buffer[0][1] > DELTA_T_MAX
                ):
                    peak_buffer.popleft()

                if hashes:
                    # find matches in the database and send back the best match with confidence
                    hash_values = [h[0] for h in hashes]
                    matches = match_fingerprints_bulk(r, hash_values)
                    votes = offset_vote(matches, hashes)
                    best_song, best_offset, best_votes = find_best_match(votes)
                    confidence = best_votes / len(hashes) if hashes else 0.0
                    if best_offset is not None:
                        offset_seconds = (
                            best_offset * audioFingerprinter.hop_length / SAMPLE_RATE
                        )
                    else:
                        offset_seconds = 0.0

                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    song_name = (
                        song_name_from_id(r, best_song) if best_song else "Unknown"
                    )
                    # Require minimum votes for reliability
                    if best_votes < 10:
                        logger.info("Skipping match: not enough votes (%d)", best_votes)
                        logger.info(
                            "Matches found: %d,song_name: %s, but below threshold. Confidence=%.2f",
                            len(matches),
                            song_name,
                            confidence,
                        )
                        continue
                    if confidence > MIN_CONFIDENCE:
                        await websocket.send_json(
                            {
                                "matched": True,
                                "name": song_name,
                                "confidence": confidence,
                                "offset_s": offset_seconds,
                                "timestamp": timestamp,
                            }
                        )
                        logger.info(
                            "Match found: matches=%d song_id=%s offset=%d votes=%d confidence=%.2f name=%s",
                            len(matches),
                            best_song,
                            best_offset,
                            best_votes,
                            confidence,
                            song_name,
                        )
                        return

                    pass
                logger.info(
                    "window=%d peaks=%d hashes=%d",
                    window_count,
                    len(peaks),
                    len(hashes),
                )

                await websocket.send_json({"match": False})
                window_count += 1

    except WebSocketDisconnect:
        logger.info(
            "Client disconnected — chunks=%d windows=%d", chunk_count, window_count
        )
    except Exception as exc:
        logger.exception("[chunk %d] Unhandled error: %s", chunk_count, exc)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
