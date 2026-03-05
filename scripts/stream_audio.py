import queue
import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.redis import get_connection
from app.core.streaming import audio_producer, matcher_worker


def _print_match(result):
    start_str = time.strftime("%H:%M:%S", time.localtime(result["start_time"]))
    end_str   = time.strftime("%H:%M:%S", time.localtime(result["end_time"]))
    d         = result["duration_s"]
    dur       = f"{int(d // 60):02d}:{int(d % 60):02d}"

    print()
    print("=" * 50)
    print("  🎵  SONG IDENTIFIED")
    print("=" * 50)
    print(f"  Name       : {result['name']}")
    print(f"  Confidence : {result['confidence']:.3f}")
    print(f"  Start time : {start_str}")
    print(f"  End time   : {end_str}")
    print(f"  Duration   : {dur}")
    print("=" * 50)


def _print_no_match():
    print("...")


def keyboard_listener(stop_flag):
    print("Press 's' + Enter to stop")
    while True:
        if input().strip().lower() == "s":
            stop_flag.set()
            break


def main():
    r = get_connection()
    if not r.exists("songs:counter"):
        print("No songs in Redis — run insert_songs.py and fingerprint_songs.py first.")
        return

    print("🎤 Listening continuously...")
    print("Processing 2-second chunks.")

    audio_queue = queue.Queue(maxsize=5)
    stop_flag   = threading.Event()

    producer = threading.Thread(
        target=audio_producer, args=(audio_queue, stop_flag), daemon=True
    )
    consumer = threading.Thread(
        target=matcher_worker,
        args=(audio_queue, stop_flag),
        kwargs={"on_match": _print_match, "on_no_match": _print_no_match},
        daemon=True,
    )
    keyboard = threading.Thread(
        target=keyboard_listener, args=(stop_flag,), daemon=True
    )

    producer.start()
    consumer.start()
    keyboard.start()

    producer.join()
    consumer.join()

    print("Stopped.")


if __name__ == "__main__":
    main()
