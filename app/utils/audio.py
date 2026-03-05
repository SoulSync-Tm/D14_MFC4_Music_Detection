"""app/utils/audio.py — Audio file utilities."""

import os

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
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )
