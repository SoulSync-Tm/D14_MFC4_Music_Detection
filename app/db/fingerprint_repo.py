"""
app/db/fingerprint_repo.py — All Redis persistence operations
=============================================================

Key schema
----------
songs:counter           STRING   Auto-incrementing integer — the last assigned song id.
song:{id}               HASH     Song data: only the "name" field (filename stem).
song:name:{name}        STRING   Maps a song's filename-stem → its integer id.
song:{id}:fingerprinted STRING   Presence flag; set after fingerprints are stored.
fp:{hash_value}         LIST     Entries "{song_id}:{time_offset}" for every indexed
                                  fingerprint that produced this hash value.
"""

import redis


def create_database(r: redis.Redis) -> None:
    """No-op — Redis requires no schema creation. Kept for API compatibility."""
    pass


def insert_song(r: redis.Redis, song_name: str) -> int:
    """
    Register a new song and return its integer id.

    Writes:
        song:{id}        — Hash with a single "name" field.
        song:name:{name} — Reverse lookup: name → id.
    """
    song_id = r.incr("songs:counter")
    r.hset(f"song:{song_id}", "name", song_name)
    r.set(f"song:name:{song_name}", song_id)
    return song_id


def insert_fingerprints_bulk(r: redis.Redis, song_id: int, hashes: list) -> None:
    """
    Bulk-insert fingerprint hashes into Redis using a pipeline.

    Each hash is appended to the list at ``fp:{hash_value}`` as
    the packed string ``"{song_id}:{time_offset}"``.
    """
    pipe = r.pipeline()
    for h, t in hashes:
        pipe.rpush(f"fp:{int(h)}", f"{int(song_id)}:{int(t)}")
    pipe.execute()


def match_fingerprints_bulk(r: redis.Redis, hash_values: list) -> list:
    """
    Batch-lookup fingerprints for a list of hash values.

    Uses a single Redis pipeline round-trip (one LRANGE per unique hash).

    Returns:
        list of (hash_value: int, song_id: int, time_offset: int)
    """
    if not hash_values:
        return []

    unique_hashes = list(set(int(h) for h in hash_values))

    pipe = r.pipeline()
    for hv in unique_hashes:
        pipe.lrange(f"fp:{hv}", 0, -1)
    raw_results = pipe.execute()

    results = []
    for hv, entries in zip(unique_hashes, raw_results):
        for entry in entries:
            sid_str, t_str = entry.split(":", 1)
            results.append((hv, int(sid_str), int(t_str)))
    return results


def get_all_songs(r: redis.Redis) -> list:
    """
    Return a list of (song_id, song_name) for every registered song.
    Uses a single pipeline round-trip.
    """
    n = int(r.get("songs:counter") or 0)
    if n == 0:
        return []

    pipe = r.pipeline()
    for i in range(1, n + 1):
        pipe.hget(f"song:{i}", "name")
    names = pipe.execute()

    return [(i, name) for i, name in enumerate(names, start=1) if name]


def song_name_from_id(r: redis.Redis, song_id: int) -> str:
    return r.hget(f"song:{song_id}", "name") or "Unknown"
