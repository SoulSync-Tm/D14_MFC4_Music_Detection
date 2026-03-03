import sqlite3
from collections import defaultdict


def match_sample(db_path, sample_hashes):
    """
    Perform offset voting match.

    sample_hashes: list of (hash_value, sample_time)

    Returns:
        best_song_id,
        best_offset,
        best_score
    """

    if not sample_hashes:
        return None, None, 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract unique hash values
    hash_values = list(set(h for h, _ in sample_hashes))

    # Build lookup: hash -> list of sample times
    sample_time_map = defaultdict(list)
    for h, t in sample_hashes:
        sample_time_map[h].append(t)

    # Query DB for all matching hashes
    placeholders = ",".join("?" for _ in hash_values)

    query = f"""
        SELECT hash_value, song_id, time_offset
        FROM fingerprints
        WHERE hash_value IN ({placeholders})
    """

    cursor.execute(query, hash_values)
    db_rows = cursor.fetchall()

    conn.close()

    # Voting dictionary
    votes = defaultdict(int)

    for hash_value, song_id, db_time in db_rows:
        for sample_time in sample_time_map[hash_value]:
            delta = db_time - sample_time
            votes[(song_id, delta)] += 1

    if not votes:
        return None, None, 0

    # Find strongest cluster
    (best_song_id, best_offset), best_score = max(
        votes.items(),
        key=lambda x: x[1]
    )

    return best_song_id, best_offset, best_score