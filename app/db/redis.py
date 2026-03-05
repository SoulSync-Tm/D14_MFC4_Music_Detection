"""
app/db/redis.py — Redis connection factory
==========================================
"""

import redis
from app.config import REDIS_HOST, REDIS_PORT, REDIS_DB


def get_connection(
    host: str = REDIS_HOST,
    port: int = REDIS_PORT,
    db: int = REDIS_DB,
) -> redis.Redis:
    """Return a Redis client with decode_responses=True and short socket timeouts."""
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
