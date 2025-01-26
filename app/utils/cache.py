from datetime import datetime, timedelta
from typing import Any, Optional


class Cache:
    """Simple cache implementation with expiration time."""

    def __init__(self, expiration_time: int = 3600) -> None:
        self._cache = {}
        self.expiration_time = expiration_time

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key (str): The key to get the value for.

        Returns:
            Optional[Any]: The value if it exists and is not expired, otherwise None.
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.expiration_time):
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key (str): The key to set the value for.
            value (Any): The value to set.
        """
        self._cache[key] = (value, datetime.now())

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def remove_expired(self) -> None:
        """Remove expired keys from the cache."""
        now = datetime.now()
        expired_keys = [
            key
            for key, (_, timestamp) in self._cache.items()
            if now - timestamp >= timedelta(seconds=self.expiration_time)
        ]

        for key in expired_keys:
            del self._cache[key]
