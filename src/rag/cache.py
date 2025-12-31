"""
RAG Cache - Caching layer for retrieval and generation

Implements:
- Query result caching
- Document embedding caching
- Response caching
- TTL-based expiration
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta


@dataclass
class CacheEntry:
    """A cached item with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: float
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def hit(self) -> None:
        self.hit_count += 1


class RAGCache:
    """
    Multi-layer caching for RAG pipeline.
    
    Caches:
    - Query expansions
    - Search results
    - Document embeddings
    - Generated responses
    
    Features:
    - TTL-based expiration
    - LRU eviction
    - Hit tracking
    - Cache warming
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl_seconds: int = 3600,  # 1 hour default
        query_ttl_seconds: int = 1800,    # 30 min for queries
        response_ttl_seconds: int = 7200,  # 2 hours for responses
    ):
        self.max_entries = max_entries
        self.default_ttl = default_ttl_seconds
        self.query_ttl = query_ttl_seconds
        self.response_ttl = response_ttl_seconds
        
        # Separate caches for different data types
        self._query_cache: Dict[str, CacheEntry] = {}
        self._search_cache: Dict[str, CacheEntry] = {}
        self._embedding_cache: Dict[str, CacheEntry] = {}
        self._response_cache: Dict[str, CacheEntry] = {}
        
        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    def _generate_key(self, data: Any) -> str:
        """Generate a cache key from data."""
        if isinstance(data, str):
            key_data = data
        else:
            key_data = json.dumps(data, sort_keys=True, default=str)
        
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def _get_from_cache(
        self,
        cache: Dict[str, CacheEntry],
        key: str
    ) -> Optional[Any]:
        """Get item from cache if valid."""
        entry = cache.get(key)
        
        if entry is None:
            self.stats["misses"] += 1
            return None
        
        if entry.is_expired():
            del cache[key]
            self.stats["misses"] += 1
            return None
        
        entry.hit()
        self.stats["hits"] += 1
        return entry.value
    
    def _set_in_cache(
        self,
        cache: Dict[str, CacheEntry],
        key: str,
        value: Any,
        ttl: int
    ) -> None:
        """Set item in cache with TTL."""
        # Evict if at capacity
        if len(cache) >= self.max_entries:
            self._evict_lru(cache)
        
        now = time.time()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl,
        )
        cache[key] = entry
    
    def _evict_lru(self, cache: Dict[str, CacheEntry]) -> None:
        """Evict least recently used entries."""
        if not cache:
            return
        
        # First, remove expired entries
        expired = [k for k, v in cache.items() if v.is_expired()]
        for key in expired:
            del cache[key]
            self.stats["evictions"] += 1
        
        # If still at capacity, remove least hit entries
        if len(cache) >= self.max_entries:
            sorted_entries = sorted(
                cache.items(),
                key=lambda x: (x[1].hit_count, x[1].created_at)
            )
            
            # Remove bottom 10%
            to_remove = max(1, len(sorted_entries) // 10)
            for key, _ in sorted_entries[:to_remove]:
                del cache[key]
                self.stats["evictions"] += 1
    
    # Query expansion cache
    def get_query_expansion(self, query: str) -> Optional[Dict]:
        """Get cached query expansion."""
        key = self._generate_key(f"query:{query}")
        return self._get_from_cache(self._query_cache, key)
    
    def set_query_expansion(self, query: str, expansion: Dict) -> None:
        """Cache query expansion."""
        key = self._generate_key(f"query:{query}")
        self._set_in_cache(self._query_cache, key, expansion, self.query_ttl)
    
    # Search results cache
    def get_search_results(self, query: str) -> Optional[List]:
        """Get cached search results."""
        key = self._generate_key(f"search:{query}")
        return self._get_from_cache(self._search_cache, key)
    
    def set_search_results(self, query: str, results: List) -> None:
        """Cache search results."""
        key = self._generate_key(f"search:{query}")
        self._set_in_cache(self._search_cache, key, results, self.query_ttl)
    
    # Embedding cache
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._generate_key(f"embed:{text[:500]}")  # Limit key size
        return self._get_from_cache(self._embedding_cache, key)
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding."""
        key = self._generate_key(f"embed:{text[:500]}")
        self._set_in_cache(self._embedding_cache, key, embedding, self.default_ttl)
    
    # Response cache
    def get_response(self, query: str, sources_hash: str) -> Optional[Dict]:
        """Get cached response."""
        key = self._generate_key(f"response:{query}:{sources_hash}")
        return self._get_from_cache(self._response_cache, key)
    
    def set_response(self, query: str, sources_hash: str, response: Dict) -> None:
        """Cache response."""
        key = self._generate_key(f"response:{query}:{sources_hash}")
        self._set_in_cache(self._response_cache, key, response, self.response_ttl)
    
    # Utility methods
    def clear_all(self) -> None:
        """Clear all caches."""
        self._query_cache.clear()
        self._search_cache.clear()
        self._embedding_cache.clear()
        self._response_cache.clear()
    
    def clear_expired(self) -> int:
        """Clear all expired entries. Returns count cleared."""
        cleared = 0
        
        for cache in [
            self._query_cache,
            self._search_cache,
            self._embedding_cache,
            self._response_cache
        ]:
            expired = [k for k, v in cache.items() if v.is_expired()]
            for key in expired:
                del cache[key]
                cleared += 1
        
        return cleared
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_entries = (
            len(self._query_cache) +
            len(self._search_cache) +
            len(self._embedding_cache) +
            len(self._response_cache)
        )
        
        hit_rate = 0.0
        total_requests = self.stats["hits"] + self.stats["misses"]
        if total_requests > 0:
            hit_rate = self.stats["hits"] / total_requests
        
        return {
            "total_entries": total_entries,
            "query_cache_size": len(self._query_cache),
            "search_cache_size": len(self._search_cache),
            "embedding_cache_size": len(self._embedding_cache),
            "response_cache_size": len(self._response_cache),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": hit_rate,
        }

