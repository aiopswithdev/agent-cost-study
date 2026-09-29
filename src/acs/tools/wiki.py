"""T5 — `Search[entity]` / `Lookup[keyword]` over the MediaWiki Action API.

Semantics mirror ReAct's `wikienv` (5-sentence observations, "Could not find … Similar: …",
cycling Lookup). Every raw API request is cached on disk, which is what guarantees that
identical queries return identical content for every scaffold for the whole study.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .. import config

_HEADER_RE = re.compile(r"^=+.*=+$")


# --------------------------------------------------------------------------- disk cache


class SqliteCache:
    """`cache/wiki.sqlite` — key = sha256 of the raw request, value = response JSON."""

    def __init__(self, path: Path = config.WIKI_CACHE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, ts TEXT)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(endpoint: str, params: dict) -> str:
        blob = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _put(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, ts) VALUES (?, ?, datetime('now'))",
                (key, value),
            )
            self._conn.commit()

    async def get(self, key: str) -> str | None:
        return await asyncio.to_thread(self._get, key)

    async def put(self, key: str, value: str) -> None:
        await asyncio.to_thread(self._put, key, value)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --------------------------------------------------------------------------- page handling


def clean_page(extract: str) -> str:
    """Drop section headers and empty lines; keep paragraphs of more than two words."""
    kept = []
    for line in extract.split("\n"):
        line = line.strip()
        if not line or _HEADER_RE.match(line):
            continue
        if len(line.split()) > 2:
            kept.append(line)
    return "\n".join(kept)


def sentences_of(page: str) -> list[str]:
    """Exactly `wikienv`'s splitting: paragraphs on `\\n`, sentences on `'. '`, re-append `.`."""
    paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
    sentences: list[str] = []
    for p in paragraphs:
        sentences += p.split(". ")
    return [s.strip() + "." for s in sentences if s.strip()]


def get_page_obs(page: str) -> str:
    """Observation = the first five sentences, joined by spaces (`wikienv.get_page_obs`)."""
    return " ".join(sentences_of(page)[:5])


def not_found_observation(entity: str, titles: list[str]) -> str:
    return f"Could not find {entity}. Similar: {titles}."


# --------------------------------------------------------------------------- lookup state


@dataclass
class LookupState:
    """Per-run Lookup cursor. Reset whenever the keyword changes."""

    page_text: str | None = None
    keyword: str | None = None
    results: list[str] = field(default_factory=list)
    cnt: int = 0


def lookup(page_text: str | None, keyword: str, state: LookupState) -> str:
    if page_text is None:
        return "No page loaded. Use Search first."
    if state.keyword != keyword:
        state.keyword = keyword
        state.results = [s for s in sentences_of(page_text) if keyword.lower() in s.lower()]
        state.cnt = 0
    if state.cnt >= len(state.results):
        return "No more results."
    obs = f"(Result {state.cnt + 1} / {len(state.results)}) {state.results[state.cnt]}"
    state.cnt += 1
    return obs


# --------------------------------------------------------------------------- client


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds from a `Retry-After` header (delta-seconds form only), capped."""
    if not value:
        return None
    try:
        return min(max(float(value), 0.0), config.WIKI_BACKOFF_CAP_S)
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    """1, 2, 4, … s capped at WIKI_BACKOFF_CAP_S, with ±25% jitter."""
    base = min(2.0**attempt, config.WIKI_BACKOFF_CAP_S)
    return base * random.uniform(0.75, 1.25)



class WikiClient:
    def __init__(
        self,
        cache: SqliteCache | None = None,
        concurrency: int = config.WIKI_CONCURRENCY,
        endpoint: str = config.WIKI_ENDPOINT,
    ) -> None:
        self.endpoint = endpoint
        self.cache = cache if cache is not None else SqliteCache()
        self._sem = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None
        self.http_requests = 0
        self.throttled = 0  # 429/5xx responses seen (for the progress log / tests)
        # Monotonic time before which no request may start: set by any 429, obeyed by all.
        self._cooldown_until = 0.0

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0, headers={"User-Agent": config.WIKI_USER_AGENT}
            )
        return self._client

    async def _get_json(self, params: dict[str, Any]) -> dict:
        key = SqliteCache.make_key(self.endpoint, params)
        cached = await self.cache.get(key)
        if cached is not None:
            self.cache.hits += 1
            return json.loads(cached)
        self.cache.misses += 1

        last_exc: Exception | None = None
        for attempt in range(config.WIKI_MAX_RETRIES):
            wait = self._cooldown_until - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            retry_after: float | None = None
            try:
                async with self._sem:
                    self.http_requests += 1
                    resp = await self._http().get(self.endpoint, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    self.throttled += 1
                    retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                    raise httpx.HTTPStatusError(
                        f"status {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # retried with backoff on 429/5xx and transport errors
                last_exc = exc
                if attempt == config.WIKI_MAX_RETRIES - 1:
                    raise
                delay = retry_after if retry_after is not None else _backoff(attempt)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                    # Throttled: every request pauses, so the pool stops hammering the API.
                    self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)
                await asyncio.sleep(delay)
                continue
            await self.cache.put(key, json.dumps(data))
            return data
        raise last_exc  # pragma: no cover

    async def _fetch_extract(self, title: str) -> str | None:
        """Step 1: exact page attempt. Returns the cleaned page text, or None."""
        data = await self._get_json(
            {
                "action": "query",
                "titles": title,
                "redirects": 1,
                "prop": "extracts",
                "explaintext": 1,
                "format": "json",
                "formatversion": "2",
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        if page.get("missing") or page.get("invalid"):
            return None
        extract = (page.get("extract") or "").strip()
        if not extract:
            return None
        return extract

    async def _search_titles(self, entity: str, limit: int = 5) -> list[str]:
        """Step 2: full-text search, up to `limit` titles."""
        data = await self._get_json(
            {
                "action": "query",
                "list": "search",
                "srsearch": entity,
                "srlimit": limit,
                "format": "json",
                "formatversion": "2",
            }
        )
        return [h["title"] for h in data.get("query", {}).get("search", [])][:limit]

    async def search(self, entity: str) -> tuple[str, str | None]:
        """Returns `(observation, page_text | None)`, mirroring `wikienv.search_step`."""
        entity = (entity or "").strip()
        if not entity:
            return not_found_observation(entity, []), None

        extract = await self._fetch_extract(entity)
        titles: list[str] | None = None

        if extract is None:
            titles = await self._search_titles(entity)
            if titles and titles[0].lower() == entity.lower():
                extract = await self._fetch_extract(titles[0])
            if extract is None:
                return not_found_observation(entity, titles or []), None

        # Step 3: disambiguation pages are treated as "not found".
        if "may refer to:" in extract[:500]:
            if titles is None:
                titles = await self._search_titles(entity)
            return not_found_observation(entity, titles), None

        page_text = clean_page(extract)
        if not page_text:
            if titles is None:
                titles = await self._search_titles(entity)
            return not_found_observation(entity, titles), None
        return get_page_obs(page_text), page_text

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


_DEFAULT_CLIENT: WikiClient | None = None


def default_client() -> WikiClient:
    """Process-wide client so the HTTP connection pool and the disk cache are shared."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = WikiClient()
    return _DEFAULT_CLIENT


async def close_default_client() -> None:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        await _DEFAULT_CLIENT.aclose()
        _DEFAULT_CLIENT.cache.close()
        _DEFAULT_CLIENT = None
