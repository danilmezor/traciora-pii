"""SQLite-backed encrypted PII token vault.

Maps deterministic reference tokens to Fernet-encrypted raw PII values,
scoped per session. Tokens are stable within a session (same normalized
value always yields the same ref token) but different across sessions.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

from cryptography.fernet import Fernet, InvalidToken

import logging

logger = logging.getLogger(__name__)

# Placeholder -> ref-token prefix and the regex that matches emitted ref
# tokens are both derived from the union of all built-in regex policies,
# so the vault recognizes every category any locale's detector layer can
# produce. Single-policy runtimes still work — they just see a slightly
# broader regex than strictly necessary.
from .policies.loader import (
    build_partial_ref_token_pattern,
    build_placeholder_to_prefix,
    build_ref_token_pattern,
    load_multi_regex_policy,
)

_DEFAULT_REGEX_POLICY = load_multi_regex_policy(["en", "de", "ru"])
_PLACEHOLDER_TO_PREFIX: dict[str, str] = build_placeholder_to_prefix(_DEFAULT_REGEX_POLICY)
REF_TOKEN_PATTERN = build_ref_token_pattern(_DEFAULT_REGEX_POLICY)
PARTIAL_REF_TOKEN_PATTERN = build_partial_ref_token_pattern(_DEFAULT_REGEX_POLICY)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_context (
    session_scope_key TEXT PRIMARY KEY,
    session_nonce     TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    last_accessed_at  TEXT NOT NULL,
    status            TEXT NOT NULL CHECK(status IN ('active','closed','expired'))
);

CREATE TABLE IF NOT EXISTS token_map (
    session_scope_key TEXT NOT NULL,
    ref_token         TEXT NOT NULL,
    pii_type          TEXT NOT NULL,
    normalized_hash   TEXT NOT NULL,
    value_ciphertext  BLOB NOT NULL,
    value_preview     TEXT NOT NULL,
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    source            TEXT NOT NULL,
    PRIMARY KEY (session_scope_key, ref_token),
    UNIQUE (session_scope_key, pii_type, normalized_hash),
    FOREIGN KEY (session_scope_key) REFERENCES session_context(session_scope_key)
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_preview(raw_value: str, pii_type: str) -> str:
    """Create a short, safe audit preview of a PII value."""
    if not raw_value:
        return ""
    v = raw_value.strip()
    if len(v) <= 4:
        return v[0] + "***"
    return v[:3] + "***" + v[-2:]


# Characters that do not change a value's identity: two values that differ
# only by these (and by case) share one token. The known-value pass in the
# virtualizer builds its matchers from the same class.
NORMALIZE_SEPARATORS = r"[\s\-\(\)]"
_SEPARATORS_RE = re.compile(NORMALIZE_SEPARATORS + "+")


def _normalize_value(raw_value: str) -> str:
    """Normalize a PII value for deterministic hashing (strip whitespace, dashes, parens)."""
    return _SEPARATORS_RE.sub("", raw_value.strip()).lower()


class KnownValue(NamedTuple):
    """One value a session has tokenized: its token, placeholder and raw value."""

    ref_token: str
    pii_type: str
    raw_value: str


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class KnownValues:
    """A session's tokenized values, oldest first.

    The vault returns the same object until the session gains a token, so a
    caller can key derived data (such as a compiled matcher) on the object
    weakly: it is dropped with the vault's cache entry.
    """

    version: tuple[int, int]
    values: tuple[KnownValue, ...]


_NO_KNOWN_VALUES = KnownValues((0, 0), ())


class PIIVaultStore:
    """Session-scoped encrypted PII token vault backed by SQLite."""

    def __init__(self, db_path: str | Path, encryption_key: str | bytes):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-entrant so the thread that holds a batch can call the token
        # methods; other threads wait until the batch commits or rolls back.
        self._lock = threading.RLock()
        self._batch_depth = 0
        # Decrypted values per active session, so the known-value pass does not
        # decrypt the whole session on every call. Least recently used first.
        self._known: OrderedDict[str, KnownValues] = OrderedDict()

        if isinstance(encryption_key, str):
            encryption_key = encryption_key.encode() if encryption_key else Fernet.generate_key()
        self._fernet = Fernet(encryption_key)

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Hold the vault for one transaction: commit once on exit, roll back on error.

        Every write made inside the block by this thread defers its commit, so a
        text with hundreds of values pays one commit, and a failure part-way leaves
        none of the block's tokens behind.
        """
        with self._lock:
            self._batch_depth += 1
            try:
                yield
            except BaseException:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    self._conn.rollback()
                    # A rolled-back row may have been cached, and its rowid
                    # can be reused by the next insert.
                    self._known.clear()
                raise
            self._batch_depth -= 1
            self._commit()

    def _commit(self) -> None:
        if self._batch_depth == 0:
            self._conn.commit()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def open_session(self, session_scope_key: str) -> str:
        """Open or resume a vault session. Returns the session nonce."""
        with self._lock:
            now = _now_iso()
            row = self._conn.execute(
                "SELECT session_nonce, status FROM session_context WHERE session_scope_key = ?",
                (session_scope_key,),
            ).fetchone()

            if row is not None:
                nonce, status = row
                if status == "active":
                    self._conn.execute(
                        "UPDATE session_context SET last_accessed_at = ? WHERE session_scope_key = ?",
                        (now, session_scope_key),
                    )
                    self._commit()
                    return nonce

            nonce = secrets.token_hex(16)
            self._conn.execute(
                "INSERT OR REPLACE INTO session_context "
                "(session_scope_key, session_nonce, created_at, last_accessed_at, status) "
                "VALUES (?, ?, ?, ?, 'active')",
                (session_scope_key, nonce, now, now),
            )
            self._commit()
            logger.debug("PII vault session opened: scope=%s", session_scope_key)
            return nonce

    def close_session(self, session_scope_key: str) -> None:
        """Mark session as closed. Tokens remain for audit but cannot be resolved."""
        with self._lock:
            self._conn.execute(
                "UPDATE session_context SET status = 'closed', last_accessed_at = ? "
                "WHERE session_scope_key = ?",
                (_now_iso(), session_scope_key),
            )
            self._commit()
            self._known.pop(session_scope_key, None)
            logger.debug("PII vault session closed: scope=%s", session_scope_key)

    def touch_session(self, session_scope_key: str) -> None:
        """Extend session TTL by updating last_accessed_at."""
        with self._lock:
            self._conn.execute(
                "UPDATE session_context SET last_accessed_at = ? "
                "WHERE session_scope_key = ? AND status = 'active'",
                (_now_iso(), session_scope_key),
            )
            self._commit()

    # ------------------------------------------------------------------
    # Token CRUD
    # ------------------------------------------------------------------

    def get_or_create_token(
        self,
        session_scope_key: str,
        pii_type: str,
        raw_value: str,
        source: str = "user_input",
        *,
        normalization_key: str | None = None,
    ) -> str:
        """Return a deterministic ref token for (session, pii_type, normalized_value).

        ``raw_value`` is what gets stored in the encrypted ciphertext — i.e.
        what a later ``resolve_token`` returns. ``normalization_key``, when
        provided, is what derives ``normalized_hash``. This split matters for
        detectors that emit a canonical lemma distinct from the surface span:
        e.g. Russian names inflect across grammatical cases on the surface
        but pymorphy3 normalizes them to a single nominative lemma, so the
        lemma is the correct identity key for cross-turn collapse. When
        ``normalization_key`` is ``None`` or empty the hash falls back to
        ``raw_value`` — preserving the original surface-based behavior for
        regex-layer findings whose surface is already stable across turns.

        If a token already exists for this session+type+key, return it
        (storage is not rewritten with the new surface — the first-seen
        surface remains the canonical rehydration value). Otherwise create
        a new token and encrypt ``raw_value``.
        """
        prefix = _PLACEHOLDER_TO_PREFIX.get(pii_type, pii_type.strip("[]").lower())
        key_seed = normalization_key.strip() if normalization_key else ""
        normalized = _normalize_value(key_seed or raw_value)
        norm_hash = hashlib.sha256(
            f"{session_scope_key}:{normalized}".encode()
        ).hexdigest()[:16]

        now = _now_iso()

        with self._lock:
            # Check existing
            row = self._conn.execute(
                "SELECT ref_token FROM token_map "
                "WHERE session_scope_key = ? AND pii_type = ? AND normalized_hash = ?",
                (session_scope_key, pii_type, norm_hash),
            ).fetchone()

            if row is not None:
                ref_token = row[0]
                self._conn.execute(
                    "UPDATE token_map SET last_seen_at = ?, source = ? "
                    "WHERE session_scope_key = ? AND ref_token = ?",
                    (now, source, session_scope_key, ref_token),
                )
                self._conn.execute(
                    "UPDATE session_context SET last_accessed_at = ? "
                    "WHERE session_scope_key = ? AND status = 'active'",
                    (now, session_scope_key),
                )
                self._commit()
                return ref_token

            # Generate new short id
            short_id = self._generate_short_id(prefix)
            ref_token = f"{prefix}_ref:{short_id}"

            ciphertext = self._fernet.encrypt(raw_value.encode("utf-8"))
            preview = _make_preview(raw_value, pii_type)

            self._conn.execute(
                "INSERT INTO token_map "
                "(session_scope_key, ref_token, pii_type, normalized_hash, "
                " value_ciphertext, value_preview, first_seen_at, last_seen_at, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_scope_key, ref_token, pii_type, norm_hash,
                 ciphertext, preview, now, now, source),
            )
            self._conn.execute(
                "UPDATE session_context SET last_accessed_at = ? "
                "WHERE session_scope_key = ? AND status = 'active'",
                (now, session_scope_key),
            )
            self._commit()
            logger.debug(
                "PII vault token created: type=%s preview=%s ref=%s",
                pii_type, preview, ref_token,
            )
            return ref_token

    def resolve_token(self, session_scope_key: str, ref_token: str) -> str | None:
        """Resolve a ref token to its raw value. Returns None if not found or session inactive."""
        with self._lock:
            row = self._conn.execute(
                "SELECT tm.value_ciphertext FROM token_map tm "
                "JOIN session_context sc ON tm.session_scope_key = sc.session_scope_key "
                "WHERE tm.session_scope_key = ? AND tm.ref_token = ? AND sc.status = 'active'",
                (session_scope_key, ref_token),
            ).fetchone()

            if row is None:
                return None

            self._conn.execute(
                "UPDATE session_context SET last_accessed_at = ? "
                "WHERE session_scope_key = ? AND status = 'active'",
                (_now_iso(), session_scope_key),
            )
            self._commit()
            ciphertext = row[0]

        try:
            return self._fernet.decrypt(ciphertext).decode("utf-8")
        except InvalidToken:
            logger.warning("PII vault decryption failed for ref=%s", ref_token)
            return None

    def resolve_all_tokens(self, session_scope_key: str) -> dict[str, str]:
        """Return all ref_token → raw_value mappings for an active session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT tm.ref_token, tm.value_ciphertext FROM token_map tm "
                "JOIN session_context sc ON tm.session_scope_key = sc.session_scope_key "
                "WHERE tm.session_scope_key = ? AND sc.status = 'active'",
                (session_scope_key,),
            ).fetchall()

            self._conn.execute(
                "UPDATE session_context SET last_accessed_at = ? "
                "WHERE session_scope_key = ? AND status = 'active'",
                (_now_iso(), session_scope_key),
            )
            self._commit()

        result: dict[str, str] = {}
        for ref_token, ciphertext in rows:
            try:
                result[ref_token] = self._fernet.decrypt(ciphertext).decode("utf-8")
            except InvalidToken:
                logger.warning("PII vault decryption failed for ref=%s", ref_token)
        return result

    # Sessions whose decrypted values stay in memory at once.
    _KNOWN_CACHE_SESSIONS = 256

    def known_values(self, session_scope_key: str) -> KnownValues:
        """Return every value an active session has tokenized, oldest first.

        The decrypted values are cached per session. Each call costs one
        indexed query on the session's rows; only rows added since the last
        call are decrypted. A closed or expired session has no known values.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT sc.status, COUNT(tm.ref_token), COALESCE(MAX(tm.rowid), 0) "
                "FROM session_context sc "
                "LEFT JOIN token_map tm ON tm.session_scope_key = sc.session_scope_key "
                "WHERE sc.session_scope_key = ?",
                (session_scope_key,),
            ).fetchone()
            status, count, max_rowid = row if row is not None else (None, 0, 0)
            if status != "active" or not count:
                self._known.pop(session_scope_key, None)
                return _NO_KNOWN_VALUES
            version = (count, max_rowid)
            cached = self._known.get(session_scope_key)
            if cached is not None and cached.version == version:
                self._known.move_to_end(session_scope_key)
                return cached

            # Rows are only ever added, so new rows sit above the cached max
            # rowid. Anything else (a rollback, an outside delete) reloads.
            query = (
                "SELECT ref_token, pii_type, value_ciphertext FROM token_map "
                "WHERE session_scope_key = ? AND rowid > ? ORDER BY rowid"
            )
            values: list[KnownValue] = []
            rows = None
            if cached is not None and cached.version[1] < max_rowid:
                rows = self._conn.execute(query, (session_scope_key, cached.version[1])).fetchall()
                if cached.version[0] + len(rows) == count:
                    values = list(cached.values)
                else:
                    rows = None
            if rows is None:
                rows = self._conn.execute(query, (session_scope_key, 0)).fetchall()
            for ref_token, pii_type, ciphertext in rows:
                try:
                    raw = self._fernet.decrypt(ciphertext).decode("utf-8")
                except InvalidToken:
                    logger.warning("PII vault decryption failed for ref=%s", ref_token)
                    continue
                values.append(KnownValue(ref_token, pii_type, raw))
            known = KnownValues(version, tuple(values))
            self._known[session_scope_key] = known
            self._known.move_to_end(session_scope_key)
            while len(self._known) > self._KNOWN_CACHE_SESSIONS:
                self._known.popitem(last=False)
            return known

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_expired(self, ttl_hours: int = 24) -> int:
        """Expire sessions older than ttl_hours and return count of expired sessions."""
        with self._lock:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat()
            cursor = self._conn.execute(
                "UPDATE session_context SET status = 'expired' "
                "WHERE status = 'active' AND last_accessed_at < ?",
                (cutoff,),
            )
            count = cursor.rowcount
            self._commit()
            if count:
                self._known.clear()
        if count:
            logger.info("PII vault expired %d sessions (ttl=%dh)", count, ttl_hours)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_short_id(self, prefix: str) -> str:
        """Generate a short unique id: 2 letter chars from prefix + 8 hex chars."""
        prefix_letters = re.sub(r"[^a-z]", "", prefix.lower())[:2].ljust(2, "x")
        hex_part = secrets.token_hex(4)
        return f"{prefix_letters}_{hex_part}"

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._known.clear()
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning("PIIVaultStore close failed: %s", exc)
