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
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def _normalize_value(raw_value: str) -> str:
    """Normalize a PII value for deterministic hashing (strip whitespace, dashes, parens)."""
    return re.sub(r"[\s\-\(\)]+", "", raw_value.strip()).lower()


class PIIVaultStore:
    """Session-scoped encrypted PII token vault backed by SQLite."""

    def __init__(self, db_path: str | Path, encryption_key: str | bytes):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        if isinstance(encryption_key, str):
            encryption_key = encryption_key.encode() if encryption_key else Fernet.generate_key()
        self._fernet = Fernet(encryption_key)

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
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
                    self._conn.commit()
                    return nonce

            nonce = secrets.token_hex(16)
            self._conn.execute(
                "INSERT OR REPLACE INTO session_context "
                "(session_scope_key, session_nonce, created_at, last_accessed_at, status) "
                "VALUES (?, ?, ?, ?, 'active')",
                (session_scope_key, nonce, now, now),
            )
            self._conn.commit()
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
            self._conn.commit()
            logger.debug("PII vault session closed: scope=%s", session_scope_key)

    def touch_session(self, session_scope_key: str) -> None:
        """Extend session TTL by updating last_accessed_at."""
        with self._lock:
            self._conn.execute(
                "UPDATE session_context SET last_accessed_at = ? "
                "WHERE session_scope_key = ? AND status = 'active'",
                (_now_iso(), session_scope_key),
            )
            self._conn.commit()

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
                self._conn.commit()
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
            self._conn.commit()
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
            self._conn.commit()
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
            self._conn.commit()

        result: dict[str, str] = {}
        for ref_token, ciphertext in rows:
            try:
                result[ref_token] = self._fernet.decrypt(ciphertext).decode("utf-8")
            except InvalidToken:
                logger.warning("PII vault decryption failed for ref=%s", ref_token)
        return result

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
            self._conn.commit()
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
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning("PIIVaultStore close failed: %s", exc)
