"""Auditable, content-addressed overrides for immutable dataset videos.

Use case: a dataset is mounted read-only and a few of its video shards are known to be
corrupt. A JSON manifest (``datasets.<block>.video_override_manifests``: dataset leaf ->
absolute manifest path, or null) may nominate a verified replacement file for one exact
logical video path, but only when the mounted file is the manifest's *known bad* object.
Any unrecognised state fails loudly instead of silently changing sample provenance.
Only LeRobot v3.0 datasets are supported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

MANIFEST_VERSION = 2


class VideoOverrideError(RuntimeError):
    """Raised when a video override cannot prove exact provenance."""


@dataclass(frozen=True)
class FileIdentity:
    size: int
    sha256: str


@dataclass(frozen=True)
class ResolvedVideoOverride:
    logical_path: str
    selected_path: Path
    selected_source: str
    expected_good: FileIdentity


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_identity(path: Path, *, label: str) -> FileIdentity:
    if path.is_symlink():
        raise VideoOverrideError(f"{label} must be a regular non-symlink file: {path}")
    if not path.is_file():
        raise VideoOverrideError(f"{label} is missing or not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise VideoOverrideError(f"{label} is empty: {path}")
    return FileIdentity(size=size, sha256=sha256_file(path))


def parse_file_identity(value: Any, *, field: str) -> FileIdentity:
    if not isinstance(value, Mapping):
        raise VideoOverrideError(f"{field} must be an object with size and sha256")
    size = value.get("size")
    digest = value.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise VideoOverrideError(f"{field}.size must be a positive integer, got {size!r}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise VideoOverrideError(f"{field}.sha256 must be a lowercase SHA-256 digest, got {digest!r}")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise VideoOverrideError(f"{field}.sha256 is not hexadecimal: {digest!r}") from exc
    if digest != digest.lower():
        raise VideoOverrideError(f"{field}.sha256 must be lowercase, got {digest!r}")
    return FileIdentity(size=size, sha256=digest)


def normalize_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise VideoOverrideError(f"logical_path must be a non-empty relative POSIX path, got {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise VideoOverrideError(f"logical_path must not be absolute or contain traversal: {value!r}")
    normalized = str(path)
    if normalized != value:
        raise VideoOverrideError(f"logical_path must already be canonical: {value!r} != {normalized!r}")
    return normalized


def load_manifest(path: Path) -> tuple[dict[str, Any], FileIdentity]:
    if path.is_symlink() or not path.is_file():
        raise VideoOverrideError(f"video override manifest must be a regular non-symlink file: {path}")
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VideoOverrideError(f"cannot read video override manifest {path}: {exc}") from exc
    if not payload:
        raise VideoOverrideError(f"video override manifest is empty: {path}")
    if not isinstance(manifest, dict):
        raise VideoOverrideError(f"video override manifest must contain a JSON object: {path}")
    if manifest.get("version") != MANIFEST_VERSION:
        raise VideoOverrideError(
            f"unsupported video override manifest version {manifest.get('version')!r}; " f"expected {MANIFEST_VERSION}"
        )
    return (
        manifest,
        FileIdentity(size=len(payload), sha256=hashlib.sha256(payload).hexdigest()),
    )


def canonical_metadata_digest(records: list[dict[str, Any]]) -> str:
    canonical_records = sorted(
        records,
        key=lambda record: json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    )
    encoded = json.dumps(
        canonical_records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_provenance_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest_leaf_contract(
    configured: Any,
    dataset_leaf_names: Sequence[str],
) -> dict[str, Any] | None:
    """Validate an exact leaf-to-manifest contract.

    ``None`` means overrides are not configured.
    Once the map is present, every mixture leaf must be named explicitly:
    either an absolute manifest path or ``null``.  This makes misspellings and
    newly added leaves fail loudly instead of silently disabling a repair.
    """

    if configured is None:
        return None
    if not isinstance(configured, Mapping) and not hasattr(configured, "items"):
        raise VideoOverrideError(
            "video_override_manifests must map every dataset leaf to an "
            f"absolute manifest path or null, got {configured!r}"
        )
    normalized = {str(key): value for key, value in configured.items()}
    expected = set(dataset_leaf_names)
    actual = set(normalized)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise VideoOverrideError(
            "video_override_manifests leaf contract mismatch: "
            f"missing={missing}, extra={extra}, expected={sorted(expected)}"
        )
    for leaf, value in normalized.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise VideoOverrideError(
                f"video_override_manifests[{leaf!r}] must be an absolute manifest " f"path or null, got {value!r}"
            )
        if not os.path.isabs(value):
            raise VideoOverrideError(f"video_override_manifests[{leaf!r}] must be absolute, got {value!r}")
    return normalized


def normalize_sample_error_policy(value: Any) -> str:
    policy = str(value if value is not None else "resample").strip().lower()
    if policy not in {"resample", "raise"}:
        raise ValueError(
            "sample_error_policy must be 'resample' (retry with another sample) or "
            f"'raise' (fail loudly), got {value!r}"
        )
    return policy


def parse_nonnegative_timestamp(value: Any, *, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise VideoOverrideError(f"Invalid video timestamp {field}={value!r}; {context}.")
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VideoOverrideError(f"Invalid video timestamp {field}={value!r}; {context}.") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise VideoOverrideError(f"Invalid video timestamp {field}={value!r}; {context}.")
    return timestamp


def parse_nonnegative_integer(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise VideoOverrideError(f"Invalid non-negative integer {field}={value!r}; {context}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VideoOverrideError(f"Invalid non-negative integer {field}={value!r}; {context}.") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise VideoOverrideError(f"Invalid non-negative integer {field}={value!r}; {context}.")
    return int(numeric)


def resolve_entry(
    *,
    dataset_path: Path,
    entry: Mapping[str, Any],
) -> ResolvedVideoOverride:
    logical_path = normalize_logical_path(entry.get("logical_path"))
    base_path = dataset_path / logical_path
    expected_good = parse_file_identity(entry.get("expected_good"), field=f"{logical_path}.expected_good")
    known_bad = parse_file_identity(entry.get("known_bad"), field=f"{logical_path}.known_bad")
    if expected_good == known_bad:
        raise VideoOverrideError(f"{logical_path}: expected_good and known_bad identities must differ")

    base_identity = file_identity(base_path, label=f"{logical_path} mounted base")
    if base_identity == expected_good:
        return ResolvedVideoOverride(
            logical_path=logical_path,
            selected_path=base_path,
            selected_source="base-good",
            expected_good=expected_good,
        )
    if base_identity != known_bad:
        raise VideoOverrideError(
            f"{logical_path}: mounted base has unrecognised identity "
            f"size={base_identity.size}, sha256={base_identity.sha256}; "
            f"expected good={expected_good} or known bad={known_bad}"
        )

    replacement_value = entry.get("replacement_path")
    if not isinstance(replacement_value, str) or not os.path.isabs(replacement_value):
        raise VideoOverrideError(f"{logical_path}.replacement_path must be an absolute path, got {replacement_value!r}")
    replacement_path = Path(replacement_value)
    if replacement_path.stem != expected_good.sha256:
        raise VideoOverrideError(
            f"{logical_path}: replacement path must be content-addressed by the "
            f"expected SHA-256; got {replacement_path.name!r}, "
            f"expected stem {expected_good.sha256!r}"
        )
    replacement_identity = file_identity(replacement_path, label=f"{logical_path} replacement")
    if replacement_identity != expected_good:
        raise VideoOverrideError(
            f"{logical_path}: replacement identity mismatch: " f"actual={replacement_identity}, expected={expected_good}"
        )
    return ResolvedVideoOverride(
        logical_path=logical_path,
        selected_path=replacement_path,
        selected_source="verified-overlay",
        expected_good=expected_good,
    )
