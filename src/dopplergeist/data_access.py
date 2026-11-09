"""Validated historical inputs and disk-backed whole-segment spatial/time queries.

These are internal data adapters, not a second pipeline request/result contract.
Coverage describes the *source query*, never receiver/network completeness.
"""
from collections import Counter, OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sqlite3
import sys
import tarfile
import time
from threading import Lock

import numpy as np
import pandas as pd
import requests

from .geometry import Track
from .terrain import _glo90_tile_degrees
from .tracks import _segment_velocity, adsblol_frame

try:
    from ._native_index import payload_columns as _native_payload_columns
except ImportError:
    _native_payload_columns = None

_PAYLOAD_COLUMNS = ("time_s", "lat", "lon", "alt_m", "east_mps", "north_mps", "up_mps",
                    "typecode", "altitude_basis", "aircraft_class")

CONUS = (-125.0, 24.0, -66.0, 50.0)
INDEX_VERSION = 2

# These exact source versions differ only in whether the same altitude-offset
# arrays are allocated once per trace or once per interpolated row.  Keep the
# mapping keyed by the current complete source hash: any later normalizer edit
# falls back to the ordinary conservative mismatch behavior.
_EQUIVALENT_NORMALIZERS = {
    # The non-jet policy only changes the legacy CSV importer; trace decoding,
    # segment velocity and all spatial-bound geometry remain unchanged.
    "cb66d1a8184ea393c9b906e803c14ee62246625d26dad84b0ffb575b44431b4b": (
        "543f483cee756fdb9efdd69c8e8f05b32e290f19187af3523160f19d2eba90e0",
        "523b9d629083ba2792a1da2cd34d8b89ef57236febcc284f62886ba103ef20ec",
    ),
    "543f483cee756fdb9efdd69c8e8f05b32e290f19187af3523160f19d2eba90e0": (
        "523b9d629083ba2792a1da2cd34d8b89ef57236febcc284f62886ba103ef20ec",
    ),
}


def _staging_rows(frame):
    """Keep pandas' normalized JSON rows without encoding each one twice."""
    payload = frame.to_json(orient="records", double_precision=15)
    decoder = json.JSONDecoder()
    position = 1  # Skip the opening array bracket.
    while position < len(payload) - 1:
        row, end = decoder.raw_decode(payload, position)
        yield str(row["flight_id"]), row["time_s"], payload[position:end]
        position = end + 1  # Skip the comma (or closing bracket).


def _frame_track(name, frame):
    velocity, inferred = _segment_velocity(frame)
    codes = {str(value).strip().upper() for value in frame.typecode.to_numpy()
             if value is not None and str(value).strip() and str(value).lower() != "nan"} if "typecode" in frame else set()
    labels = {str(value).strip().lower() for value in frame.aircraft_class.to_numpy()
              if value is not None and str(value).strip() and str(value).lower() != "nan"} if "aircraft_class" in frame else set()
    code = next(iter(codes)) if len(codes) == 1 else ""
    label = "jet" if "jet" in labels else next(iter(labels)) if len(labels) == 1 else "unclassified"
    if len(codes) > 1:
        label = "jet" if label == "jet" else "unclassified"
    track = Track(name, frame.time_s.to_numpy(), frame.lat.to_numpy(), frame.lon.to_numpy(), frame.alt_m.to_numpy(),
                  code, velocity)
    track.altitude_basis = frame.altitude_basis.tolist() if "altitude_basis" in frame else ["unspecified"] * len(frame)
    track.aircraft_class = label
    track.velocity_inferred_components = inferred
    return track


class _IndexColumn:
    """The column operations used by the unchanged track normalization code."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = values

    def to_numpy(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    @property
    def iloc(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


class _IndexColumns:
    """A bounded column view of one canonical decoded index segment."""

    __slots__ = ("columns", "count")

    def __init__(self, columns, count):
        self.columns, self.count = columns, count

    def __len__(self):
        return self.count

    def __contains__(self, name):
        return name in self.columns

    def __getitem__(self, name):
        return self.columns[name]

    def __getattr__(self, name):
        try:
            return self.columns[name]
        except KeyError:
            raise AttributeError(name) from None


def _payload_frame(records):
    """Avoid pandas construction for canonical float64/string index records.

    Both paths still use _frame_track and _segment_velocity. Restrict this
    shortcut to representations whose pandas column conversion is exact;
    mixed schemas, integer position fields and unusual labels keep pandas'
    original dtype inference and failure behavior.
    """
    if _native_payload_columns is not None:
        values = _native_payload_columns(records)
        if values is not None:
            columns = {name: _IndexColumn(np.asarray(column, dtype=float if i < 7 else object))
                       for i, (name, column) in enumerate(zip(_PAYLOAD_COLUMNS, values))
                       if column is not None}
            return _IndexColumns(columns, len(records))
    if type(records) is not list or not records or any(type(row) is not dict for row in records):
        return pd.DataFrame(records)
    absent = object()
    columns = {}
    for name in ("time_s", "lat", "lon", "alt_m"):
        values = [row.get(name, absent) for row in records]
        if any(type(value) is not float for value in values):
            return pd.DataFrame(records)
        columns[name] = _IndexColumn(np.asarray(values, dtype=float))
    for name in ("east_mps", "north_mps", "up_mps", "typecode", "altitude_basis", "aircraft_class"):
        values = [row.get(name, absent) for row in records]
        if all(value is absent for value in values):
            continue
        numeric = name in ("east_mps", "north_mps", "up_mps")
        if numeric:
            canonical = all(type(value) is float or value is None for value in values)
        else:
            canonical = all(type(value) is str for value in values)
        if not canonical:
            return pd.DataFrame(records)
        columns[name] = _IndexColumn(np.asarray(values, dtype=float if numeric else object))
    return _IndexColumns(columns, len(records))


_CANONICAL_FRAME_TRACK = _frame_track
_CANONICAL_PAYLOAD_FRAME = _payload_frame
_CANONICAL_SEGMENT_VELOCITY = _segment_velocity
_CANONICAL_TRACK = Track
_PREPARED_ERROR_POLICY = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


@lru_cache(maxsize=1)
def _prepared_track_components():
    """Hash immutable runtime inputs once when checking compatible builders."""
    from . import flight_payload
    try:
        import numpy._core._multiarray_umath as numpy_core
    except ImportError:
        import numpy.core._multiarray_umath as numpy_core
    files = (Path(__file__), Path(__file__).with_name("tracks.py"),
             Path(__file__).with_name("geometry.py"), Path(flight_payload.__file__),
             Path(numpy_core.__file__))
    sources = [sha256(path) for path in files]
    return {"sources": sources, "numpy": np.__version__, "pandas": pd.__version__,
            "python": sys.version, "cpu": getattr(numpy_core, "__cpu_features__", {})}


@lru_cache(maxsize=16)
def _prepared_track_identity(data_access_digest=None, geometry_digest=None):
    """Use prepared velocities only with identical normalization and NumPy."""
    components = dict(_prepared_track_components())
    sources = list(components["sources"])
    if data_access_digest is not None:
        sources[0] = data_access_digest
    if geometry_digest is not None:
        sources[2] = geometry_digest
    components["sources"] = sources
    return content_hash(components)


@lru_cache(maxsize=1)
def _ecef_to_geographic():
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


@lru_cache(maxsize=32)
def _terrain_transformer(crs):
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def _continuous_bounds(track):
    """Bound entire Hermite curves using their cubic Bezier control hulls.

    Each curve lies inside a ball enclosing its four ECEF control points.
    Longitude extrema of that ball are analytic; latitude uses a conservative
    5500 km lower bound on geodetic curvature radii outside the 6000 km sphere.
    Invalid/underground extreme curves fall back to a global box, never a miss.
    """
    p0, p1 = track.xyz[:-1], track.xyz[1:]
    dt = np.diff(track.time)[:, None]
    velocity = track.speed_vector(track.time)
    controls = np.stack((p0, p0 + velocity[:-1] * dt / 3, p1 - velocity[1:] * dt / 3, p1), axis=1)
    centers = (p0 + p1) / 2
    radius = np.linalg.norm(controls - centers[:, None, :], axis=2).max(axis=1)
    if np.any(np.linalg.norm(centers, axis=1) - radius < 6000000):
        return (-180., 180., -90., 90.)
    lon, lat, _ = _ecef_to_geographic().transform(*centers.T)
    latitude_pad = np.rad2deg(radius / 5500000)
    horizontal = np.linalg.norm(centers[:, :2], axis=1)
    if np.any(radius >= horizontal):
        return (-180., 180., max(-90., float(np.min(lat - latitude_pad))), min(90., float(np.max(lat + latitude_pad))))
    longitude_pad = np.rad2deg(np.arcsin(radius / horizontal))
    west, east = float(np.min(lon-longitude_pad)), float(np.max(lon+longitude_pad))
    if west < -180 or east > 180:
        west, east = -180., 180.
    return (west, east, max(-90., float(np.min(lat-latitude_pad))), min(90., float(np.max(lat+latitude_pad))))


class DataUnavailable(RuntimeError):
    """Missing/corrupt provider input; distinct from an acoustic non-detection."""


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value, *, compact=False):
    """Atomically write JSON, with optional compact encoding for checkpoints."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=None if compact else 2,
                                   separators=(",", ":") if compact else None,
                                   allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def cache_lock(path):
    import portalocker
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        portalocker.lock(lock, portalocker.LOCK_EX)
        yield


def _file_stat_identity(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _path_stat_identity(path):
    """Return a path's identity comparable with an ``os.fstat`` identity.

    On Windows ``os.stat`` reports the creation time as ``st_ctime`` while
    ``os.fstat`` reports the last change time, so an unchanged file would look
    modified. Take the change time from an open handle there instead.
    """
    identity = _file_stat_identity(path.stat())
    if os.name == "nt":
        with path.open("rb") as stream:
            identity[-1] = os.fstat(stream.fileno()).st_ctime_ns
    return identity


def _checked_sha256(path, identity=None):
    """Hash the opened file, rejecting source mutation or path replacement."""
    with path.open("rb") as stream:
        before = _file_stat_identity(os.fstat(stream.fileno()))
        if identity is not None and before != identity:
            raise OSError(f"File changed while hashing: {path}")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = _file_stat_identity(os.fstat(stream.fileno()))
        if before != after or before != _path_stat_identity(path):
            raise OSError(f"File changed while hashing: {path}")
    return digest


def cached_sha256(path):
    """Reuse a stat-validated digest, falling back to hashing read-only inputs.

    Receipt and lock files are optional acceleration. Their I/O failures must
    not prevent reading an otherwise usable index or weaken source-change checks.
    """
    path = Path(path)
    receipt = path.with_name(path.name + ".sha256.json")
    lock = receipt.with_name(receipt.name + ".lock")
    with ExitStack() as stack:
        try:
            stack.enter_context(cache_lock(lock))
        except OSError:
            return _checked_sha256(path)
        identity = _path_stat_identity(path)
        try:
            saved = json.loads(receipt.read_text())
            digest = saved.get("sha256")
            if (saved.get("version") == 1 and saved.get("identity") == identity and
                    isinstance(digest, str) and len(digest) == 64 and
                    all(char in "0123456789abcdef" for char in digest)):
                current_identity = _path_stat_identity(path)
                if identity == current_identity:
                    return digest
                identity = current_identity
        except (OSError, UnicodeError, AttributeError, json.JSONDecodeError):
            pass
        digest = _checked_sha256(path, identity)
        try:
            write_json(receipt, {"version": 1, "identity": identity, "sha256": digest})
        except OSError:
            # The digest already passed the same source checks. Cache metadata
            # may be unwritable even when an existing lock could be acquired.
            pass
        return digest


def download(url, path, *, size=None, digest=None, offline=False, session=None,
             progress_callback=None, progress_label=None):
    """Resume only verified byte ranges; atomically publish validated files.

    A provider SHA-256 is preferred. Otherwise a locally computed hash receipt
    detects subsequent cache corruption, but does not authenticate the provider.
    The optional callback reports transfer/validation activity, not source trust.
    """
    path = Path(path)
    receipt = path.with_name(path.name + ".sha256.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with cache_lock(path.with_name(path.name + ".lock")):
        if path.exists():
            if progress_callback:
                progress_callback(phase="validating_cached_input", label=progress_label,
                                  bytes_done=0, bytes_total=path.stat().st_size)
            actual = sha256(path)
            expected = digest or (json.loads(receipt.read_text())["sha256"] if receipt.exists() else None)
            if size is not None and path.stat().st_size != size:
                raise DataUnavailable(f"Truncated cache: {path}")
            if expected is None or actual != expected:
                raise DataUnavailable(f"Unvalidated or corrupt cache: {path}")
            if progress_callback:
                cached_size = path.stat().st_size
                progress_callback(phase="cached_input_ready", label=progress_label,
                                  bytes_done=cached_size, bytes_total=cached_size)
            return {"url": url, "path": str(path), "sha256": actual, "bytes": path.stat().st_size}
        if offline:
            raise DataUnavailable(f"Offline cache miss: {path}")
        temporary = path.with_name(path.name + ".part")
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-", "Accept-Encoding": "identity"} if offset else {"Accept-Encoding": "identity"}
        try:
            if progress_callback:
                progress_callback(phase="starting_download", label=progress_label,
                                  bytes_done=offset, bytes_total=size)
            with (session or requests).get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise DataUnavailable("Provider did not honor requested byte range; partial preserved")
                expected_size = size
                if expected_size is None and response.headers.get("Content-Length"):
                    expected_size = offset + int(response.headers["Content-Length"])
                downloaded = offset
                with temporary.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(phase="downloading", label=progress_label,
                                                  bytes_done=downloaded, bytes_total=expected_size)
            if expected_size is not None and temporary.stat().st_size != expected_size:
                raise DataUnavailable(f"Truncated download, rerun to resume: {temporary}")
            actual = sha256(temporary)
            if digest and actual != digest:
                raise DataUnavailable(f"Provider checksum mismatch: {temporary}")
            temporary.replace(path)
            result = {"url": url, "path": str(path), "sha256": actual, "bytes": path.stat().st_size}
            write_json(receipt, result)
            if progress_callback:
                progress_callback(phase="download_ready", label=progress_label,
                                  bytes_done=result["bytes"], bytes_total=result["bytes"])
            return result
        except requests.RequestException as error:
            raise DataUnavailable(f"Download failed: {url}: {error}") from error


class SplitReader(io.RawIOBase):
    """Read archive parts as one stream without copying/extracting the archive."""
    def __init__(self, paths):
        self.paths, self.current = iter(paths), None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = Path(path).open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def _archive_members(paths, max_member_bytes):
    try:
        with io.BufferedReader(SplitReader(paths)) as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as archive:
                for member in archive:
                    if not member.isfile() or "trace_full_" not in member.name or not member.name.endswith(".json"):
                        continue
                    if member.size > max_member_bytes:
                        raise DataUnavailable(f"Oversized trace member: {member.name}")
                    yield archive.extractfile(member).read()
    except (tarfile.TarError, EOFError, OSError, ValueError) as error:
        raise DataUnavailable(f"Invalid/truncated ADS-B archive: {error}") from error


def _decode_archive_frame(raw, max_member_bytes):
    if raw[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
            raw = compressed.read(max_member_bytes + 1)
        if len(raw) > max_member_bytes:
            raise DataUnavailable("Oversized expanded trace")
    return adsblol_frame(json.loads(raw))


def archive_frames(paths, max_member_bytes=64 * 1024 * 1024):
    """Preserve all types, full geometry and altitude labels before indexing."""
    try:
        for raw in _archive_members(paths, max_member_bytes):
            frame = _decode_archive_frame(raw, max_member_bytes)
            if not frame.empty:
                yield frame
    except (EOFError, OSError, ValueError) as error:
        raise DataUnavailable(f"Invalid/truncated ADS-B archive: {error}") from error


def _prepare_staging(frame, class_map):
    required = ["flight_id", "time_s", "lat", "lon", "alt_m"]
    if not set(required).issubset(frame):
        raise DataUnavailable(f"Missing canonical columns: {required}")
    valid = np.isfinite(frame[["time_s", "lat", "lon", "alt_m"]].to_numpy(dtype=float)).all(axis=1)
    valid &= frame.lat.between(-90, 90) & frame.lon.between(-180, 180) & frame.flight_id.notna()
    clean = frame.loc[valid].copy()
    clean["aircraft_class"] = clean.typecode.map(class_map).astype(object).fillna("unclassified") if "typecode" in clean else "unclassified"
    altitude = clean.altitude_basis.value_counts().to_dict() if "altitude_basis" in clean else {"unspecified": len(clean)}
    return clean, int((~valid).sum()), altitude


_STAGING_CLASS_MAP = None
_STAGING_MEMBER_LIMIT = None


def _init_staging_worker(class_map, max_member_bytes):
    global _STAGING_CLASS_MAP, _STAGING_MEMBER_LIMIT
    _STAGING_CLASS_MAP, _STAGING_MEMBER_LIMIT = class_map, max_member_bytes


def _archive_confirmed_nonjet(frame):
    """Keep a complete archive trace only when every type is confirmed non-jet."""
    if frame.empty:
        return frame
    labels = set(frame.aircraft_class.astype(str))
    if len(labels) == 1 and labels.issubset({"piston", "turbine"}):
        return frame
    return frame.iloc[:0]


def _archive_nonjet_staging(frame, class_map):
    # Examine type metadata before discarding invalid positions, so a
    # conflicting record cannot be hidden by coordinate validation.
    if "typecode" in frame:
        raw_labels = set(frame.typecode.map(class_map).fillna("unclassified").astype(str))
    else:
        raw_labels = {"unclassified"}
    clean, invalid, _ = _prepare_staging(frame, class_map)
    if len(raw_labels) != 1 or not raw_labels.issubset({"piston", "turbine"}):
        clean = clean.iloc[:0]
    clean = _archive_confirmed_nonjet(clean)
    altitude = clean.altitude_basis.value_counts().to_dict() if "altitude_basis" in clean else {"unspecified": len(clean)}
    return list(_staging_rows(clean)), invalid, altitude


def _stage_archive_member(raw):
    frame = _decode_archive_frame(raw, _STAGING_MEMBER_LIMIT)
    if frame.empty:
        return None
    return _archive_nonjet_staging(frame, _STAGING_CLASS_MAP)


def _index_workers():
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, available - 1)


@contextmanager
def _index_pool(workers, initializer=None, initargs=()):
    # The parent writes SQLite. Each spawned worker must not start its own
    # multithreaded BLAS pool and oversubscribe the available cores.
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initializer, initargs=initargs) as pool:
            yield pool
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def archive_staging(paths, class_map, workers=None, max_member_bytes=64 * 1024 * 1024,
                    progress_callback=None):
    """Prepare confirmed non-jet archive rows with bounded parallelism."""
    if workers is None:
        workers = _index_workers()
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1:
        batches = 0
        for frame in archive_frames(paths, max_member_bytes):
            batches += 1
            if progress_callback:
                progress_callback(phase="decoding_archive", batches=batches)
            yield _archive_nonjet_staging(frame, class_map)
        return
    try:
        with _index_pool(workers, _init_staging_worker, (class_map, max_member_bytes)) as pool:
            pending = deque()
            batches = 0
            for raw in _archive_members(paths, max_member_bytes):
                pending.append(pool.submit(_stage_archive_member, raw))
                if len(pending) >= workers:
                    batch = pending.popleft().result()
                    batches += 1
                    if progress_callback:
                        progress_callback(phase="decoding_archive", batches=batches)
                    if batch is not None:
                        yield batch
            while pending:
                batch = pending.popleft().result()
                batches += 1
                if progress_callback:
                    progress_callback(phase="decoding_archive", batches=batches)
                if batch is not None:
                    yield batch
    except (EOFError, OSError, ValueError) as error:
        raise DataUnavailable(f"Invalid/truncated ADS-B archive: {error}") from error


def _build_flight_segments(flight, records, max_gap_s, namespace):
    frame = pd.DataFrame([json.loads(row) for row in records]).drop_duplicates("time_s")
    groups = (frame.time_s.diff() > max_gap_s).cumsum()
    segments = []
    for index, (_, chunk) in enumerate(frame.groupby(groups)):
        if len(chunk) < 3:
            continue
        name = flight + (f"@{namespace}" if namespace else "") + (f":{index}" if groups.max() else "")
        track = _frame_track(name, chunk)
        payload = gzip.compress(chunk.to_json(orient="records", double_precision=15).encode(), compresslevel=6, mtime=0)
        segments.append((name, float(chunk.time_s.iloc[0]), float(chunk.time_s.iloc[-1]),
                         payload, hashlib.sha256(payload).hexdigest(), _continuous_bounds(track)))
    return segments


def _flight_inputs(db, max_flight_samples):
    for (flight,) in db.execute("SELECT DISTINCT flight FROM samples ORDER BY flight"):
        count = db.execute("SELECT count(*) FROM samples WHERE flight=?", (flight,)).fetchone()[0]
        if count > max_flight_samples:
            raise DataUnavailable(f"Flight {flight} exceeds {max_flight_samples} sample memory limit")
        records = [row[0] for row in db.execute("SELECT row FROM samples WHERE flight=? ORDER BY time,rowid", (flight,))]
        yield flight, records


def _parallel_flight_segments(db, max_flight_samples, max_gap_s, namespace, workers):
    # Keep source order and cap queued serialized inputs as well as task count.
    max_pending_bytes = 128 * 1024 * 1024
    with _index_pool(workers) as pool:
        pending, queued_bytes = deque(), 0
        for flight, records in _flight_inputs(db, max_flight_samples):
            size = sum(len(row.encode("utf-8")) for row in records)
            while pending and (len(pending) >= workers or queued_bytes + size > max_pending_bytes):
                future, used = pending.popleft()
                queued_bytes -= used
                yield future.result()
            pending.append((pool.submit(_build_flight_segments, flight, records, max_gap_s, namespace), size))
            queued_bytes += size
        while pending:
            future, _ = pending.popleft()
            yield future.result()


class FlightIndex:
    """SQLite R-tree of whole segments with an optional per-instance track cache.

    ``track_cache_bytes`` bounds serialized snapshot bytes retained until this
    instance is released; zero disables reuse. At most 2,048 entries are kept.
    The limit excludes returned tracks, temporary serialization and map overhead.
    Concurrent queries use separate SQLite connections and return independent
    tracks, even when they reuse the same verified snapshot.
    """
    def __init__(self, path, *, track_cache_bytes=128 * 1024 * 1024):
        if type(track_cache_bytes) is not int or track_cache_bytes < 0:
            raise ValueError("Track cache size must be a nonnegative integer")
        self.path = Path(path)
        self._track_cache = OrderedDict()
        self._track_cache_bytes = 0
        self._track_cache_limit = track_cache_bytes
        self._track_cache_lock = Lock()
        if not self.path.exists():
            raise DataUnavailable(f"Missing flight index: {path}")
        try:
            with self.connect() as db:
                self.report = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
                if self.report["index_version"] not in (1, INDEX_VERSION):
                    raise DataUnavailable("Flight index version mismatch")
                self._prepared_tracks = self._prepared_available(db, self.report)
        except (sqlite3.DatabaseError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise DataUnavailable(f"Invalid flight index: {path}") from error

    def connect(self):
        return sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)

    @staticmethod
    def _prepared_available(db, report=None):
        # Recheck optional metadata and schema in the actual read snapshot.
        # The unique integer key must prevent duplicate original segments.
        try:
            if report is None:
                report = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
            prepared = report.get("prepared_tracks") if isinstance(report, dict) else None
            if not isinstance(prepared, dict) or prepared.get("version") != 1:
                return False
            builder = prepared.get("builder")
            identity = _prepared_track_identity()
            if builder != identity:
                from .index_compatibility import PREPARED_DATA_ACCESS_EQUIVALENCE, PREPARED_GEOMETRY_EQUIVALENCE
                current = sha256(Path(__file__))
                geometry = sha256(Path(__file__).with_name("geometry.py"))
                access_sources = (current, *PREPARED_DATA_ACCESS_EQUIVALENCE.get(current, ()))
                geometry_sources = (geometry, *PREPARED_GEOMETRY_EQUIVALENCE.get(geometry, ()))
                if (_prepared_track_identity(current, geometry) != identity
                        or all(_prepared_track_identity(access, geom) != builder
                               for access in access_sources for geom in geometry_sources)):
                    return False
            columns = [(row[1], row[2].upper(), row[5]) for row in
                       db.execute("PRAGMA table_info(prepared_tracks)")]
            return columns == [("id", "INTEGER", 1), ("source_hash", "TEXT", 0),
                               ("payload", "BLOB", 0), ("payload_hash", "TEXT", 0)]
        except (sqlite3.DatabaseError, TypeError, KeyError, json.JSONDecodeError):
            return False

    def _decode_track(self, name, payload, expected, *, max_samples=None, cache_scan=None, prepared=None):
        # Verify every fetched payload, including hits. A cached track must not
        # conceal damage to an index that was changed after the previous query.
        if hashlib.sha256(payload).hexdigest() != expected:
            raise DataUnavailable(f"Corrupt flight payload: {name}")
        key = (name, expected)
        with self._track_cache_lock:
            entry = self._track_cache.get(key)
            if entry is not None:
                count, cached = entry
                if max_samples is not None and count > max_samples:
                    raise DataUnavailable("Flight lookup sample budget exceeded")
                self._track_cache.move_to_end(key)
                if cache_scan is not None:
                    cache_scan["used"].add(key)
        if entry is not None:
            return pickle.loads(cached)
        track = None
        if (prepared is not None and _frame_track is _CANONICAL_FRAME_TRACK
                and _segment_velocity is _CANONICAL_SEGMENT_VELOCITY
                and _payload_frame is _CANONICAL_PAYLOAD_FRAME
                and Track is _CANONICAL_TRACK and np.geterr() == _PREPARED_ERROR_POLICY):
            source_hash, packed, packed_hash = prepared
            if source_hash == expected and type(packed) is bytes:
                from .flight_payload import decode_track
                # A damaged optional representation cannot replace the verified
                # original observations. Recompute through the legacy reader.
                if hashlib.sha256(packed).hexdigest() == packed_hash:
                    try:
                        track = decode_track(name, packed, max_samples=max_samples)
                    except DataUnavailable:
                        track = None
        if track is None:
            frame = _payload_frame(json.loads(gzip.decompress(payload)))
            if max_samples is not None and len(frame) > max_samples:
                raise DataUnavailable("Flight lookup sample budget exceeded")
            track = _frame_track(name, frame)
        if self._track_cache_limit and (cache_scan is None or cache_scan["admit"]):
            # These bytes are produced here, never read from a file/request.
            # Restoring them returns independent arrays and mutable diagnostics,
            # while reusing JSON decoding, velocity inference and spline setup.
            cached = pickle.dumps(track, protocol=pickle.HIGHEST_PROTOCOL)
            if len(cached) <= self._track_cache_limit:
                with self._track_cache_lock:
                    # Another query can fill this key while decoding runs
                    # outside the lock. Never count the same entry twice.
                    if key in self._track_cache:
                        self._track_cache.move_to_end(key)
                        if cache_scan is not None:
                            cache_scan["used"].add(key)
                        return track
                    # Bound serialized values and entry count separately so many
                    # tiny segments cannot grow the key/container overhead forever.
                    while self._track_cache and (self._track_cache_bytes + len(cached) > self._track_cache_limit
                                                 or len(self._track_cache) >= 2048):
                        # A query larger than the cache must not evict entries it
                        # just used. Otherwise repeated ordered scans thrash with
                        # zero hits. Older entries unused by this query can go.
                        if cache_scan is not None and next(iter(self._track_cache)) in cache_scan["used"]:
                            # Further misses still decode normally, but do not pay
                            # serialization costs after admission stops.
                            cache_scan["admit"] = False
                            return track
                        _, (_, previous) = self._track_cache.popitem(last=False)
                        self._track_cache_bytes -= len(previous)
                    self._track_cache[key] = (len(track.time), cached)
                    self._track_cache_bytes += len(cached)
                    if cache_scan is not None:
                        cache_scan["used"].add(key)
        return track

    @staticmethod
    def _prepare_tracks(db, scope=None, progress_callback=None):
        """Warm optional constructor inputs without removing observations."""
        from .flight_payload import encode_track
        db.execute("DROP TABLE IF EXISTS prepared_tracks")
        db.execute("CREATE TABLE prepared_tracks(id INTEGER PRIMARY KEY, source_hash TEXT, payload BLOB, payload_hash TEXT)")
        count = size = 0
        last_report = time.monotonic()
        if (_frame_track is not _CANONICAL_FRAME_TRACK
                or _segment_velocity is not _CANONICAL_SEGMENT_VELOCITY
                or _payload_frame is not _CANONICAL_PAYLOAD_FRAME
                or Track is not _CANONICAL_TRACK or np.geterr() != _PREPARED_ERROR_POLICY):
            return {"version": 1, "builder": None, "tracks": count, "bytes": size}
        query = "SELECT s.id,s.name,s.payload,s.payload_hash FROM segments s"
        parameters = ()
        if scope is not None:
            bounds, times = scope
            west, south, east, north = bounds
            start, end = times
            query += (" JOIN spatial r ON s.id=r.id WHERE r.west<=? AND r.east>=?"
                      " AND r.south<=? AND r.north>=? AND s.start<=? AND s.end>=?")
            parameters = (east, west, north, south, end, start)
        for identifier, name, payload, expected in db.execute(query + " ORDER BY s.id", parameters):
            if hashlib.sha256(payload).hexdigest() != expected:
                raise DataUnavailable(f"Corrupt flight payload: {name}")
            # Use the precise values the old reader gets AFTER the index's
            # JSON serialization and dtype inference, not the staging frame.
            track = _frame_track(name, _payload_frame(json.loads(gzip.decompress(payload))))
            try:
                packed = encode_track(track)
            except (DataUnavailable, TypeError, ValueError):
                continue
            db.execute("INSERT INTO prepared_tracks VALUES(?,?,?,?)",
                       (identifier, expected, packed, hashlib.sha256(packed).hexdigest()))
            count += 1
            size += len(packed)
            now = time.monotonic()
            if progress_callback and now - last_report >= 3.0:
                progress_callback(phase="preparing_tracks", tracks=count, bytes=size)
                last_report = now
        report = {"version": 1, "builder": _prepared_track_identity(), "tracks": count, "bytes": size}
        if scope is not None:
            report["scope"] = {"bounds": list(bounds), "time_range_s": list(times)}
        return report

    def prepare(self, destination, *, prepare_scope=None):
        """Create a new index copy with faster verified track decoding.

        The source stays read-only and the destination must not exist. Coverage,
        original payloads, spatial bounds and scientific input identity stay
        unchanged. Old readers ignore the optional numeric representation.
        Optional ``prepare_scope=(bounds_wsen, time_range_s)`` limits warming;
        queries outside it still decode the complete original observations.
        """
        from uuid import uuid4
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with cache_lock(destination.with_suffix(".lock")):
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Prepared index destination already exists: {destination}")
            temporary = destination.with_name(destination.name + ".preparing-" + uuid4().hex)
            db = sqlite3.connect(temporary)
            try:
                with self.connect() as source:
                    source.backup(db)
                # Copy metadata from the same snapshot as the observations.
                # The caller's older report object is mutable and may be stale.
                try:
                    report = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
                    if report["index_version"] not in (1, INDEX_VERSION):
                        raise DataUnavailable("Flight index version mismatch")
                except (sqlite3.DatabaseError, TypeError, KeyError, json.JSONDecodeError) as error:
                    raise DataUnavailable("Invalid source flight index metadata") from error
                report["prepared_tracks"] = self._prepare_tracks(db, prepare_scope)
                db.execute("UPDATE metadata SET value=?", (json.dumps(report),))
                db.commit()
                db.close()
                temporary.replace(destination)
            except BaseException:
                db.close()
                temporary.unlink(missing_ok=True)
                raise
        return type(self)(destination, track_cache_bytes=self._track_cache_limit)

    @classmethod
    def build(cls, path, frames, *, sources, coverage, max_gap_s=30.0, max_flight_samples=250000, class_map=None,
              prepared=False, prepare_scope=None, progress_callback=None):
        """Disk staging permits arbitrary source order without whole-table RAM.

        Explicit source coverage is required: cropped or prop-only historical
        data must never become an apparently complete national all-type index.
        The optional callback reports build phases and observed counts only.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        class_map = class_map or {}
        identity = {"index_version": INDEX_VERSION, "sources": sources, "coverage": coverage,
                    "normalizer_sha256": sha256(Path(__file__).with_name("tracks.py")),
                    "classification_map_hash": content_hash(class_map),
                    "max_gap_s": max_gap_s, "max_flight_samples": max_flight_samples}
        with cache_lock(path.with_suffix(".lock")):
            if path.exists():
                result = cls(path)
                if result.report["input_hash"] != content_hash(identity):
                    raise DataUnavailable("Existing flight index has different inputs/configuration")
                if progress_callback:
                    progress_callback(phase="flight_index_ready", rows=result.report.get("rows", 0),
                                      segments=result.report.get("segments", 0), cached=True)
                return result
            temporary = path.with_suffix(".building")
            temporary.unlink(missing_ok=True)
            db = sqlite3.connect(temporary)
            activity = {"phase": "creating_flight_index"}
            last_progress_report = 0.0

            def emit_progress(force=False, **fields):
                nonlocal last_progress_report
                if progress_callback is None:
                    return
                activity.update(fields)
                now = time.monotonic()
                if force or now - last_progress_report >= 3.0:
                    progress_callback(**activity)
                    last_progress_report = now

            def sqlite_progress():
                emit_progress()
                return 0

            try:
                emit_progress(force=True)
                if progress_callback:
                    db.set_progress_handler(sqlite_progress, 1_000_000)
                db.executescript("""
                    PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-16384;
                    PRAGMA temp_store=FILE;
                    CREATE TABLE samples(flight TEXT, time REAL, row TEXT);
                    CREATE TABLE segments(id INTEGER PRIMARY KEY, name TEXT UNIQUE, start REAL, end REAL,
                        payload BLOB, payload_hash TEXT);
                    CREATE VIRTUAL TABLE spatial USING rtree(id, west, east, south, north);
                    CREATE INDEX time_index ON segments(start,end);
                    CREATE TABLE metadata(value TEXT);
                """)
                rows, invalid, altitude = 0, 0, Counter()
                batches = 0
                emit_progress(phase="staging_samples", rows=rows, batches=batches,
                              invalid_rows=invalid, force=True)
                for frame in frames:
                    if prepared:
                        staged, invalid_count, altitude_counts = frame
                        count = len(staged)
                    else:
                        clean, invalid_count, altitude_counts = _prepare_staging(frame, class_map)
                        count, staged = len(clean), _staging_rows(clean)
                    invalid += invalid_count
                    altitude.update(altitude_counts)
                    db.executemany("INSERT INTO samples VALUES(?,?,?)", staged)
                    rows += count
                    batches += 1
                    emit_progress(phase="staging_samples", rows=rows, batches=batches, invalid_rows=invalid)
                emit_progress(phase="creating_sample_lookup", rows=rows, batches=batches, force=True)
                db.execute("CREATE INDEX flight_samples ON samples(flight,time)")
                db.commit()
                segments = 0
                flights_processed = 0
                emit_progress(phase="building_segments", rows=rows, segments=segments,
                              flights=flights_processed, force=True)
                namespace = coverage.get("release_tag", "")
                workers = _index_workers() if prepared else 1
                if workers > 1:
                    flights = _parallel_flight_segments(db, max_flight_samples, max_gap_s, namespace, workers)
                else:
                    flights = (_build_flight_segments(flight, records, max_gap_s, namespace)
                               for flight, records in _flight_inputs(db, max_flight_samples))
                for flight_segments in flights:
                    flights_processed += 1
                    for name, start, end, payload, payload_hash, bounds in flight_segments:
                        cur = db.execute("INSERT INTO segments(name,start,end,payload,payload_hash) VALUES(?,?,?,?,?)",
                                         (name, start, end, payload, payload_hash))
                        db.execute("INSERT INTO spatial VALUES(?,?,?,?,?)", (cur.lastrowid, *bounds))
                        segments += 1
                    emit_progress(phase="building_segments", rows=rows, segments=segments,
                                  flights=flights_processed)
                db.execute("DROP TABLE samples")
                report = dict(identity, input_hash=content_hash(identity), rows=rows, invalid_rows=invalid,
                              segments=segments, altitude_basis_counts=dict(altitude),
                              jets_excluded=coverage.get("aircraft_policy") == "confirmed_nonjet",
                              unknown_types_retained=coverage.get("aircraft_policy") != "confirmed_nonjet", geometry_clipped=False, continuous_geometry_bounds=True)
                emit_progress(phase="preparing_tracks", rows=rows, segments=segments, force=True)
                track_progress = None
                if progress_callback:
                    def track_progress(**fields):
                        emit_progress(force=True, **fields)
                report["prepared_tracks"] = cls._prepare_tracks(
                    db, prepare_scope, progress_callback=track_progress)
                db.execute("INSERT INTO metadata VALUES(?)", (json.dumps(report),))
                db.commit()
                emit_progress(phase="compacting_flight_index", rows=rows, segments=segments, force=True)
                db.execute("VACUUM")
                db.close()
                temporary.replace(path)
                emit_progress(phase="flight_index_ready", rows=rows, segments=segments, force=True)
                return cls(path)
            except BaseException:
                db.close()
                temporary.unlink(missing_ok=True)
                raise

    @classmethod
    def from_parquet(cls, path, parquet, *, coverage, **kwargs):
        import pyarrow.parquet as pq
        parquet = Path(parquet)
        try:
            parquet_file = pq.ParquetFile(parquet)
            metadata = parquet_file.schema_arrow.metadata or {}
            if b"dopplergeist.selection.v1" in metadata:
                raise DataUnavailable("Legacy selected Parquet lacks archive-day scope; reingest it")
            if b"dopplergeist.selection.v2" in metadata:
                selected = json.loads(metadata[b"dopplergeist.selection.v2"])
                bounds, times = coverage["bounds"], coverage["time_range_s"]
                box, interval = selected["bounds"], selected["time_range_s"]
                source_day = date.fromisoformat(selected["archive_day_utc"])
                if source_day.isoformat() != selected["archive_day_utc"]:
                    raise DataUnavailable("Invalid Parquet archive day")
                day_start = datetime.combine(source_day, datetime.min.time(), tzinfo=timezone.utc).timestamp()
                day_end = day_start + 86400.0
                if (len(bounds) != 4 or len(box) != 4 or len(times) != 2 or
                        not np.isfinite([*bounds, *box, *times]).all() or
                        type(selected["all_aircraft"]) is not bool or
                        not np.isfinite(selected["max_gap_s"]) or selected["max_gap_s"] <= 0 or
                        not (box[0] <= bounds[0] <= bounds[2] <= box[2] and
                             box[1] <= bounds[1] <= bounds[3] <= box[3] and times[0] <= times[1])):
                    raise DataUnavailable("Declared flight coverage exceeds the Parquet selection bounds")
                if (not isinstance(interval, list) or len(interval) != 2 or
                        not np.isfinite(interval).all() or
                        not day_start <= interval[0] <= times[0] <= times[1] <= interval[1] <= day_end):
                    raise DataUnavailable("Declared flight coverage exceeds the Parquet selection time")
                if coverage.get("all_aircraft", False) and selected["all_aircraft"] is not True:
                    raise DataUnavailable("Parquet selection excluded aircraft types")
                if (coverage.get("aircraft_policy") == "confirmed_nonjet" and not selected["all_aircraft"]
                        and selected.get("aircraft_policy") != "confirmed_nonjet"):
                    raise DataUnavailable("Parquet selection does not establish confirmed non-jet scope")
                if selected["max_gap_s"] != kwargs.get("max_gap_s", 30.0):
                    raise DataUnavailable("Index gap limit differs from the Parquet selection gap")
            source = {"path": str(parquet.resolve()), "sha256": sha256(parquet), "bytes": parquet.stat().st_size}
            frames = (batch.to_pandas() for batch in parquet_file.iter_batches(batch_size=16384))
            return cls.build(path, frames, sources=[source], coverage=coverage, **kwargs)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise DataUnavailable(f"Invalid/truncated Parquet: {parquet}: {error}") from error

    def _query_scope(self, bounds, time_range):
        """Use the same conservative envelope for broad and named queries."""
        west, south, east, north = bounds
        start, end = time_range
        current_normalizer = sha256(Path(__file__).with_name("tracks.py"))
        compatible = (current_normalizer, *_EQUIVALENT_NORMALIZERS.get(current_normalizer, ()))
        if (not self.report.get("continuous_geometry_bounds") or
                self.report.get("normalizer_sha256") not in compatible):
            # Old indexes used sample-only bounds or different derivatives.
            # Disable unsafe spatial pruning, preserving explicit budgets.
            return "", "s.start<=? AND s.end>=?", (end, start)
        return (" JOIN spatial r ON s.id=r.id",
                "r.west<=? AND r.east>=? AND r.south<=? AND r.north>=? AND s.start<=? AND s.end>=?",
                (east, west, north, south, end, start))

    def query(self, bounds, time_range):
        """Yield complete segments touching the query, retaining gaps and labels.

        Callers expand receiver bounds for acoustic range and event duration.
        An R-tree rectangle can return harmless extra tracks, never clipped ones.
        """
        join, predicate, params = self._query_scope(bounds, time_range)
        cache_scan = {"used": set(), "admit": True}
        with self.connect() as db:
            prepared = self._prepared_tracks
            if prepared:
                db.execute("BEGIN")
                prepared = self._prepared_available(db)
            fields = ",p.source_hash,p.payload,p.payload_hash" if prepared else ""
            prepared_join = " LEFT JOIN prepared_tracks p ON p.id=s.id" if prepared else ""
            sql = ("SELECT s.name,s.payload,s.payload_hash" + fields + " FROM segments s" + join + prepared_join
                   + " WHERE " + predicate + " ORDER BY s.name")
            for row in db.execute(sql, params):
                yield self._decode_track(*row[:3], cache_scan=cache_scan,
                                         prepared=row[3:] if prepared else None)

    def lookup(self, identifiers, *, max_tracks=2000, max_samples=500000,
               bounds=None, time_range=None):
        """Yield named whole segments without scanning unrelated trajectories.

        Missing names are omitted; callers must compare requested and returned
        IDs. Limits reject excess work explicitly, never truncate a trajectory.
        Supply both bounds and time_range to apply the same envelope as query,
        including its conservative fallback for older geometry indexes.
        This lookup establishes no spatial, temporal or receiver completeness.
        """
        if any(type(value) is not int or value <= 0 for value in (max_tracks, max_samples)):
            raise ValueError("Flight lookup limits must be positive integers")
        if (bounds is None) != (time_range is None):
            raise ValueError("Flight lookup bounds and time_range must be supplied together")
        names = set()
        for name in identifiers:
            if not isinstance(name, str) or not name:
                raise ValueError("Flight lookup requires nonempty string identifiers")
            names.add(name)
            if len(names) > max_tracks:
                raise DataUnavailable("Flight lookup identifier budget exceeded")
        names, samples = sorted(names), 0
        join, predicate, scope_params = ("", "", ())
        if bounds is not None:
            join, predicate, scope_params = self._query_scope(bounds, time_range)
        cache_scan = {"used": set(), "admit": True}
        with self.connect() as db:
            prepared = self._prepared_tracks
            if prepared:
                # Retain one snapshot across the checks and all lookup batches.
                db.execute("BEGIN")
                prepared = self._prepared_available(db)
            # Stay below SQLite's conservative bound on bound SQL parameters.
            for first in range(0, len(names), 500):
                batch = names[first:first+500]
                placeholders = ",".join("?" for _ in batch)
                fields = ",p.source_hash,p.payload,p.payload_hash" if prepared else ""
                prepared_join = " LEFT JOIN prepared_tracks p ON p.id=s.id" if prepared else ""
                sql = ("SELECT s.name,s.payload,s.payload_hash" + fields + " FROM segments s" + join + prepared_join
                       + " WHERE s.name IN (" + placeholders + ")"
                       + (" AND " + predicate if predicate else "") + " ORDER BY s.name")
                for row in db.execute(sql, (*batch, *scope_params)):
                    track = self._decode_track(*row[:3], max_samples=max_samples - samples,
                                               cache_scan=cache_scan,
                                               prepared=row[3:] if prepared else None)
                    samples += len(track.time)
                    yield track

    def coverage_for(self, bounds, time_range):
        source = self.report["coverage"]
        box, times = source["bounds"], source["time_range_s"]
        complete = (box[0] <= bounds[0] <= bounds[2] <= box[2] and box[1] <= bounds[1] <= bounds[3] <= box[3]
                    and times[0] <= time_range[0] <= time_range[1] <= times[1]
                    and (source.get("all_aircraft", False) or source.get("aircraft_policy") == "confirmed_nonjet"))
        return {"requested_bounds": list(bounds), "requested_time_range_s": list(time_range),
                "source": source, "scope_complete": bool(complete), "receiver_completeness_known": False,
                "input_hash": self.report["input_hash"]}


class FlightCache:
    """Date-to-index acquisition; immutable index per source/version/scope."""
    def __init__(self, root, *, offline=False, session=None):
        self.root, self.offline = Path(root), offline
        self.session = session or requests.Session()

    def acquire_day(self, day, *, provider="staging-0", prepare_scope=None, progress_callback=None):
        from . import tracks as normalizer
        day = date.fromisoformat(day)
        tag = f"v{day:%Y.%m.%d}-planes-readsb-{provider}"
        directory = self.root / day.isoformat() / provider
        directory.mkdir(parents=True, exist_ok=True)
        normalizer_hash = sha256(normalizer.__file__)
        index = directory / f"nonjet-v{INDEX_VERSION}-{normalizer_hash[:12]}.sqlite"
        if index.exists():
            existing = FlightIndex(index)
            if (existing.report.get("coverage", {}).get("aircraft_policy") != "confirmed_nonjet"
                    or existing.report.get("jets_excluded") is not True):
                raise DataUnavailable("Cached flight index does not establish confirmed non-jet scope")
            if progress_callback:
                progress_callback(phase="cached_flight_index", rows=existing.report.get("rows", 0),
                                  segments=existing.report.get("segments", 0))
            return existing
        for old_hash in _EQUIVALENT_NORMALIZERS.get(normalizer_hash, ()):
            old_index = directory / f"nonjet-v{INDEX_VERSION}-{old_hash[:12]}.sqlite"
            if old_index.exists():
                existing = FlightIndex(old_index)
                start = pd.Timestamp(day.isoformat(), tz="UTC").timestamp()
                try:
                    full_day = existing.coverage_for((-180, -90, 180, 90), (start, start + 86400))["scope_complete"]
                except (KeyError, TypeError, ValueError):
                    full_day = False
                if (existing.report.get("index_version") != INDEX_VERSION or
                        existing.report.get("normalizer_sha256") != old_hash or
                        existing.report.get("coverage", {}).get("aircraft_policy") != "confirmed_nonjet" or
                        existing.report.get("jets_excluded") is not True or
                        existing.report.get("coverage", {}).get("release_tag") != tag or not full_day):
                    raise DataUnavailable("Compatible flight index metadata mismatch")
                if progress_callback:
                    progress_callback(phase="cached_flight_index", rows=existing.report.get("rows", 0),
                                      segments=existing.report.get("segments", 0))
                return existing
        release_path = directory / "release.json"
        if not release_path.exists():
            if self.offline:
                raise DataUnavailable(f"Offline release metadata missing: {day}")
            try:
                response = self.session.get(f"https://api.github.com/repos/adsblol/globe_history_{day.year}/releases/tags/{tag}", timeout=60)
                response.raise_for_status()
                write_json(release_path, response.json())
            except requests.RequestException as error:
                raise DataUnavailable(f"Historical release unavailable: {day}") from error
        try:
            release = json.loads(release_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DataUnavailable(f"Invalid historical release metadata: {release_path}") from error
        if not isinstance(release, dict) or release.get("tag_name") != tag:
            raise DataUnavailable(f"Historical release metadata does not match requested tag: {tag}")
        assets = sorted((a for a in release["assets"] if a["name"].endswith(".tar") or ".tar." in a["name"]),
                        key=lambda a: a["name"])
        if not assets:
            raise DataUnavailable("Release has no trace archive")
        sources = []
        for number, asset in enumerate(assets, 1):
            digest = asset.get("digest")
            if digest and not digest.startswith("sha256:"):
                raise DataUnavailable("Unsupported archive digest algorithm")
            sources.append(download(asset["browser_download_url"], directory / asset["name"], size=asset["size"],
                                    digest=digest.split(":", 1)[1] if digest else None, offline=self.offline,
                                    session=self.session, progress_callback=progress_callback,
                                    progress_label=f"Historical flight archive {number}/{len(assets)}"))
        start = pd.Timestamp(day.isoformat(), tz="UTC").timestamp()
        coverage = {"bounds": [-180, -90, 180, 90], "time_range_s": [start, start + 86400],
                    "all_aircraft": False, "aircraft_policy": "confirmed_nonjet", "provider": "ADSB.lol", "release_tag": tag,
                    "release_sha256": sha256(release_path), "license": "ODbL", "missing_receiver_data_possible": True}
        coverage["normalizer_sha256"] = normalizer_hash
        types_path = self.root / "icao_aircraft_types.js"
        type_source = download("https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types.js",
                               types_path, offline=self.offline, session=self.session,
                               progress_callback=progress_callback, progress_label="Aircraft type metadata")
        raw = types_path.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        text = raw.decode()
        types = json.loads(text[text.index("{"):text.rindex("}") + 1])
        class_map = {}
        for code, value in types.items():
            description = value.get("desc", "") if isinstance(value, dict) else (value[2] if len(value) > 2 else "")
            # ICAO T also covers helicopter turboshafts; keep the family label broad.
            class_map[code] = {"J": "jet", "T": "turbine", "P": "piston"}.get(description[-1:], "unclassified")
        return FlightIndex.build(index, archive_staging([s["path"] for s in sources], class_map,
                                                        progress_callback=progress_callback),
                                 sources=sources + [type_source],
                                 coverage=coverage, class_map=class_map, prepared=True,
                                 prepare_scope=prepare_scope, progress_callback=progress_callback)


class CachedTerrain:
    """GLO-90 + EGM2008 with explicit offline operation and file provenance.

    Only catalogue-absent tiles are treated as ocean. Download errors and raster
    nodata remain missing coverage. Existing operator-supplied cache files can be
    adopted explicitly after format validation; receipts then detect corruption.
    """
    base = "https://copernicus-dem-90m.s3.amazonaws.com"

    def __init__(self, root, *, offline=False, geoid_path=None, adopt_existing=False):
        self.root, self.offline = Path(root), offline
        self.root.mkdir(parents=True, exist_ok=True)
        self.files, self.ocean_tiles, self.failures = {}, set(), []
        self._tile_cache = OrderedDict()
        self._tile_cache_bytes = 0
        self.adopt_existing = adopt_existing
        catalog = self.root / "tileList.txt"
        self._file(self.base + "/tileList.txt", catalog, raster=False)
        self.catalog = {line.rstrip("/") for line in catalog.read_text().splitlines()}
        if not self.catalog or not all(s.startswith("Copernicus_DSM_COG_30_") for s in self.catalog):
            raise DataUnavailable("Invalid GLO-90 catalogue")
        self.geoid_path = Path(geoid_path) if geoid_path else self.root / "us_nga_egm08_25.tif"
        self._file("https://cdn.proj.org/us_nga_egm08_25.tif", self.geoid_path, raster=True)

    def _file(self, url, path, *, raster):
        if path.name in self.files:
            return path
        receipt = path.with_name(path.name + ".sha256.json")
        if self.adopt_existing:
            # Adoption and download share the same file lock, so workers cannot
            # race while creating the receipt for an existing terrain file.
            with cache_lock(path.with_name(path.name + ".lock")):
                if path.exists() and not receipt.exists():
                    if raster:
                        import rasterio
                        with rasterio.open(path) as dataset:
                            dataset.read(1, window=((0, 1), (0, 1)))
                    write_json(receipt, {"sha256": sha256(path), "origin": "operator-supplied-cache", "url": url})
        result = download(url, path, offline=self.offline)
        if raster:
            import rasterio
            try:
                with rasterio.open(path) as dataset:
                    dataset.read(1, window=((0, 1), (0, 1)))
            except rasterio.errors.RasterioError as error:
                raise DataUnavailable(f"Invalid terrain raster: {path}") from error
        self.files[path.name] = result
        return path

    def sample(self, lat, lon):
        lat, lon = np.broadcast_arrays(lat, lon)
        shape = lat.shape
        lat, lon = lat.ravel(), lon.ravel()
        degrees = _glo90_tile_degrees(lat, lon)
        self.last_sample_files = {"tileList.txt", self.geoid_path.name}
        self.last_sample_ocean_tiles = set()
        tiles, inverse = np.unique(degrees, axis=0, return_inverse=True)
        height = np.zeros(len(lat))
        try:
            for i, (north, east) in enumerate(tiles):
                tag = f"Copernicus_DSM_COG_30_{'N' if north >= 0 else 'S'}{abs(north):02d}_00_{'E' if east >= 0 else 'W'}{abs(east):03d}_00_DEM"
                if tag not in self.catalog:
                    self.ocean_tiles.add(tag)
                    self.last_sample_ocean_tiles.add(tag)
                    continue
                path = self._file(f"{self.base}/{tag}/{tag}.tif", self.root / f"{tag}.tif", raster=True)
                self.last_sample_files.add(path.name)
                selected = inverse == i
                height[selected] = self._raster_sample(path, lat[selected], lon[selected])
            height += self._raster_sample(self.geoid_path, lat, lon)
        except (DataUnavailable, ValueError) as error:
            self.failures.append(str(error))
            raise DataUnavailable(f"Terrain coverage unavailable: {error}") from error
        return height.reshape(shape)

    def _raster_sample(self, path, lat, lon):
        """Cache small DEM tiles; keep large rasters on bounded window reads."""
        import rasterio
        from rasterio.windows import Window
        path = Path(path)
        if not len(lat):
            return np.array([])
        try:
            stat = path.stat()
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            entry = self._tile_cache.get(path)
            if entry is not None:
                if entry[0] != signature:
                    raise DataUnavailable(f"Terrain raster changed after caching: {path}")
                self._tile_cache.move_to_end(path)
                _, pixels, mask, crs, inverse, width, height = entry
                x, y = _terrain_transformer(crs).transform(lon, lat)
                col, row = inverse @ (np.asarray(x), np.asarray(y))
                col, row = np.floor(col).astype(int), np.floor(row).astype(int)
                if np.any((col < 0) | (col >= width) | (row < 0) | (row >= height)):
                    raise DataUnavailable(f"Raster extent missing candidate points: {path}")
                values = np.ma.array(pixels[row, col], mask=mask[row, col])
                if np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
                    raise DataUnavailable(f"Raster nodata at candidate points: {path}")
                return np.asarray(values, dtype=float)
            with rasterio.open(path) as raster:
                x, y = _terrain_transformer(raster.crs).transform(lon, lat)
                col, row = (~raster.transform) @ (np.asarray(x), np.asarray(y))
                col, row = np.floor(col).astype(int), np.floor(row).astype(int)
                if np.any((col < 0) | (col >= raster.width) | (row < 0) | (row >= raster.height)):
                    raise DataUnavailable(f"Raster extent missing candidate points: {path}")
                # Keep at most 32 MiB of decoded DEM pixels and masks per
                # terrain instance; global geoids retain window-only reads.
                limit = 32 * 1024 * 1024
                size = raster.width * raster.height * (np.dtype(raster.dtypes[0]).itemsize + 1)
                if path.name.startswith("Copernicus_DSM_COG_30_") and size <= limit:
                    while self._tile_cache and self._tile_cache_bytes + size > limit:
                        _, old = self._tile_cache.popitem(last=False)
                        self._tile_cache_bytes -= old[1].nbytes + old[2].nbytes
                        del old
                    tile = raster.read(1, masked=True)
                    pixels, mask = np.asarray(tile), np.ma.getmaskarray(tile)
                    after = path.stat()
                    if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                        raise DataUnavailable(f"Terrain raster changed while reading: {path}")
                    entry = (signature, pixels, mask, raster.crs, ~raster.transform, raster.width, raster.height)
                    self._tile_cache[path] = entry
                    self._tile_cache_bytes += entry[1].nbytes + entry[2].nbytes
                    values = np.ma.array(entry[1][row, col], mask=entry[2][row, col])
                else:
                    left, top, right, bottom = int(col.min()), int(row.min()), int(col.max()), int(row.max())
                    data = raster.read(1, window=Window(left, top, right-left+1, bottom-top+1), masked=True)
                    values = data[row-top, col-left]
                if np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
                    raise DataUnavailable(f"Raster nodata at candidate points: {path}")
                return np.asarray(values, dtype=float)
        except (rasterio.errors.RasterioError, OSError) as error:
            raise DataUnavailable(f"Cannot sample terrain raster: {path}") from error

    def provenance(self):
        return {"provider": "Copernicus GLO-90", "vertical_datum": "EGM2008 converted to ellipsoidal",
                "files": list(self.files.values()), "catalogue_absent_ocean_tiles": sorted(self.ocean_tiles),
                "failures": list(self.failures), "offline": self.offline}


@dataclass
class AcquiredInputs:
    indexes: list
    terrain: object
    bounds: tuple
    time_range_s: tuple
    recording_start_window_s: tuple
    coverage: dict


def acquire_inputs(request, recording, *, cache_root, index_paths=None, offline=False,
                   terrain_options=None, max_sound_range_m=20000.0, earliest_emission_s=None,
                   latest_emission_s=None, progress_callback=None):
    """Contract adapter for orchestration; no hand-prepared Parquet is required.

    `index_paths` is an operator-owned explicit offline override, not a user
    search input. Its source scope remains visible. The default obtains complete
    daily all-type archives once and reuses their disk indexes on later jobs.
    The optional progress callback reports acquisition activity, not coverage.
    """
    from .contracts import utc_timestamp, validate_request, validate_recording
    from .geography import profile_for_search
    request, recording = validate_request(request), validate_recording(recording)
    window = request.get("time_window") or recording.get("time_window")
    if window is None:
        raise ValueError("A defensible recording-start time window is required")
    start, end = (utc_timestamp(window[k]).timestamp() for k in ("start_utc", "end_utc"))
    first = start - max_sound_range_m / 300.0 - 30.0
    if earliest_emission_s is not None:
        if not np.isfinite(earliest_emission_s):
            raise ValueError("Earliest emission time must be finite")
        # The caller's bound covers its actual CPA and refinement queries. It
        # can be earlier or later than the generic whole-recording estimate.
        first = earliest_emission_s
    last = end + recording["duration_s"] + 30.0
    if latest_emission_s is not None:
        if not np.isfinite(latest_emission_s):
            raise ValueError("Latest emission time must be finite")
        last = max(last, latest_emission_s)
    if first > last:
        raise ValueError("Required emission interval must be ordered")
    times = (first, last)
    root = Path(cache_root)
    profile = profile_for_search(request["search"])
    bounds = tuple(request["search"].get("bounds_wsen", profile.bounds_wsen))
    if index_paths:
        indexes = [FlightIndex(path) for path in index_paths]
    else:
        cache = FlightCache(root / "flights", offline=offline)
        first, last = (datetime.fromtimestamp(t, timezone.utc).date() for t in times)
        indexes = []
        day = first
        while day <= last:
            indexes.append(cache.acquire_day(day.isoformat(), prepare_scope=(bounds, times),
                                             progress_callback=progress_callback))
            day += timedelta(days=1)
    options = dict(terrain_options or {})
    terrain = CachedTerrain(root / "terrain", offline=offline, **options)
    coverage = {"requested_area": request["search"], "sources": [i.coverage_for(bounds, times) for i in indexes],
                "limitations": ["ADS-B receiver completeness is unknown; unobserved flights remain possible.",
                                "Copernicus is a surface model and cannot establish microphone height."]}
    return AcquiredInputs(indexes, terrain, bounds, times, (start, end), coverage)
