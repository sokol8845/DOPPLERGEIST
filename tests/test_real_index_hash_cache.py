"""Digest-cache controls on an index built from archived aircraft observations."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3

import pandas as pd
import pytest

from dopplergeist import data_access
from dopplergeist.data_access import FlightIndex, cached_sha256, sha256


@pytest.fixture
def observed_index(tmp_path):
    source = Path(__file__).resolve().parents[1] / "research/trajectory-motion/frozen/canonical-observations.json"
    frame = pd.DataFrame(json.loads(source.read_text()))
    frame = frame.loc[frame.typecode.isin(["C172", "H500"])].copy()
    coverage = {"bounds": [-180., -90., 180., 90.],
                "time_range_s": [float(frame.time_s.min()), float(frame.time_s.max())],
                "all_aircraft": False, "source_note": "Archived real observations; incomplete excerpt"}
    index = FlightIndex.build(tmp_path / "observed.sqlite", [frame], sources=[], coverage=coverage)
    assert index.report["segments"] > 0
    return index.path


def test_unchanged_real_index_reuses_verified_digest(observed_index, monkeypatch):
    expected = sha256(observed_index)
    original, calls = data_access.hashlib.file_digest, []

    def count(stream, algorithm):
        calls.append(stream.name)
        return original(stream, algorithm)

    monkeypatch.setattr(data_access.hashlib, "file_digest", count)
    assert cached_sha256(observed_index) == expected
    assert cached_sha256(observed_index) == expected
    assert calls == [str(observed_index)]


@pytest.mark.parametrize("failure", ["lock", "receipt"])
def test_real_index_remains_readable_when_sidecars_are_unwritable(observed_index, monkeypatch, failure):
    expected = sha256(observed_index)
    receipt = observed_index.with_name(observed_index.name + ".sha256.json")

    @contextmanager
    def blocked_lock(path):
        raise PermissionError("read-only index directory")
        yield

    def blocked_receipt(path, value):
        raise PermissionError("read-only digest receipt")

    # Exercise both failures even under accounts that bypass chmod restrictions.
    monkeypatch.setattr(data_access, "cache_lock" if failure == "lock" else "write_json",
                        blocked_lock if failure == "lock" else blocked_receipt)
    assert cached_sha256(observed_index) == expected
    assert not receipt.exists()
    assert FlightIndex(observed_index).report["segments"] > 0


def test_modified_real_index_is_rehashed_despite_restored_mtime(observed_index):
    before = observed_index.stat()
    old_digest = cached_sha256(observed_index)
    # Modify metadata only in this disposable index. Observations stay intact.
    with sqlite3.connect(observed_index) as database:
        metadata = json.loads(database.execute("SELECT value FROM metadata").fetchone()[0])
        metadata["digest_cache_test"] = "updated metadata"
        database.execute("UPDATE metadata SET value=?", (json.dumps(metadata),))
    os.utime(observed_index, ns=(before.st_atime_ns, before.st_mtime_ns))
    expected = sha256(observed_index)
    assert expected != old_digest
    assert cached_sha256(observed_index) == expected


@pytest.mark.parametrize("sidecar_unavailable", [False, True])
def test_real_index_mutation_during_hash_is_rejected(observed_index, monkeypatch, sidecar_unavailable):
    original = data_access.hashlib.file_digest

    def change_during_hash(stream, algorithm):
        digest = original(stream, algorithm)
        # Deliberately corrupt only the disposable copy after its bytes were read.
        with observed_index.open("ab") as changed:
            changed.write(b"\0")
        return digest

    @contextmanager
    def blocked_lock(path):
        raise PermissionError("read-only index directory")
        yield

    monkeypatch.setattr(data_access.hashlib, "file_digest", change_during_hash)
    if sidecar_unavailable:
        monkeypatch.setattr(data_access, "cache_lock", blocked_lock)
    with pytest.raises(OSError, match="File changed while hashing"):
        cached_sha256(observed_index)
    assert not observed_index.with_name(observed_index.name + ".sha256.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows refuses to replace a file that is open for hashing")
def test_real_index_replacement_during_hash_is_rejected(observed_index, monkeypatch):
    replacement = observed_index.with_name("replacement.sqlite")
    shutil.copyfile(observed_index, replacement)
    original = data_access.hashlib.file_digest

    def replace_during_hash(stream, algorithm):
        digest = original(stream, algorithm)
        replacement.replace(observed_index)
        return digest

    monkeypatch.setattr(data_access.hashlib, "file_digest", replace_during_hash)
    with pytest.raises(OSError, match="File changed while hashing"):
        cached_sha256(observed_index)


def test_change_during_receipt_hit_rehashes_current_bytes(observed_index, monkeypatch):
    old_digest = cached_sha256(observed_index)
    original, calls = Path.stat, 0

    def stat(path, *args, **kwargs):
        nonlocal calls
        if path == observed_index:
            calls += 1
            if calls == 2:
                with observed_index.open("ab") as changed:
                    changed.write(b"\0")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    current = cached_sha256(observed_index)
    assert current != old_digest
    assert current == sha256(observed_index)
