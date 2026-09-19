#!/usr/bin/env python3
"""Classify modified files as whitespace-only or semantic vs HEAD.

For each tracked file with unstaged or staged changes, this strips all
whitespace from both the HEAD blob and the working-tree version, then
compares. Files whose stripped contents match are pure formatting changes;
files whose stripped contents differ contain real edits.

Usage:
    scripts/check_whitespace_only.py
    scripts/check_whitespace_only.py --semantic   # only list semantic-diff files
    scripts/check_whitespace_only.py --formatting # only list formatting-only files

Exits 0 always. Does not modify the working tree, the index, or any commits.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def modified_files() -> list[str]:
    """Return tracked files that differ from HEAD (staged or unstaged)."""
    out = subprocess.run(
        ["git", "diff", "HEAD", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def head_blob(path: str) -> bytes | None:
    """Return the HEAD blob for `path`, or None if not tracked at HEAD."""
    res = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True,
    )
    if res.returncode != 0:
        return None
    return res.stdout


def strip_ws(data: bytes) -> bytes:
    """Drop every ASCII whitespace byte. Bytes outside ASCII are kept verbatim."""
    return bytes(b for b in data if b not in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20))


def rustfmt(data: bytes) -> bytes | None:
    """Pipe `data` through rustfmt with the project config. None on rustfmt error."""
    res = subprocess.run(
        ["rustfmt", "--edition", "2024", "--emit", "stdout"],
        input=data,
        capture_output=True,
    )
    if res.returncode != 0:
        return None
    return res.stdout


def equivalent_after_rustfmt(head: bytes, wt: bytes) -> bool | None:
    """True if both versions normalize to the same rustfmt output. None on error."""
    h = rustfmt(head)
    if h is None:
        return None
    w = rustfmt(wt)
    if w is None:
        return None
    return h == w


def classify(path: str) -> str:
    """Return 'formatting', 'semantic', 'untracked', or 'missing'.

    For .rs files, normalises both versions through rustfmt before comparing,
    so import-order, method-chain, and other rustfmt-driven reflows count as
    formatting. For other files, falls back to byte-level whitespace stripping.
    """
    head = head_blob(path)
    if head is None:
        return "untracked"
    wt = Path(path)
    if not wt.exists():
        return "missing"
    wt_bytes = wt.read_bytes()
    if path.endswith(".rs"):
        eq = equivalent_after_rustfmt(head, wt_bytes)
        if eq is not None:
            return "formatting" if eq else "semantic"
    return "formatting" if strip_ws(head) == strip_ws(wt_bytes) else "semantic"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--semantic",
        action="store_true",
        help="only list files whose stripped contents differ",
    )
    group.add_argument(
        "--formatting",
        action="store_true",
        help="only list files whose stripped contents match",
    )
    args = parser.parse_args()

    files = modified_files()
    if not files:
        print("no modified files vs HEAD")
        return 0

    buckets: dict[str, list[str]] = {
        "formatting": [],
        "semantic": [],
        "untracked": [],
        "missing": [],
    }
    for path in files:
        buckets[classify(path)].append(path)

    if args.semantic:
        for path in buckets["semantic"]:
            print(path)
        return 0
    if args.formatting:
        for path in buckets["formatting"]:
            print(path)
        return 0

    for label in ("semantic", "formatting", "untracked", "missing"):
        entries = buckets[label]
        if not entries:
            continue
        print(f"{label} ({len(entries)}):")
        for path in entries:
            print(f"  {path}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
