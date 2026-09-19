#!/usr/bin/env python3
"""Manage docs/issues/ entries.

Each entry is either docs/issues/<id>.md or docs/issues/<id>/<id>.md (with
sibling assets). The frontmatter on the indexed doc drives placement and the
generated indexes.

Run with no args to:
  - move entries with triage=wont-fix or resolution in {resolved, obsolete}
    into docs/issues/archive/, and pull anything back if it became active
  - regenerate docs/issues/INDEX.md and docs/issues/archive/ARCHIVE.md

Frontmatter fields read:
  id, title, type, triage, resolution, severity, area, created, updated,
  resolved_in, plan, note
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUES = ROOT / "docs" / "issues"
ARCHIVE = ISSUES / "archive"

VALID_TRIAGE = {"unreviewed", "reviewed", "wont-fix"}
VALID_RESOLUTION = {"none", "planned", "in-progress", "resolved", "obsolete"}
VALID_PRIORITY = {"low", "medium", "high", "critical"}
VALID_EFFORT = {"S", "M", "L", "XL"}
VALID_PLANNED = {"yes", "no"}
ARCHIVE_RESOLUTIONS = {"resolved", "obsolete"}
ARCHIVE_TRIAGE = {"wont-fix"}

TYPE_ORDER = ["bug", "problem", "request", "report"]
ACTIVE_SECTIONS = [
    ("unreviewed", lambda fm: fm.triage == "unreviewed"),
    ("reviewed, no plan", lambda fm: fm.triage == "reviewed" and fm.resolution == "none"),
    ("planned", lambda fm: fm.resolution == "planned"),
    ("in progress", lambda fm: fm.resolution == "in-progress"),
]


@dataclass
class Entry:
    path: Path  # the indexed .md file
    root: Path  # the entry root (file or its parent dir if multi-file)
    fm: "Frontmatter"
    multi_file: bool


@dataclass
class Frontmatter:
    id: str = ""
    title: str = ""
    type: str = ""
    triage: str = "unreviewed"
    resolution: str = "none"
    priority: str = ""
    effort: str = ""
    planned: str = ""
    severity: str = ""
    area: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    resolved_in: str = ""
    plan: str = ""
    note: str = ""


FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_frontmatter(text: str) -> Frontmatter | None:
    m = FM_RE.match(text)
    if not m:
        return None
    fm = Frontmatter()
    block = m.group(1)
    for line in block.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [v.strip() for v in val[1:-1].split(",") if v.strip()]
            if hasattr(fm, key):
                setattr(fm, key, items)
        else:
            if hasattr(fm, key):
                setattr(fm, key, val)
    return fm


def _scan(base: Path) -> list[Entry]:
    """Scan a single directory (one level deep + one nested-entry-dir level)."""
    out: list[Entry] = []
    if not base.exists():
        return out
    for md in sorted(base.rglob("*.md")):
        if md.name in {"INDEX.md", "ARCHIVE.md", "README.md"}:
            continue
        parent = md.parent
        if parent == base:
            multi = False
        else:
            # only allow one nesting level: <base>/<entry-dir>/<entry-dir>.md
            if parent.parent != base:
                continue
            if md.stem != parent.name:
                continue  # sidecar inside an entry dir
            multi = True
        text = md.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if fm is None:
            continue
        root = parent if multi else md
        out.append(Entry(path=md, root=root, fm=fm, multi_file=multi))
    return out


def find_entries() -> list[Entry]:
    # scan active root (non-recursive into archive) and archive separately
    active: list[Entry] = []
    for md in sorted(ISSUES.glob("*.md")):
        if md.name in {"INDEX.md", "README.md"}:
            continue
        fm = parse_frontmatter(md.read_text(encoding="utf-8"))
        if fm is None:
            continue
        active.append(Entry(path=md, root=md, fm=fm, multi_file=False))
    # multi-file entries at top level: <id>/<id>.md
    for d in sorted(p for p in ISSUES.iterdir() if p.is_dir() and p != ARCHIVE):
        md = d / f"{d.name}.md"
        if not md.exists():
            continue
        fm = parse_frontmatter(md.read_text(encoding="utf-8"))
        if fm is None:
            continue
        active.append(Entry(path=md, root=d, fm=fm, multi_file=True))
    # archive: same shape
    archived = _scan(ARCHIVE)
    return active + archived


def should_archive(fm: Frontmatter) -> bool:
    return fm.resolution in ARCHIVE_RESOLUTIONS or fm.triage in ARCHIVE_TRIAGE


def validate(entries: list[Entry]) -> int:
    """Print warnings for unknown triage/resolution values. Returns warning count."""
    warnings = 0
    for e in entries:
        rel = e.path.relative_to(ROOT)
        if e.fm.triage and e.fm.triage not in VALID_TRIAGE:
            print(f"WARN: {rel}: unknown triage value {e.fm.triage!r} (valid: {sorted(VALID_TRIAGE)}). Entry will NOT be archived.")
            warnings += 1
        if e.fm.resolution and e.fm.resolution not in VALID_RESOLUTION:
            print(f"WARN: {rel}: unknown resolution value {e.fm.resolution!r} (valid: {sorted(VALID_RESOLUTION)}). Entry will NOT be archived.")
            warnings += 1
        if e.fm.priority and e.fm.priority not in VALID_PRIORITY:
            print(f"WARN: {rel}: unknown priority value {e.fm.priority!r} (valid: {sorted(VALID_PRIORITY)}).")
            warnings += 1
        if e.fm.effort and e.fm.effort not in VALID_EFFORT:
            print(f"WARN: {rel}: unknown effort value {e.fm.effort!r} (valid: {sorted(VALID_EFFORT)}).")
            warnings += 1
        if e.fm.planned and e.fm.planned not in VALID_PLANNED:
            print(f"WARN: {rel}: unknown planned value {e.fm.planned!r} (valid: {sorted(VALID_PLANNED)}).")
            warnings += 1
    return warnings


def move(src: Path, dst: Path, *, apply: bool) -> None:
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def relocate(entries: list[Entry], *, apply: bool) -> int:
    moved = 0
    for e in entries:
        in_archive = ARCHIVE in e.root.parents or e.root == ARCHIVE
        wants_archive = should_archive(e.fm)
        if wants_archive and not in_archive:
            dst = ARCHIVE / e.root.name
            verb = "archive" if apply else "would archive"
            print(f"{verb}: {e.root.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            move(e.root, dst, apply=apply)
            moved += 1
        elif not wants_archive and in_archive:
            dst = ISSUES / e.root.name
            verb = "restore" if apply else "would restore"
            print(f"{verb}: {e.root.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            move(e.root, dst, apply=apply)
            moved += 1
    return moved


def entry_link(e: Entry, base: Path) -> str:
    rel = e.path.relative_to(base)
    title = e.fm.title or e.fm.id or e.path.stem
    bits = [f"[{title}]({rel})"]
    if e.multi_file:
        bits.append("(dir)")
    tags = []
    if e.fm.type:
        tags.append(e.fm.type)
    if e.fm.priority:
        tags.append(f"p:{e.fm.priority}")
    if e.fm.effort:
        tags.append(f"e:{e.fm.effort}")
    if e.fm.planned == "yes":
        tags.append("planned")
    if e.fm.severity:
        tags.append(e.fm.severity)
    if e.fm.area:
        tags.extend(e.fm.area)
    if tags:
        bits.append("`" + " ".join(tags) + "`")
    return " ".join(bits)


INDEX_PREAMBLE = """\
# Issues index

Auto-generated by `scripts/issues.py`. Do not edit by hand. See [README.md](README.md) for the source-reliability caveats, the cross-consumer evaluation guidance, the plugins-first rule, and the full lifecycle workflow.

"""

def write_active_index(entries: list[Entry]) -> None:
    active = [e for e in entries if not should_archive(e.fm)]
    lines = [INDEX_PREAMBLE.rstrip(), ""]
    lines.append(f"## Active ({len(active)})")
    lines.append("")
    seen: set[str] = set()
    for title, pred in ACTIVE_SECTIONS:
        bucket = [e for e in active if pred(e.fm) and e.fm.id not in seen]
        if not bucket:
            continue
        lines.append(f"### {title} ({len(bucket)})")
        lines.append("")
        bucket.sort(key=lambda e: (TYPE_ORDER.index(e.fm.type) if e.fm.type in TYPE_ORDER else 99, e.fm.id))
        for e in bucket:
            lines.append(f"- {entry_link(e, ISSUES)}")
            seen.add(e.fm.id)
        lines.append("")
    leftover = [e for e in active if e.fm.id not in seen]
    if leftover:
        lines.append("### other")
        lines.append("")
        for e in leftover:
            lines.append(f"- {entry_link(e, ISSUES)}")
        lines.append("")
    (ISSUES / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def write_archive_index(entries: list[Entry]) -> None:
    archived = [e for e in entries if should_archive(e.fm)]
    lines = ["# Archived issues", ""]
    lines.append(f"{len(archived)} archived entries.")
    lines.append("")
    buckets: dict[str, list[Entry]] = {}
    for e in archived:
        key = e.fm.resolution if e.fm.resolution in ARCHIVE_RESOLUTIONS else e.fm.triage
        buckets.setdefault(key, []).append(e)
    for key in sorted(buckets):
        bucket = buckets[key]
        lines.append(f"## {key} ({len(bucket)})")
        lines.append("")
        bucket.sort(key=lambda e: e.fm.id)
        for e in bucket:
            extra = f" — resolved in `{e.fm.resolved_in}`" if e.fm.resolved_in else ""
            lines.append(f"- {entry_link(e, ARCHIVE)}{extra}")
        lines.append("")
    (ARCHIVE / "ARCHIVE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Manage docs/issues/ entries (relocate + index).")
    ap.add_argument("--apply", action="store_true", help="actually move files and write indexes (default is dry-run)")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN (pass --apply to execute)"
    print(f"=== issues.py: {mode} ===")
    if args.apply:
        ISSUES.mkdir(parents=True, exist_ok=True)
        ARCHIVE.mkdir(parents=True, exist_ok=True)
    if not ISSUES.exists():
        print("docs/issues/ does not exist yet; nothing to do.")
        return 0
    entries = find_entries()
    warnings = validate(entries)
    if warnings:
        print(f"({warnings} warning{'s' if warnings != 1 else ''} above — fix typos before --apply, or files will stay put.)\n")
    moved = relocate(entries, apply=args.apply)
    if moved and args.apply:
        entries = find_entries()
    if args.apply:
        write_active_index(entries)
        write_archive_index(entries)
        print(f"indexed {len(entries)} entries -> {ISSUES.relative_to(ROOT)}/INDEX.md, {ARCHIVE.relative_to(ROOT)}/ARCHIVE.md")
    else:
        active = sum(1 for e in entries if not should_archive(e.fm))
        archived = len(entries) - active
        print(f"would index {len(entries)} entries ({active} active, {archived} archived); pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
