#!/usr/bin/env python3
"""Manage docs/plans/ entries.

Each plan is a single docs/plans/<name>.md file (flat layout, no nested dirs).
Frontmatter on the file drives placement and the generated indexes.

Run with no args to dry-run; pass --apply to:
  - move plans with status in {done, abandoned} into docs/plans/archive/,
    UNLESS deferred=unresolved (those stay active and surface in INDEX.md
    under "needs follow-up plan")
  - pull plans back to active/ if their status changed
  - regenerate docs/plans/INDEX.md and docs/plans/archive/ARCHIVE.md

Frontmatter fields read:
  id, title, status, priority, area, source_issue, created, updated,
  deferred, supersedes, superseded_by, note

Companion one-shot: scripts/plans_bootstrap.py (adds frontmatter to legacy
plans and audits for un-spun-off deferred sections).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANS = ROOT / "docs" / "plans"
ARCHIVE = PLANS / "archive"

VALID_STATUS = {"rough", "draft", "active", "done", "abandoned"}
VALID_PRIORITY = {"low", "medium", "high", "critical"}
ARCHIVE_STATUSES = {"done", "abandoned"}
# status values that don't require a Progress table (idea-stage sketches)
ROUGH_STATUSES = {"rough"}

STATUS_ORDER = ["active", "draft", "rough", "done", "abandoned"]
ACTIVE_SECTIONS = [
    ("needs follow-up plan", lambda fm: fm.status == "done" and fm.deferred == "unresolved"),
    ("active", lambda fm: fm.status == "active"),
    ("draft", lambda fm: fm.status == "draft"),
    ("rough (idea / sketch)", lambda fm: fm.status == "rough"),
]


@dataclass
class Frontmatter:
    id: str = ""
    title: str = ""
    status: str = "draft"
    priority: str = ""
    area: list[str] = field(default_factory=list)
    source_issue: str = ""
    created: str = ""
    updated: str = ""
    deferred: str = "none"
    supersedes: str = ""
    superseded_by: str = ""
    note: str = ""


@dataclass
class Entry:
    path: Path
    fm: Frontmatter
    has_progress_table: bool
    has_canonical_header: bool


FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
PROGRESS_RE = re.compile(r"^##\s+Progress\s*$", re.MULTILINE)
CANONICAL_HEADER_RE = re.compile(
    r"^\|\s*Phase\s*\|\s*Description\s*\|\s*Progress\s*\|\s*Effort\s*\|\s*Priority\s*\|\s*$",
    re.MULTILINE | re.IGNORECASE,
)
ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$",
    re.MULTILINE,
)
VALID_PROGRESS_WORDS = {"done", "complete", "completed", "shipped", "landed"}
PERCENT_RE = re.compile(r"^\d{1,3}\s*%$")
VALID_EFFORT = {"small", "medium", "large", "xl"}
VALID_ROW_PRIORITY = {"low", "medium", "high", "critical"}


def parse_frontmatter(text: str) -> Frontmatter | None:
    m = FM_RE.match(text)
    if not m:
        return None
    fm = Frontmatter()
    for line in m.group(1).splitlines():
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
            # strip optional quotes
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if hasattr(fm, key):
                setattr(fm, key, val)
    return fm


def has_progress_table(text: str) -> bool:
    """Detect `## Progress` followed (within a few lines) by a markdown table header."""
    m = PROGRESS_RE.search(text)
    if not m:
        return False
    tail = text[m.end():m.end() + 400]
    return "| Phase" in tail or "|Phase" in tail


def has_canonical_header(text: str) -> bool:
    """Detect the canonical `| Phase | Description | Progress | Effort | Priority |` header."""
    m = PROGRESS_RE.search(text)
    if not m:
        return False
    tail = text[m.end():m.end() + 400]
    return CANONICAL_HEADER_RE.search(tail) is not None


def progress_value_problems(text: str) -> list[str]:
    """Return a list of non-canonical Progress/Effort/Priority values found in the Progress table.

    Each item is a human-readable description of the offending row + column.
    """
    m = PROGRESS_RE.search(text)
    if not m:
        return []
    tail = text[m.end():]
    next_h2 = re.search(r"^##\s+\S", tail, re.MULTILINE)
    if next_h2:
        tail = tail[:next_h2.start()]
    problems: list[str] = []
    for rm in ROW_RE.finditer(tail):
        c1, _, c3, c4, c5 = (rm.group(i).strip() for i in range(1, 6))
        # skip header / separator
        if c1.lower() == "phase":
            continue
        if set(c1.replace(" ", "")) <= {"-", ":"}:
            continue
        prog = c3.lower().strip()
        if prog not in VALID_PROGRESS_WORDS and not PERCENT_RE.match(prog):
            problems.append(f"row {c1!r}: Progress value {c3!r} is not a percentage or one of {sorted(VALID_PROGRESS_WORDS)}")
        eff = c4.lower().strip()
        if eff not in VALID_EFFORT:
            problems.append(f"row {c1!r}: Effort value {c4!r} is not one of Small/Medium/Large/XL")
        pri = c5.lower().strip()
        if pri not in VALID_ROW_PRIORITY:
            problems.append(f"row {c1!r}: Priority value {c5!r} is not one of Low/Medium/High/Critical")
    return problems


def _scan_dir(base: Path) -> list[Entry]:
    out: list[Entry] = []
    if not base.exists():
        return out
    for md in sorted(base.glob("*.md")):
        if md.name in {"INDEX.md", "ARCHIVE.md", "README.md"}:
            continue
        text = md.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if fm is None:
            # surface as a no-frontmatter entry so the warning system can flag it
            out.append(Entry(path=md, fm=Frontmatter(id=md.stem, title=md.stem), has_progress_table=has_progress_table(text), has_canonical_header=has_canonical_header(text)))
            continue
        out.append(Entry(path=md, fm=fm, has_progress_table=has_progress_table(text), has_canonical_header=has_canonical_header(text)))
    return out


def find_entries() -> tuple[list[Entry], list[Entry]]:
    active = _scan_dir(PLANS)
    archived = _scan_dir(ARCHIVE)
    return active, archived


def wants_archive(fm: Frontmatter) -> bool:
    if fm.status not in ARCHIVE_STATUSES:
        return False
    # deferred=unresolved blocks archive even when status=done
    if fm.status == "done" and fm.deferred == "unresolved":
        return False
    return True


def validate(entries: list[Entry]) -> int:
    warnings = 0
    for e in entries:
        rel = e.path.relative_to(ROOT)
        # missing frontmatter shows up as default values; detect via empty id+title=stem
        text = e.path.read_text(encoding="utf-8")
        if not FM_RE.match(text):
            print(f"WARN: {rel}: no frontmatter. Run scripts/plans_bootstrap.py to add one.")
            warnings += 1
            continue
        if e.fm.status and e.fm.status not in VALID_STATUS:
            print(f"WARN: {rel}: unknown status {e.fm.status!r} (valid: {sorted(VALID_STATUS)}). Entry will NOT be archived.")
            warnings += 1
        if e.fm.priority and e.fm.priority not in VALID_PRIORITY:
            print(f"WARN: {rel}: unknown priority {e.fm.priority!r} (valid: {sorted(VALID_PRIORITY)}).")
            warnings += 1
        if not e.has_progress_table and e.fm.status not in ROUGH_STATUSES:
            print(f"WARN: {rel}: missing `## Progress` table (required by docs/plans/README.md).")
            warnings += 1
        elif e.has_progress_table and not e.has_canonical_header:
            print(f"WARN: {rel}: Progress table header is not the canonical `| Phase | Description | Progress | Effort | Priority |` (see docs/plans/README.md).")
            warnings += 1
        elif e.has_progress_table:
            text = e.path.read_text(encoding="utf-8")
            for problem in progress_value_problems(text):
                print(f"WARN: {rel}: {problem}")
                warnings += 1
        # deferred field: 'none', 'unresolved', or a path that must exist
        d = e.fm.deferred
        if d and d not in {"none", "unresolved"}:
            target = ROOT / d
            if not target.exists():
                print(f"WARN: {rel}: deferred path {d!r} does not exist.")
                warnings += 1
        if e.fm.status == "done" and e.fm.deferred == "":
            print(f"WARN: {rel}: status=done but deferred is unset. Set to 'none', 'unresolved', or a plan path.")
            warnings += 1
        for ref_field, ref_val in (("source_issue", e.fm.source_issue), ("supersedes", e.fm.supersedes), ("superseded_by", e.fm.superseded_by)):
            if ref_val and not (ROOT / ref_val).exists():
                print(f"WARN: {rel}: {ref_field} path {ref_val!r} does not exist.")
                warnings += 1
    return warnings


def move(src: Path, dst: Path, *, apply: bool) -> None:
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def relocate(active: list[Entry], archived: list[Entry], *, apply: bool) -> int:
    moved = 0
    for e in active:
        if wants_archive(e.fm):
            dst = ARCHIVE / e.path.name
            verb = "archive" if apply else "would archive"
            print(f"{verb}: {e.path.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            move(e.path, dst, apply=apply)
            moved += 1
    for e in archived:
        if not wants_archive(e.fm):
            dst = PLANS / e.path.name
            verb = "restore" if apply else "would restore"
            print(f"{verb}: {e.path.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            move(e.path, dst, apply=apply)
            moved += 1
    return moved


def entry_link(e: Entry, base: Path) -> str:
    rel = e.path.relative_to(base)
    title = e.fm.title or e.fm.id or e.path.stem
    bits = [f"[{title}]({rel})"]
    tags = []
    if e.fm.priority:
        tags.append(f"p:{e.fm.priority}")
    if e.fm.area:
        tags.extend(e.fm.area)
    if e.fm.deferred and e.fm.deferred not in {"none", ""}:
        if e.fm.deferred == "unresolved":
            tags.append("deferred:unresolved")
        else:
            tags.append("deferred:->")
    if e.fm.superseded_by:
        tags.append("superseded")
    if tags:
        bits.append("`" + " ".join(tags) + "`")
    if e.fm.note:
        bits.append(f"— {e.fm.note}")
    return " ".join(bits)


INDEX_PREAMBLE = """\
# Plans index

Auto-generated by `scripts/plans.py`. Do not edit by hand. See [README.md](README.md) for the frontmatter schema, the deferred-items policy, and the lifecycle workflow.

"""


def _group_by_area(bucket: list[Entry]) -> dict[str, list[Entry]]:
    """Group by the first area tag only; secondary tags surface inline via entry_link."""
    groups: dict[str, list[Entry]] = {}
    for e in bucket:
        key = e.fm.area[0] if e.fm.area else "(unfiled)"
        groups.setdefault(key, []).append(e)
    return groups


def write_active_index(active: list[Entry]) -> None:
    lines = [INDEX_PREAMBLE.rstrip(), ""]
    lines.append(f"## Active ({len(active)})")
    lines.append("")
    seen: set[Path] = set()
    for title, pred in ACTIVE_SECTIONS:
        bucket = [e for e in active if pred(e.fm) and e.path not in seen]
        if not bucket:
            continue
        lines.append(f"### {title} ({len(bucket)})")
        lines.append("")
        groups = _group_by_area(bucket)
        for area in sorted(groups):
            entries = sorted(groups[area], key=lambda e: e.fm.id or e.path.stem)
            lines.append(f"**{area}**")
            lines.append("")
            for e in entries:
                lines.append(f"- {entry_link(e, PLANS)}")
                seen.add(e.path)
            lines.append("")
    leftover = [e for e in active if e.path not in seen]
    if leftover:
        lines.append("### other")
        lines.append("")
        for e in leftover:
            lines.append(f"- {entry_link(e, PLANS)}")
        lines.append("")
    (PLANS / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def write_archive_index(archived: list[Entry]) -> None:
    lines = ["# Archived plans", ""]
    lines.append(f"{len(archived)} archived plans.")
    lines.append("")
    buckets: dict[str, list[Entry]] = {}
    for e in archived:
        buckets.setdefault(e.fm.status or "unknown", []).append(e)
    for key in sorted(buckets):
        bucket = buckets[key]
        lines.append(f"## {key} ({len(bucket)})")
        lines.append("")
        bucket.sort(key=lambda e: e.fm.id or e.path.stem)
        for e in bucket:
            extra = ""
            if e.fm.deferred and e.fm.deferred not in {"none", "", "unresolved"}:
                extra = f" — follow-ups: `{e.fm.deferred}`"
            lines.append(f"- {entry_link(e, ARCHIVE)}{extra}")
        lines.append("")
    (ARCHIVE / "ARCHIVE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Manage docs/plans/ entries (relocate + index).")
    ap.add_argument("--apply", action="store_true", help="actually move files and write indexes (default is dry-run)")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN (pass --apply to execute)"
    print(f"=== plans.py: {mode} ===")
    if args.apply:
        PLANS.mkdir(parents=True, exist_ok=True)
        ARCHIVE.mkdir(parents=True, exist_ok=True)
    if not PLANS.exists():
        print("docs/plans/ does not exist yet; nothing to do.")
        return 0
    active, archived = find_entries()
    all_entries = active + archived
    warnings = validate(all_entries)
    if warnings:
        print(f"({warnings} warning{'s' if warnings != 1 else ''} above — fix before --apply, or files will stay put.)\n")
    moved = relocate(active, archived, apply=args.apply)
    if moved and args.apply:
        active, archived = find_entries()
    if args.apply:
        write_active_index(active)
        write_archive_index(archived)
        print(f"indexed {len(active)} active + {len(archived)} archived -> {PLANS.relative_to(ROOT)}/INDEX.md, {ARCHIVE.relative_to(ROOT)}/ARCHIVE.md")
    else:
        print(f"would index {len(active)} active + {len(archived)} archived; pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
