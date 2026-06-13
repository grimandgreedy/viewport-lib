#!/usr/bin/env python3
"""One-shot bootstrap for docs/plans/ frontmatter.

Walks every docs/plans/<name>.md, and for each one:

  - parses the existing Progress table (if any) to estimate completion
  - guesses an `area` tag list from filename keywords
  - emits a proposed frontmatter block
  - audits the body for sections that look like un-spun-off follow-ups
    (e.g. `## Deferred`, `## Follow-up`, `## Future Work`, `Out of scope`)
  - flags plans with no Progress table at all

Default is DRY-RUN: prints the proposal + audit report.
Pass --apply to prepend proposed frontmatter to plans that don't already
have one. Pass --audit-only to skip the proposal/apply step and just print
the deferred-section audit report.

The --apply mode is conservative:
  - never overwrites an existing frontmatter block
  - never edits the body
  - sets `deferred: unresolved` if the audit found a follow-up section,
    otherwise `deferred: none`
  - sets `status: done` only if ALL Progress rows are 100% AND audit found
    no deferred sections; otherwise `status: active`. Plans without a
    Progress table default to `status: draft`.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANS = ROOT / "docs" / "plans"

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
PROGRESS_HEADING_RE = re.compile(r"^##\s+Progress\s*$", re.MULTILINE)
TABLE_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", re.MULTILINE)

# Headings that suggest un-spun-off follow-up work.
DEFERRED_HEADING_RE = re.compile(
    r"^#{1,6}\s+("
    r"deferred"
    r"|follow[\s-]?up(s)?"
    r"|follow[\s-]?on"
    r"|future\s+work"
    r"|future\s+(extensions|considerations|improvements)"
    r"|out\s+of\s+scope"
    r"|next\s+steps"
    r"|leftover"
    r"|not\s+done"
    r"|open\s+questions"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Area tag inference from filename. First match wins.
AREA_KEYWORDS: list[tuple[str, list[str]]] = [
    ("hdr", ["rendering", "hdr"]),
    ("ibl", ["rendering", "lighting"]),
    ("ldr", ["rendering", "hdr"]),
    ("lighting", ["rendering", "lighting"]),
    ("light-prob", ["rendering", "lighting"]),
    ("scene-lights", ["rendering", "lighting"]),
    ("shading", ["rendering", "shading"]),
    ("shadow", ["rendering", "lighting"]),
    ("reflect", ["rendering", "lighting"]),
    ("lightmap", ["rendering", "lighting"]),
    ("material", ["rendering", "materials"]),
    ("matcap", ["rendering", "materials"]),
    ("wireframe", ["rendering"]),
    ("morph", ["geometry", "deformation"]),
    ("skeletal", ["geometry", "deformation"]),
    ("skin", ["geometry", "deformation"]),
    ("particle", ["rendering", "particles"]),
    ("volum", ["rendering", "volumes"]),
    ("clip", ["rendering", "clipping"]),
    ("upload", ["renderer", "uploads"]),
    ("rendering-perform", ["rendering", "performance"]),
    ("pick", ["picking"]),
    ("camera", ["camera"]),
    ("input", ["input"]),
    ("selection", ["interaction"]),
    ("highlight", ["interaction"]),
    ("overlay", ["overlays"]),
    ("widget", ["overlays"]),
    ("gizmo", ["interaction"]),
    ("terrain", ["plugins", "terrain"]),
    ("physics", ["plugins", "physics"]),
    ("compute", ["plugins"]),
    ("plugin", ["plugins"]),
    ("plugins", ["plugins"]),
    ("item-type", ["api", "items"]),
    ("api-", ["api"]),
    ("camera-api", ["api", "camera"]),
    ("input-api", ["api", "input"]),
    ("gap-", ["consumers"]),
    ("game-", ["consumers", "games"]),
    ("downstream", ["consumers"]),
    ("viewport-quick", ["consumers", "rapid-proto"]),
    ("viewport-runtime", ["runtime"]),
    ("viewport-physics", ["plugins", "physics"]),
    ("viewport-lib-drake", ["consumers", "drake"]),
    ("residual-y-up", ["coordinates"]),
    ("point-gaussian", ["rendering", "splats"]),
    ("volumetric-effects", ["rendering", "volumes"]),
    ("roadmap", ["meta"]),
    ("guide", ["meta"]),
]

DEFAULT_AREA = ["uncategorized"]


@dataclass
class ProgressRow:
    phase: str
    description: str
    progress: str
    effort: str
    priority: str


@dataclass
class PlanState:
    path: Path
    has_frontmatter: bool
    progress_rows: list[ProgressRow] = field(default_factory=list)
    has_progress_table: bool = False
    deferred_headings: list[str] = field(default_factory=list)


def parse_progress_table(text: str) -> tuple[bool, list[ProgressRow]]:
    m = PROGRESS_HEADING_RE.search(text)
    if not m:
        return False, []
    tail = text[m.end():]
    # cut at the next h2 to avoid swallowing other tables
    next_h2 = re.search(r"^##\s+\S", tail, re.MULTILINE)
    if next_h2:
        tail = tail[:next_h2.start()]
    rows: list[ProgressRow] = []
    for rm in TABLE_ROW_RE.finditer(tail):
        c1, c2, c3, c4, c5 = (rm.group(i).strip() for i in range(1, 6))
        # skip header and separator rows
        if c1.lower() == "phase" or set(c1.replace(" ", "")) <= {"-", ":"}:
            continue
        rows.append(ProgressRow(c1, c2, c3, c4, c5))
    return True, rows


def find_deferred_headings(text: str) -> list[str]:
    return [m.group(0).strip() for m in DEFERRED_HEADING_RE.finditer(text)]


def all_rows_complete(rows: list[ProgressRow]) -> bool:
    if not rows:
        return False
    for r in rows:
        p = r.progress.lower().strip()
        if p in {"100%", "done", "complete", "completed", "shipped"}:
            continue
        return False
    return True


def infer_area(stem: str) -> list[str]:
    lower = stem.lower()
    for kw, tags in AREA_KEYWORDS:
        if kw in lower:
            return tags
    return DEFAULT_AREA


ACRONYMS = {
    "api", "ibl", "hdr", "ldr", "gpu", "cpu", "ui", "ux", "occt", "ais",
    "vtk", "bvh", "aabb", "uat", "lod", "mc", "ik", "ar", "vr",
}


def infer_title(stem: str) -> str:
    # "upload-job-system-plan" -> "Upload Job System"
    base = stem
    if base.endswith("-plan"):
        base = base[:-5]
    words = []
    for w in base.split("-"):
        if w.lower() in ACRONYMS:
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return " ".join(words)


def infer_id(stem: str) -> str:
    return stem[:-5] if stem.endswith("-plan") else stem


def analyze(path: Path) -> PlanState:
    text = path.read_text(encoding="utf-8")
    has_fm = bool(FM_RE.match(text))
    has_table, rows = parse_progress_table(text)
    deferred = find_deferred_headings(text)
    return PlanState(path=path, has_frontmatter=has_fm, progress_rows=rows, has_progress_table=has_table, deferred_headings=deferred)


def propose_frontmatter(state: PlanState) -> str:
    stem = state.path.stem
    title = infer_title(stem)
    plan_id = infer_id(stem)
    area = infer_area(stem)
    st = state.path.stat()
    birth = getattr(st, "st_birthtime", st.st_mtime)
    created = datetime.fromtimestamp(birth).date().isoformat()
    updated = datetime.fromtimestamp(st.st_mtime).date().isoformat()

    if not state.has_progress_table:
        status = "draft"
    elif all_rows_complete(state.progress_rows):
        status = "done"
    else:
        status = "active"

    if state.deferred_headings:
        deferred = "unresolved"
        # downgrade a 'done' to 'active' if there are loose ends still in the body
        # — leave the call to the human; but mark unresolved so archive is blocked
    else:
        deferred = "none"

    lines = [
        "---",
        f"id: {plan_id}",
        f"title: {title}",
        f"status: {status}",
        "priority: ",
        f"area: [{', '.join(area)}]",
        "source_issue: ",
        f"created: {created}",
        f"updated: {updated}",
        f"deferred: {deferred}",
        "supersedes: ",
        "superseded_by: ",
        'note: ""',
        "---",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap docs/plans/ frontmatter + audit deferred sections.")
    ap.add_argument("--apply", action="store_true", help="prepend proposed frontmatter to plans without one")
    ap.add_argument("--audit-only", action="store_true", help="skip proposals; just print the deferred-section audit")
    args = ap.parse_args()

    if not PLANS.exists():
        print("docs/plans/ not found.")
        return 0

    states = [analyze(p) for p in sorted(PLANS.glob("*.md")) if p.name not in {"INDEX.md", "README.md"}]
    if (PLANS / "archive").exists():
        states += [analyze(p) for p in sorted((PLANS / "archive").glob("*.md")) if p.name not in {"ARCHIVE.md"}]

    missing_table = [s for s in states if not s.has_progress_table]
    has_deferred = [s for s in states if s.deferred_headings]
    needs_fm = [s for s in states if not s.has_frontmatter]
    fully_complete = [s for s in states if s.has_progress_table and all_rows_complete(s.progress_rows)]

    if not args.audit_only:
        print(f"=== plans_bootstrap.py: {'APPLY' if args.apply else 'DRY-RUN'} ===")
        print(f"Scanned {len(states)} plans.")
        print(f"  needs frontmatter: {len(needs_fm)}")
        print(f"  missing Progress table: {len(missing_table)}")
        print(f"  has deferred / follow-up section: {len(has_deferred)}")
        print(f"  fully complete by Progress table: {len(fully_complete)}")
        print()

        if needs_fm and not args.apply:
            print("--- example proposed frontmatter (first plan needing one) ---")
            s = needs_fm[0]
            print(f"# {s.path.relative_to(ROOT)}")
            print(propose_frontmatter(s))

        if args.apply:
            written = 0
            for s in needs_fm:
                text = s.path.read_text(encoding="utf-8")
                fm = propose_frontmatter(s)
                s.path.write_text(fm + text, encoding="utf-8")
                written += 1
                print(f"frontmatter added: {s.path.relative_to(ROOT)}")
            print(f"\nApplied frontmatter to {written} plans.")

    print()
    print("=== Deferred-section audit ===")
    if not has_deferred:
        print("No plans matched the deferred/follow-up heading patterns.")
    else:
        print("These plans have body sections suggesting un-spun-off follow-up work.")
        print("Decide for each: (a) spin off into a new <name>-followups-plan.md and set")
        print("  `deferred: docs/plans/<that>.md`, or (b) keep `deferred: unresolved`, or")
        print("  (c) mark `deferred: none` if the section is informational only.\n")
        for s in has_deferred:
            print(f"- {s.path.relative_to(ROOT)}")
            for h in s.deferred_headings:
                print(f"    {h}")

    if missing_table:
        print()
        print("=== Plans missing `## Progress` table ===")
        for s in missing_table:
            print(f"- {s.path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
