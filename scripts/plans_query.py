#!/usr/bin/env python3
"""Query Progress tables in docs/plans/.

Three subcommands:

  show [SELECTOR ...]  prettified Progress tables for selected plans
  stats [SELECTOR ...] per-plan completion rollup (% complete, phase counts)
  list [filters]       plans matching completion / status filters

A SELECTOR is either:
  - a file path (with `/`, `.md`, or that resolves to an existing file) — supports shell tab-completion
  - a substring matched against plan id / filename (multiple plans may match)

Examples:
  plans_query.py show                                          # all plans
  plans_query.py show docs/plans/lighting-shading-consistency-plan.md
  plans_query.py show lighting-shading                         # substring
  plans_query.py show gap-                                     # all matching plans
  plans_query.py stats
  plans_query.py stats --status active
  plans_query.py list --almost-done                            # 80-99% complete
  plans_query.py list --min 50 --status active
  plans_query.py list --max 0                                  # not started

Plan-level completion = simple mean of phase progress values. Phases with
non-numeric progress (e.g. 'n/a') are skipped. Plans with `status: rough`
are excluded by default (no Progress table to aggregate).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANS = ROOT / "docs" / "plans"
ARCHIVE = PLANS / "archive"

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
PROGRESS_HEADING_RE = re.compile(r"^##\s+Progress\s*$", re.MULTILINE)
TABLE_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$",
    re.MULTILINE,
)

PERCENT_RE = re.compile(r"^(\d{1,3})\s*%$")
DONE_WORDS = {"done", "complete", "completed", "shipped", "landed"}


@dataclass
class Frontmatter:
    id: str = ""
    title: str = ""
    status: str = ""
    priority: str = ""
    area: list[str] = field(default_factory=list)
    deferred: str = ""
    note: str = ""


@dataclass
class PhaseRow:
    phase: str
    description: str
    progress_raw: str
    effort: str
    priority: str

    @property
    def progress_pct(self) -> float | None:
        v = self.progress_raw.strip().lower()
        if v in DONE_WORDS:
            return 100.0
        m = PERCENT_RE.match(v)
        if m:
            return float(m.group(1))
        return None


@dataclass
class Plan:
    path: Path
    archived: bool
    fm: Frontmatter
    rows: list[PhaseRow] = field(default_factory=list)

    @property
    def numeric_rows(self) -> list[PhaseRow]:
        return [r for r in self.rows if r.progress_pct is not None]

    @property
    def completion_pct(self) -> float | None:
        nums = [r.progress_pct for r in self.numeric_rows]
        if not nums:
            return None
        return sum(nums) / len(nums)  # type: ignore[arg-type]

    @property
    def phase_summary(self) -> str:
        """e.g. '7/9' (completed/total). Counts 100% rows as completed."""
        if not self.rows:
            return "—"
        total = len(self.rows)
        done_count = sum(1 for r in self.rows if (r.progress_pct or 0) >= 100)
        partial = sum((r.progress_pct or 0) / 100.0 for r in self.rows if 0 < (r.progress_pct or 0) < 100)
        # Show fractional completion if any phase is partial
        effective = done_count + partial
        if effective == int(effective):
            return f"{int(effective)}/{total}"
        return f"{effective:.1f}/{total}"


# ---------- parsing ----------

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
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if hasattr(fm, key):
                setattr(fm, key, val)
    return fm


def parse_rows(text: str) -> list[PhaseRow]:
    m = PROGRESS_HEADING_RE.search(text)
    if not m:
        return []
    tail = text[m.end():]
    next_h2 = re.search(r"^##\s+\S", tail, re.MULTILINE)
    if next_h2:
        tail = tail[:next_h2.start()]
    rows: list[PhaseRow] = []
    for rm in TABLE_ROW_RE.finditer(tail):
        c1, c2, c3, c4, c5 = (rm.group(i).strip() for i in range(1, 6))
        # skip header row (case-insensitive)
        if c1.lower() == "phase":
            continue
        # skip separator row
        if set(c1.replace(" ", "")) <= {"-", ":"}:
            continue
        rows.append(PhaseRow(c1, c2, c3, c4, c5))
    return rows


def load_plan(path: Path, archived: bool) -> Plan | None:
    text = path.read_text(encoding="utf-8")
    fm = parse_frontmatter(text)
    if fm is None:
        return None
    return Plan(path=path, archived=archived, fm=fm, rows=parse_rows(text))


def load_all_plans() -> list[Plan]:
    out: list[Plan] = []
    for p in sorted(PLANS.glob("*.md")):
        if p.name in {"INDEX.md", "README.md"}:
            continue
        plan = load_plan(p, archived=False)
        if plan is not None:
            out.append(plan)
    if ARCHIVE.exists():
        for p in sorted(ARCHIVE.glob("*.md")):
            if p.name == "ARCHIVE.md":
                continue
            plan = load_plan(p, archived=True)
            if plan is not None:
                out.append(plan)
    return out


# ---------- selection ----------

def select_plans(plans: list[Plan], selectors: list[str]) -> list[Plan]:
    """If no selectors, return everything. Otherwise match each selector and union results."""
    if not selectors:
        return plans
    by_path = {p.path: p for p in plans}
    by_path_str = {str(p.path): p for p in plans}
    by_relpath = {str(p.path.relative_to(ROOT)): p for p in plans}
    out: list[Plan] = []
    seen: set[Path] = set()
    for sel in selectors:
        matched = _match_selector(plans, sel, by_path, by_path_str, by_relpath)
        if not matched:
            print(f"warning: no plans matched selector {sel!r}", file=sys.stderr)
            continue
        for p in matched:
            if p.path not in seen:
                out.append(p)
                seen.add(p.path)
    return out


def _match_selector(plans, sel, by_path, by_path_str, by_relpath) -> list[Plan]:
    # Path mode if it has a slash or ends in .md or exists as a file
    looks_like_path = "/" in sel or sel.endswith(".md")
    if looks_like_path:
        p = Path(sel)
        candidates = [p, ROOT / p, ROOT / "docs" / "plans" / p]
        for c in candidates:
            c = c.resolve() if c.exists() else c
            if c in by_path:
                return [by_path[c]]
            if str(c) in by_path_str:
                return [by_path_str[str(c)]]
        # try as relative-to-root
        if sel in by_relpath:
            return [by_relpath[sel]]
    # Substring match on id and stem
    needle = sel.lower()
    return [p for p in plans if needle in p.fm.id.lower() or needle in p.path.stem.lower()]


# ---------- output ----------

def fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    if v == 100:
        return "100%"
    if v == 0:
        return "0%"
    return f"{v:.0f}%"


def status_label(plan: Plan) -> str:
    base = plan.fm.status or "?"
    if plan.archived:
        return f"{base} (archived)"
    return base


def cmd_show(plans: list[Plan]) -> int:
    if not plans:
        print("no plans matched.")
        return 1
    for i, plan in enumerate(plans):
        if i > 0:
            print()
        rel = plan.path.relative_to(ROOT)
        title = plan.fm.title or plan.fm.id
        comp = fmt_pct(plan.completion_pct)
        print(f"# {title}  [{comp} · {plan.phase_summary} · {status_label(plan)}]")
        print(f"  {rel}")
        print()
        if not plan.rows:
            print("  (no Progress table)")
            continue
        # column widths
        widths = [
            max(5, max(len(r.phase) for r in plan.rows)),
            max(11, max(len(r.description) for r in plan.rows)),
            max(8, max(len(r.progress_raw) for r in plan.rows)),
            max(6, max(len(r.effort) for r in plan.rows)),
            max(8, max(len(r.priority) for r in plan.rows)),
        ]
        header = ("Phase", "Description", "Progress", "Effort", "Priority")
        fmt = "  {:<{w0}}  {:<{w1}}  {:<{w2}}  {:<{w3}}  {:<{w4}}"
        print(fmt.format(*header, w0=widths[0], w1=widths[1], w2=widths[2], w3=widths[3], w4=widths[4]))
        print("  " + "  ".join("-" * w for w in widths))
        for r in plan.rows:
            print(fmt.format(r.phase, r.description, r.progress_raw, r.effort, r.priority,
                             w0=widths[0], w1=widths[1], w2=widths[2], w3=widths[3], w4=widths[4]))
    return 0


def cmd_stats(plans: list[Plan]) -> int:
    if not plans:
        print("no plans matched.")
        return 1
    # sort by completion descending (None last)
    plans_sorted = sorted(plans, key=lambda p: (p.completion_pct is None, -(p.completion_pct or 0), p.fm.id))
    name_w = max(20, max(len(p.fm.id) for p in plans_sorted))
    phase_w = max(7, max(len(p.phase_summary) for p in plans_sorted))
    status_w = max(8, max(len(status_label(p)) for p in plans_sorted))
    fmt = "{:<{nw}}  {:>8}  {:>{pw}}  {:<{sw}}"
    print(fmt.format("plan", "progress", "phases", "status", nw=name_w, pw=phase_w, sw=status_w))
    print(fmt.format("-" * name_w, "-" * 8, "-" * phase_w, "-" * status_w, nw=name_w, pw=phase_w, sw=status_w))
    for p in plans_sorted:
        print(fmt.format(
            p.fm.id, fmt_pct(p.completion_pct), p.phase_summary, status_label(p),
            nw=name_w, pw=phase_w, sw=status_w,
        ))
    # totals
    total_phases = sum(len(p.rows) for p in plans_sorted)
    total_done = sum(1 for p in plans_sorted for r in p.rows if (r.progress_pct or 0) >= 100)
    plans_with_rows = [p for p in plans_sorted if p.completion_pct is not None]
    if plans_with_rows:
        avg = sum(p.completion_pct for p in plans_with_rows) / len(plans_with_rows)  # type: ignore[misc]
        print()
        print(f"totals: {len(plans_sorted)} plans, {total_phases} phases ({total_done} at 100%), mean completion {avg:.0f}%")
    return 0


def cmd_list(plans: list[Plan], args) -> int:
    sel = plans
    if args.status:
        sel = [p for p in sel if p.fm.status == args.status]
    if args.archived_only:
        sel = [p for p in sel if p.archived]
    if args.active_only:
        sel = [p for p in sel if not p.archived]
    sel = [p for p in sel if p.completion_pct is not None]
    if args.min is not None:
        sel = [p for p in sel if (p.completion_pct or 0) >= args.min]
    if args.max is not None:
        sel = [p for p in sel if (p.completion_pct or 0) <= args.max]
    if not sel:
        print("no plans matched filters.")
        return 1
    return cmd_stats(sel)


# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="Query docs/plans/ Progress tables.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_show = sub.add_parser("show", help="show Progress tables")
    ap_show.add_argument("selectors", nargs="*", help="path or name substring; multiple allowed; empty = all plans")

    ap_stats = sub.add_parser("stats", help="per-plan completion rollup")
    ap_stats.add_argument("selectors", nargs="*", help="path or name substring; empty = all plans")
    ap_stats.add_argument("--status", help="filter by status (draft/active/done/abandoned/rough)")
    ap_stats.add_argument("--active-only", action="store_true", help="exclude archived plans")
    ap_stats.add_argument("--archived-only", action="store_true", help="only archived plans")

    ap_list = sub.add_parser("list", help="filter plans by completion %")
    ap_list.add_argument("--min", type=float, help="min completion %")
    ap_list.add_argument("--max", type=float, help="max completion %")
    ap_list.add_argument("--almost-done", action="store_true", help="alias for --min 80 --max 99")
    ap_list.add_argument("--not-started", action="store_true", help="alias for --max 0")
    ap_list.add_argument("--status", help="filter by status")
    ap_list.add_argument("--active-only", action="store_true", help="exclude archived plans")
    ap_list.add_argument("--archived-only", action="store_true", help="only archived plans")

    args = ap.parse_args()
    plans = load_all_plans()

    if args.cmd == "show":
        selected = select_plans(plans, args.selectors)
        return cmd_show(selected)

    if args.cmd == "stats":
        selected = select_plans(plans, args.selectors)
        if args.status:
            selected = [p for p in selected if p.fm.status == args.status]
        if args.archived_only:
            selected = [p for p in selected if p.archived]
        if args.active_only:
            selected = [p for p in selected if not p.archived]
        return cmd_stats(selected)

    if args.cmd == "list":
        if args.almost_done:
            args.min = 80 if args.min is None else args.min
            args.max = 99 if args.max is None else args.max
        if args.not_started:
            args.max = 0 if args.max is None else args.max
        return cmd_list(plans, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
