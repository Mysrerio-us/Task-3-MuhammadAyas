from __future__ import annotations

import os
import sys
import time
import logging

from techmatch import config

logger = logging.getLogger(__name__)

# ANSI colour support detection 

def _supports_color() -> bool:

    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":
        return os.environ.get("TERM_PROGRAM") == "vscode" or "WT_SESSION" in os.environ
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


class C:
    """ANSI escape sequences.  All codes degrade gracefully to empty strings."""
    RESET   = "\033[0m"  if _COLOR else ""
    BOLD    = "\033[1m"  if _COLOR else ""
    DIM     = "\033[2m"  if _COLOR else ""
    CYAN    = "\033[96m" if _COLOR else ""
    GREEN   = "\033[92m" if _COLOR else ""
    YELLOW  = "\033[93m" if _COLOR else ""
    RED     = "\033[91m" if _COLOR else ""
    WHITE   = "\033[97m" if _COLOR else ""
    MAGENTA = "\033[95m" if _COLOR else ""


# Convenience wrappers
def cyan(s: str)    -> str: return f"{C.CYAN}{s}{C.RESET}"
def green(s: str)   -> str: return f"{C.GREEN}{s}{C.RESET}"
def yellow(s: str)  -> str: return f"{C.YELLOW}{s}{C.RESET}"
def red(s: str)     -> str: return f"{C.RED}{s}{C.RESET}"
def bold(s: str)    -> str: return f"{C.BOLD}{s}{C.RESET}"
def dim(s: str)     -> str: return f"{C.DIM}{s}{C.RESET}"
def magenta(s: str) -> str: return f"{C.MAGENTA}{s}{C.RESET}"


# Primitives

def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def rule(char: str = "─", width: int = 62) -> None:
    print(dim(char * width))


def blank() -> None:
    print()


# ── Header / Branding ────────────────────────────────────────────────────────

LOGO = r"""
  ████████╗███████╗ ██████╗██╗  ██╗
     ██╔══╝██╔════╝██╔════╝██║  ██║
     ██║   █████╗  ██║     ███████║
     ██║   ██╔══╝  ██║     ██╔══██║
     ██║   ███████╗╚██████╗██║  ██║
     ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝
"""


def print_header() -> None:

    clear_screen()
    print(cyan(LOGO))
    print(f"  {bold(config.APP_NAME)}  {dim('·')}  {dim(config.APP_TAGLINE)}")
    print(f"  {dim(config.APP_AUTHOR + ' · ' + config.APP_PROJECT + ' · Batch ' + config.APP_BATCH)}")
    blank()
    rule()


# Pipeline bar

_PIPELINE_STEPS = ["INGESTION", "SCORING", "SORTING", "FILTERING"]


def print_pipeline(active_step: int) -> None:

    parts: list[str] = []
    for i, name in enumerate(_PIPELINE_STEPS, start=1):
        if i < active_step:
            parts.append(green(f"[{i}:{name} ✓]"))
        elif i == active_step:
            parts.append(cyan(bold(f"[{i}:{name}]")))
        else:
            parts.append(dim(f"[{i}:{name}]"))
        if i < len(_PIPELINE_STEPS):
            parts.append(dim(" → "))
    print("  " + "".join(parts))
    blank()


# Spinner animation

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def spinner(message: str, duration: float) -> None:

    if not _COLOR or not sys.stdout.isatty():
        print(f"  ✓  {message}")
        return

    end_time = time.monotonic() + duration
    frame = 0
    try:
        while time.monotonic() < end_time:
            print(
                f"\r  {cyan(_SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)])}  {dim(message)}",
                end="",
                flush=True,
            )
            time.sleep(config.SPINNER_FRAME_DELAY)
            frame += 1
    finally:
        # Always clear the line and print the success tick
        print(f"\r  {green('✓')}  {dim(message)}{' ' * 10}")


#n nScorebar

def score_bar(score: float) -> str:

    width = config.SCORE_BAR_WIDTH
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = round(score * 100)

    if pct >= round(config.SCORE_HIGH_THRESHOLD * 100):
        colour_fn = green
    elif pct >= round(config.SCORE_MID_THRESHOLD * 100):
        colour_fn = yellow
    else:
        colour_fn = red

    return colour_fn(bar) + f"  {bold(str(pct) + '%')}"


# skill pick menu

def print_skill_menu(skill_categories: dict[str, list[str]]) -> list[str]:

    all_skills: list[str] = []
    blank()
    rule()
    print(bold("  AVAILABLE SKILLS"))
    rule()

    for category, skills in skill_categories.items():
        print(f"\n  {cyan(category)}")
        row: list[str] = []
        for skill in skills:
            idx = len(all_skills) + 1
            all_skills.append(skill)
            row.append(f"  {dim(str(idx).rjust(3))}) {skill:<22}")
            if len(row) == 3:
                print("".join(row))
                row = []
        if row:
            print("".join(row))

    blank()
    return all_skills


def print_selected_chips(selected: list[str]) -> None:
    if not selected:
        print(dim("  No skills selected yet."))
    else:
        chips = "   ".join(cyan(f"[{s}]") for s in selected)
        print(f"  Selected: {chips}")
        print(dim(f"  Count: {len(selected)} skill(s)"))


# Result

_RANK_LABELS  = ["🥇 BEST MATCH", "🥈 2ND MATCH", "🥉 3RD MATCH"]
_RANK_COLOURS = [cyan, yellow, dim]


def print_results(
    results: list[dict],
    selected_skills: list[str],
    cold_start: bool,
) -> None:

    print_header()
    print(bold("  TOP CAREER PATH RECOMMENDATIONS"))
    print(dim(
        f"  {len(selected_skills)} skill(s) evaluated  ·  "
        f"12 roles scored  ·  Top {len(results)} shown"
    ))
    rule()

    if cold_start:
        blank()
        print(yellow("  ⚠  COLD START DETECTED"))
        print(dim("  None of your skills matched the corpus vocabulary."))
        print(dim("  Showing globally trending roles as a fallback."))
        print(dim("  Tip: add skills from the catalogue for personalised results."))
        blank()

    for i, role in enumerate(results):
        _print_role_card(role, i, cold_start)

    # Summary line
    if results:
        top  = results[0]
        pct  = round(top.get("score", top.get("popularity", 0)) * 100)
        blank()
        print(green(
            f"  ✦  Strongest alignment: '{top['title']}' ({pct}% match)."
        ))
    blank()


def _print_role_card(role: dict, rank: int, cold_start: bool) -> None:

    score   = role.get("score", role.get("popularity", 0.0))
    matched = role.get("matched_tags", [])
    gap     = role.get("gap_tags", [])
    colour  = _RANK_COLOURS[rank]

    blank()
    print(f"  {colour(_RANK_LABELS[rank])}")
    print(f"  {bold(role['title'])}")
    blank()
    print(f"  Match Score   {score_bar(score)}")
    blank()
    print(f"  {dim(role['description'])}")
    blank()

    if not cold_start:
        if matched:
            hits = "   ".join(green(f"✓ {t}") for t in matched)
            print(f"  Matched:  {hits}")
        if gap:
            shown = gap[:config.MAX_GAP_TAGS_SHOWN]
            extra = f"  {dim(f'+{len(gap) - config.MAX_GAP_TAGS_SHOWN} more')}" \
                    if len(gap) > config.MAX_GAP_TAGS_SHOWN else ""
            gaps  = "   ".join(dim(f"○ {t}") for t in shown)
            print(f"  Gaps:     {gaps}{extra}")

    rule("·")


# ── TF-IDF Transparency Panel ────────────────────────────────────────────────

def print_tfidf_debug(user_vec: dict, idf: dict) -> None:

    if not user_vec:
        print(yellow("  No vector data available (cold start was active)."))
        return

    blank()
    rule()
    print(bold("  TRANSPARENCY LAYER — TF-IDF PROFILE WEIGHTS"))
    print(dim("  Higher weight → more specific / distinctive skill"))
    rule()
    blank()

    sorted_vec = sorted(user_vec.items(), key=lambda x: x[1], reverse=True)
    max_w = sorted_vec[0][1] if sorted_vec else 1.0
    col   = config.TABLE_COL_WIDTH

    for term, weight in sorted_vec:
        bar_len  = round((weight / max_w) * 25)
        bar      = cyan("▪" * bar_len) + dim("·" * (25 - bar_len))
        print(f"  {term:<{col}} {bar}  {dim(f'{weight:.4f}')}")

    blank()
    print(dim("  IDF scores (lower IDF = more generic across all roles):"))
    blank()

    idf_subset = sorted(
        [(t, v) for t, v in idf.items() if t in user_vec],
        key=lambda x: x[1],
    )
    for term, val in idf_subset:
        if val < 1.5:
            label, colour_fn = "generic",  red
        elif val > 2.5:
            label, colour_fn = "specific", green
        else:
            label, colour_fn = "moderate", yellow
        print(f"  {term:<{col}} IDF={val:.3f}  {colour_fn(label)}")

    blank()
