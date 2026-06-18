from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from techmatch import config
from techmatch.data import JOB_CORPUS, SKILL_CATEGORIES, validate_corpus
from techmatch.engine import recommend
from techmatch.exceptions import (
    TechMatchError,
    DuplicateSkillError,
    EmptyInputError,
    InsufficientSkillsError,
    InvalidSkillIndexError,
)
from techmatch import display as ui


# Logging setup

def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)s: %(message)s",
    )



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="techmatch",
        description="TechMatch — AI Tech Stack Recommender (Project 3)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


# Startup

def _startup_checks() -> None:
    try:
        config.validate()
        validate_corpus()
    except TechMatchError as exc:
        print(f"\n  [STARTUP ERROR] {exc}\n")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  [UNEXPECTED STARTUP ERROR] {exc}\n")
        sys.exit(1)


# Ingestion

def _get_user_skills() -> list[str]:
    ui.print_header()
    print(ui.bold("  STEP 1 — SKILL INGESTION"))
    print(ui.dim(
        f"  Select your skills by number or type them directly.\n"
        f"  Minimum {config.MIN_SKILLS} required · Maximum {config.MAX_SKILLS} accepted."
    ))

    all_skills = ui.print_skill_menu(SKILL_CATEGORIES)
    selected: list[str] = []

    while True:
        ui.rule()
        ui.print_selected_chips(selected)
        ui.blank()
        print(ui.dim("  Enter numbers (e.g. 1,3,7), skill name, or press Enter when done."))
        print(ui.dim("  Type 'q' to quit."))

        try:
            raw = input(f"  {ui.cyan('›')} ").strip()
        except (EOFError, KeyboardInterrupt):
            _graceful_exit()

        # Quit
        if raw.lower() in ("q", "quit", "exit"):
            _graceful_exit()

        # Empty → try to submit
        if raw == "":
            try:
                _validate_submission(selected)
                break
            except InsufficientSkillsError as exc:
                print(ui.red(f"\n  ✗  {exc.message}\n"))
                continue

        # Parse tokens
        tokens = [t.strip() for t in raw.replace(" ", ",").split(",") if t.strip()]
        _process_tokens(tokens, all_skills, selected)

    return selected


def _validate_submission(selected: list[str]) -> None:
    if len(selected) < config.MIN_SKILLS:
        raise InsufficientSkillsError(
            provided=len(selected),
            required=config.MIN_SKILLS,
        )


def _process_tokens(
    tokens: list[str],
    all_skills: list[str],
    selected: list[str],
) -> None:
    added: list[str] = []

    for token in tokens:
        if len(selected) >= config.MAX_SKILLS:
            print(ui.yellow(
                f"\n  ⚠  Maximum {config.MAX_SKILLS} skills reached. "
                "Press Enter to continue.\n"
            ))
            break

        if token.isdigit():
            idx = int(token) - 1
            if not (0 <= idx < len(all_skills)):
                err = InvalidSkillIndexError(index=int(token), max_index=len(all_skills))
                print(ui.red(f"\n  ✗  {err.message}"))
                continue
            skill = all_skills[idx]
        else:
            skill = token.strip().title()

        skill_lower = skill.lower()
        if skill_lower in (s.lower() for s in selected):
            warn = DuplicateSkillError(skill)
            print(ui.yellow(f"\n  ⚠  {warn.message}"))
            continue

        selected.append(skill)
        added.append(skill)

    if added:
        print(ui.green(f"\n  ✓  Added: {', '.join(added)}\n"))


# pipeline run

def _run_pipeline(selected_skills: list[str]) -> dict:
    delays = config.PIPELINE_STEP_DELAYS

    ui.print_header()
    print(ui.bold("  RUNNING RECOMMENDATION ENGINE"))
    ui.blank()

    ui.print_pipeline(1)
    ui.spinner(f"Ingesting {len(selected_skills)} skill(s) into user profile…", delays["ingestion"])
    skills_preview = ", ".join(s.lower() for s in selected_skills[:6])
    if len(selected_skills) > 6:
        skills_preview += "…"
    print(ui.dim(f"  Profile: {skills_preview}"))
    ui.blank()

    ui.print_pipeline(2)
    ui.spinner("Computing IDF weights across 421-posting CSV corpus…", delays["idf"])
    ui.spinner("Building TF-IDF vectors for user + all role clusters…", delays["tfidf"])
    ui.spinner(f"Calculating cosine similarity for {len(JOB_CORPUS)} role clusters…", delays["scoring"])
    ui.blank()

    ui.print_pipeline(3)
    ui.spinner("Sorting roles by similarity score (descending)…", delays["sorting"])
    ui.blank()

    ui.print_pipeline(4)
    ui.spinner("Filtering to Top-3 results…", delays["filtering"])
    ui.blank()

    # engine call
    output = recommend(selected_skills, JOB_CORPUS, top_n=config.TOP_N)
    return output


# main loop

def _graceful_exit() -> None:
    ui.blank()
    print(ui.dim("  Thanks for using TechMatch. Good luck with your career path!"))
    ui.blank()
    sys.exit(0)


def main() -> None:
    args = _parse_args()

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    _setup_logging(args.debug)
    _startup_checks()

    while True:
        try:
            # Step 1
            selected_skills = _get_user_skills()

            # Step 2–4
            output = _run_pipeline(selected_skills)

            # Output
            ui.print_results(
                results        = output["results"],
                selected_skills = selected_skills,
                cold_start     = output["cold_start"],
            )

            # Transparency layer
            print(ui.dim("  Press T to view TF-IDF weight breakdown, or Enter to continue."))
            try:
                choice = input(f"  {ui.cyan('›')} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                _graceful_exit()

            if choice == "t":
                ui.print_tfidf_debug(output["user_vec"], output["idf"])

            # Exit
            ui.rule()
            ui.blank()
            print(ui.dim("  Press R to run again with different skills, or any other key to quit."))
            try:
                again = input(f"  {ui.cyan('›')} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                _graceful_exit()

            if again != "r":
                _graceful_exit()

        except TechMatchError as exc:
            # Local Handling
            ui.blank()
            print(ui.red(f"  [ERROR] {exc.message}"))
            if exc.hint:
                print(ui.dim(f"  Hint: {exc.hint}"))
            ui.blank()
            print(ui.dim("  Press R to try again, or any other key to quit."))
            try:
                retry = input(f"  {ui.cyan('›')} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                _graceful_exit()
            if retry != "r":
                _graceful_exit()

        except KeyboardInterrupt:
            _graceful_exit()

        except Exception as exc:  # noqa: BLE001
            # Unexpected errors
            logging.exception("Unexpected error")
            ui.blank()
            print(ui.red("  [UNEXPECTED ERROR] Something went wrong."))
            print(ui.dim(f"  Detail: {exc}"))
            print(ui.dim("  Run with --debug for full traceback."))
            ui.blank()
            sys.exit(1)


if __name__ == "__main__":
    main()
