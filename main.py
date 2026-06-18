

import os
import time
from data import JOB_CORPUS, SKILL_CATEGORIES
from engine import recommend, build_tfidf_vector, compute_idf


class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    WHITE  = "\033[97m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"

def cyan(s):    return f"{C.CYAN}{s}{C.RESET}"
def green(s):   return f"{C.GREEN}{s}{C.RESET}"
def yellow(s):  return f"{C.YELLOW}{s}{C.RESET}"
def red(s):     return f"{C.RED}{s}{C.RESET}"
def bold(s):    return f"{C.BOLD}{s}{C.RESET}"
def dim(s):     return f"{C.DIM}{s}{C.RESET}"
def magenta(s): return f"{C.MAGENTA}{s}{C.RESET}"


def clear():
    os.system("cls" if os.name == "nt" else "clear")

def line(char="─", width=60):
    print(dim(char * width))

def header():
    clear()
    print()
    print(cyan("  ████████╗███████╗ ██████╗██╗  ██╗"))
    print(cyan("     ██╔══╝██╔════╝██╔════╝██║  ██║"))
    print(cyan("     ██║   █████╗  ██║     ███████║"))
    print(cyan("     ██║   ██╔══╝  ██║     ██╔══██║"))
    print(cyan("     ██║   ███████╗╚██████╗██║  ██║"))
    print(cyan("     ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝"))
    print()
    print(bold("  TechMatch") + dim(" · AI Tech Stack Recommender"))
    print(dim("  DecodeLabs · Project 3 · Batch 2026"))
    print()
    line()

def pipeline_bar(active_step: int):
    
    steps = ["INGESTION", "SCORING", "SORTING", "FILTERING"]
    parts = []
    for i, name in enumerate(steps, 1):
        if i < active_step:
            parts.append(green(f"[{i}:{name} ✓]"))
        elif i == active_step:
            parts.append(cyan(bold(f"[{i}:{name}]")))
        else:
            parts.append(dim(f"[{i}:{name}]"))
        if i < 4:
            parts.append(dim(" → "))
    print("  " + "".join(parts))
    print()

def spinner(msg: str, duration: float = 0.6):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end = time.time() + duration
    i = 0
    while time.time() < end:
        print(f"\r  {cyan(frames[i % len(frames)])}  {dim(msg)}", end="", flush=True)
        time.sleep(0.07)
        i += 1
    print(f"\r  {green('✓')}  {dim(msg)}{' ' * 10}")


#  Skill picker

def show_skill_menu():
    all_skills = []
    print()
    line()
    print(bold("  AVAILABLE SKILLS"))
    line()

    for category, skills in SKILL_CATEGORIES.items():
        print(f"\n  {cyan(category)}")
        row = []
        for skill in skills:
            idx = len(all_skills) + 1
            all_skills.append(skill)
            row.append(f"  {dim(str(idx).rjust(3))}) {skill:<22}")
            if len(row) == 3:
                print("".join(row))
                row = []
        if row:
            print("".join(row))

    print()
    return all_skills


def get_user_skills() -> list[str]:

    header()
    print(bold("  STEP 1 — SKILL INGESTION"))
    print(dim("  Select your skills by number (e.g. 1,5,12) or type them directly."))
    print(dim("  Minimum 3 required for accurate recommendations.\n"))

    all_skills = show_skill_menu()

    selected = []

    while True:
        line()
        if selected:
            chips = "  Selected: " + "  ".join(cyan(f"[{s}]") for s in selected)
            print(chips)
            print(dim(f"  Count: {len(selected)} skill(s) selected"))
        else:
            print(dim("  No skills selected yet."))

        print()
        print(dim("  Enter skill numbers (e.g. 1,3,7), type a skill name, or press Enter when done."))
        raw = input(f"  {cyan('›')} ").strip()

        if raw == "" and len(selected) >= 3:
            break
        elif raw == "" and len(selected) < 3:
            print(red(f"\n  ✗  Need at least 3 skills. You have {len(selected)}.\n"))
            continue
        elif raw.lower() in ("q", "quit", "exit"):
            print(dim("\n  Goodbye!\n"))
            raise SystemExit

        # Parse: numbers, ranges, or plain text
        added = []
        parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]

        for part in parts:
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(all_skills):
                    skill = all_skills[idx]
                    if skill not in selected:
                        selected.append(skill)
                        added.append(skill)
                    else:
                        print(yellow(f"  Already added: {skill}"))
                else:
                    print(red(f"  ✗  Invalid number: {part}"))
            else:
                # Treat as custom typed skill
                skill = part.strip().title()
                if skill and skill not in selected:
                    selected.append(skill)
                    added.append(skill)

        if added:
            print(green(f"\n  ✓  Added: {', '.join(added)}\n"))

    return selected



#  Pipeline runner


def run_pipeline_animated(selected_skills: list[str]) -> dict:
    header()
    print(bold("  RUNNING RECOMMENDATION ENGINE\n"))

    # Step 1 — Ingestion
    pipeline_bar(1)
    spinner(f"Ingesting {len(selected_skills)} skill(s) into user profile…", 0.7)
    print(dim(f"  Profile: {', '.join(s.lower() for s in selected_skills[:6])}{'...' if len(selected_skills) > 6 else ''}"))
    print()

    # Step 2 — Scoring
    pipeline_bar(2)
    spinner("Computing IDF weights across 12-role corpus…", 0.8)
    spinner("Building TF-IDF vectors for user + all roles…", 0.7)
    spinner(f"Calculating cosine similarity for {len(JOB_CORPUS)} roles…", 0.9)
    print()

    # Step 3 — Sorting
    pipeline_bar(3)
    spinner("Sorting roles by similarity score (descending)…", 0.5)
    print()

    # Step 4 — Filtering
    pipeline_bar(4)
    spinner("Filtering to Top-3 results…", 0.4)
    print()

    # run the engine
    output = recommend(selected_skills, JOB_CORPUS, top_n=3)
    return output


# Results render

RANK_LABELS  = ["🥇 BEST MATCH", "🥈 2ND MATCH", "🥉 3RD MATCH"]
RANK_COLOURS = [cyan, yellow, dim]


def score_bar(score: float, width: int = 30) -> str:
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = round(score * 100)
    if pct >= 70:
        colour = green
    elif pct >= 40:
        colour = yellow
    else:
        colour = red
    return colour(bar) + f"  {bold(str(pct) + '%')}"


def render_results(output: dict, selected_skills: list[str]):
    header()
    results   = output["results"]
    user_vec  = output["user_vec"]
    cold_start = output["cold_start"]

    print(bold("  TOP CAREER PATH RECOMMENDATIONS"))
    print(dim(f"  {len(selected_skills)} skills input  ·  {len(JOB_CORPUS)} roles evaluated  ·  Top 3 shown"))
    line()

    if cold_start:
        print()
        print(yellow("  ⚠  COLD START DETECTED"))
        print(dim("  Your skills didn't match the corpus vocabulary."))
        print(dim("  Showing globally trending roles as fallback.\n"))

    for i, role in enumerate(results):
        score = role.get("score", role.get("popularity", 0))
        matched = role.get("matched_tags", [])
        gap = role.get("gap_tags", [])

        colour = RANK_COLOURS[i]
        print()
        print(f"  {colour(RANK_LABELS[i])}")
        print(f"  {bold(role['title'])}")
        print()

        # Score bar
        print(f"  Match Score   {score_bar(score)}")
        print()

        # Description
        print(f"  {dim(role['description'])}")
        print()

        # Matched skills (green)
        if matched:
            hits = "  ".join(green(f"✓ {t}") for t in matched)
            print(f"  Matched:  {hits}")

        # Skill gaps (dim)
        if gap:
            gaps_to_show = gap[:6]
            gaps = "  ".join(dim(f"○ {t}") for t in gaps_to_show)
            extra = f"  {dim(f'+{len(gap)-6} more')}" if len(gap) > 6 else ""
            print(f"  Gaps:     {gaps}{extra}")

        line("·")

    # Summary
    if results:
        top = results[0]
        top_score = round(top.get("score", 0) * 100)
        print()
        print(green(f"  ✦  Your strongest alignment is with '{top['title']}' ({top_score}% match)."))

    print()


#  Transparancy layer — TF-IDF debug view
#  Shows the user's vector weights so students
#  understand WHY certain skills matter more.


def show_tfidf_debug(output: dict):
    user_vec = output["user_vec"]
    idf = output["idf"]

    if not user_vec:
        return

    print()
    line()
    print(bold("  TRANSPARENCY LAYER — TF-IDF PROFILE WEIGHTS"))
    print(dim("  Higher weight = more specific / distinctive skill"))
    line()
    print()

    sorted_vec = sorted(user_vec.items(), key=lambda x: x[1], reverse=True)
    max_w = sorted_vec[0][1] if sorted_vec else 1

    for term, weight in sorted_vec:
        bar_len = round((weight / max_w) * 25)
        bar = cyan("▪" * bar_len) + dim("·" * (25 - bar_len))
        print(f"  {term:<25} {bar}  {dim(f'{weight:.4f}')}")

    print()
    print(dim("  IDF scores (lower = more generic across all roles):"))
    idf_sorted = sorted(
        [(t, v) for t, v in idf.items() if t in user_vec],
        key=lambda x: x[1]
    )
    for term, val in idf_sorted:
        signal = "generic" if val < 1.5 else "specific" if val > 2.5 else "moderate"
        colour = red if signal == "generic" else green if signal == "specific" else yellow
        print(f"  {term:<25} IDF={val:.3f}  {colour(signal)}")
    print()



#  Main loop


def main():
    while True:
        try:
            # step 1
            selected_skills = get_user_skills()

            # step 2-4
            output = run_pipeline_animated(selected_skills)

            # output
            render_results(output, selected_skills)

            # transparency
            print(dim("  Press T to see TF-IDF weight breakdown, or Enter to continue."))
            choice = input(f"  {cyan('›')} ").strip().lower()
            if choice == "t":
                show_tfidf_debug(output)

            # loop?
            line()
            print()
            print(dim("  Press R to run again with different skills, or Q to quit."))
            choice = input(f"  {cyan('›')} ").strip().lower()
            if choice != "r":
                print(dim("\n  Thanks for using TechMatch! Good luck with your career path.\n"))
                break

        except (KeyboardInterrupt, EOFError):
            print(dim("\n\n  Interrupted. Goodbye!\n"))
            break


if __name__ == "__main__":
    main()
