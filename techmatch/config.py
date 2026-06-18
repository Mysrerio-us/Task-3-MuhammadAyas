from __future__ import annotations
import os

from techmatch.exceptions import ConfigError

# dataset

CSV_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)),"raw_skills.csv")
CSV_CORPUS_SIZE: int = 421

# Recommendation Engine

TOP_N: int = 3

MIN_SKILLS: int = 3

MAX_SKILLS: int = 20

IDF_SMOOTHING: float = 1.0

# Additive smoothing constant applied inside the IDF formula Prevents division-by-zero and keeps values comparable across corpus sizes


OOV_IDF_DEFAULT: float = 1.0

#IDF weight assigned to out-of-vocabulary (OOV) terms 
#skills the user typed that don't appear in any job-role tag list.
#Set to 1.0 so OOV terms contribute equally to TF but add no IDF boost.


COLD_START_THRESHOLD: float = 1e-9
# Cold Start condition. Accounts for floating-point rounding.
SCORE_HIGH_THRESHOLD: float = 0.70
SCORE_MID_THRESHOLD: float = 0.40


# UI / Display

SCORE_BAR_WIDTH: int = 30
MAX_GAP_TAGS_SHOWN: int = 6

SPINNER_FRAME_DELAY: float = 0.07
# Seconds between spinner animation frames.
PIPELINE_STEP_DELAYS: dict[str, float] = {
    "ingestion": 0.7,
    "idf":       0.8,
    "tfidf":     0.7,
    "scoring":   0.9,
    "sorting":   0.5,
    "filtering": 0.4,
}

TABLE_COL_WIDTH: int = 25

# metadata

APP_NAME:    str = "TechMatch"
APP_VERSION: str = "3.0.0"
APP_TAGLINE: str = "AI Tech Stack Recommender"
APP_AUTHOR:  str = "Muhammad Ayas"
APP_BATCH:   str = "2026"
APP_PROJECT: str = "AI Recommendation Logic"

# Validation 

def validate() -> None:
  
    checks: list[tuple[bool, str, str]] = [
        (isinstance(TOP_N, int) and TOP_N >= 1,
         "TOP_N", "Must be a positive integer."),
        (isinstance(MIN_SKILLS, int) and MIN_SKILLS >= 1,
         "MIN_SKILLS", "Must be a positive integer."),
        (isinstance(MAX_SKILLS, int) and MAX_SKILLS >= MIN_SKILLS,
         "MAX_SKILLS", f"Must be >= MIN_SKILLS ({MIN_SKILLS})."),
        (0.0 <= IDF_SMOOTHING <= 10.0,
         "IDF_SMOOTHING", "Must be between 0.0 and 10.0."),
        (0.0 < COLD_START_THRESHOLD < 0.01,
         "COLD_START_THRESHOLD", "Must be a very small positive float (e.g. 1e-9)."),
        (0.0 < SCORE_HIGH_THRESHOLD <= 1.0,
         "SCORE_HIGH_THRESHOLD", "Must be between 0.0 and 1.0."),
        (0.0 < SCORE_MID_THRESHOLD < SCORE_HIGH_THRESHOLD,
         "SCORE_MID_THRESHOLD", f"Must be less than SCORE_HIGH_THRESHOLD ({SCORE_HIGH_THRESHOLD})."),
        (isinstance(SCORE_BAR_WIDTH, int) and SCORE_BAR_WIDTH >= 10,
         "SCORE_BAR_WIDTH", "Must be an integer >= 10."),
        (os.path.isfile(CSV_PATH),
         "CSV_PATH", f"raw_skills.csv not found at: {CSV_PATH}"),
    ]

    for condition, key, detail in checks:
        if not condition:
            raise ConfigError(key, detail)
