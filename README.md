<div align="center">

# TechMatch

### AI Tech Stack Recommender

**Content-Based Filtering | TF-IDF Vector Weighting | Cosine Similarity**


---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-39%20passing-brightgreen)
![Dependencies](https://img.shields.io/badge/Dependencies-zero-brightgreen)

</div>

---

## What It Does

TechMatch is a command-line recommendation engine that maps a user's skills to the most relevant tech career paths. Given 3 or more skills, it returns a ranked list of job roles sorted by similarity score, the same algorithmic logic used by Netflix, Spotify, and Amazon's recommendation systems.

```
  You enter:  Python · Machine Learning · Docker · AWS · NLP

  TechMatch returns:
  ─────────────────────────────────────────────────────────────
  🥇 AI Engineer           ████████████████████████░░░░░░  83%
  🥈 ML Engineer           ████████████████████░░░░░░░░░░  71%
  🥉 Data Scientist        ████████████████░░░░░░░░░░░░░░  58%
  ─────────────────────────────────────────────────────────────
```

---

## Quickstart

**Requirements:** Python 3.10 or higher. No external libraries needed.

```bash
# 1. Clone the repository
git clone https://github.com/Mysrerio-us/Task-3-MuhammadAyas.git
cd Task-3-MuhammadAyas

# 2. Run
python main.py

# 3. Optional flags
python main.py --no-color    # disable ANSI colours (e.g. for plain terminals)
python main.py --debug       # enable verbose engine logging
```

---

## Project Structure

```
Task-3-MuhammadAyas/
├── main.py                  # Entry point — CLI loop & user interaction
├── techmatch/
│   ├── __init__.py          # Package metadata
│   ├── config.py            # All tuneable constants (TOP_N, MIN_SKILLS, etc.)
│   ├── data.py              # Job-role corpus (12 roles) + skill taxonomy
│   ├── display.py           # Terminal rendering — colours, bars, cards
│   ├── engine.py            # Core AI logic — TF-IDF + Cosine Similarity
│   └── exceptions.py        # Custom exception hierarchy
├── tests/
│   ├── __init__.py
│   └── test_engine.py       # 39 unit tests for every engine function
├── .gitignore
├── requirements.txt         # Runtime deps (none)
├── requirements-dev.txt     # Dev deps (pytest)
├── README.md
└── setup.py                 # Package setup

```

---

## How It Works

TechMatch implements every technique from the DecodeLabs Project 3 curriculum:

### 1 · The 4-Step Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  INGESTION  │ ──▶ │   SCORING   │ ──▶ │   SORTING   │ ──▶ │  FILTERING  │
│             │     │             │     │             │     │             │
│  Capture &  │     │  TF-IDF +   │     │  Rank all   │     │  Top-3 only │
│  normalise  │     │  Cosine     │     │  roles by   │     │  returned   │
│  user input │     │  Similarity │     │  score desc │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 2 · Content-Based Filtering

The engine compares the **user's skill attributes** directly against **each job role's tag attributes**. It never requires historical data from other users — this means it works immediately with zero historical data (unlike collaborative filtering).

### 3 · Vector Mapping

Machines understand numbers, not words. Every skill is converted to a position in a **shared vocabulary space**. The normalise step ensures `"Python"` and `"python"` map to the same dimension.

### 4 · TF-IDF Weighting — Beyond Binary 1s and 0s

Simple tag counting treats "Python" (in 9 of 12 roles) identically to "CUDA" (in 1 role). That's wrong — a match on a generic term tells you almost nothing.

**TF-IDF** solves this:

```
TF(t, d)     = count(t in document d) / total tags in d
IDF(t)       = log( (N + 1) / (df(t) + 1) ) + 1
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

The **logarithm in IDF** is the dampening effect — it compresses the penalty scale so values stay comparable. A term in all 12 roles gets IDF ≈ 1.0. A term in only 1 role gets a much higher weight.

### 5 · Cosine Similarity

With both the user profile and each job role expressed as TF-IDF vectors, the engine measures how **directionally aligned** they are:

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

This is **invariant to magnitude** — a role with 20 tags and a role with 5 tags are compared fairly, purely on the orientation of their skill vectors.

| Score | Meaning |
|-------|---------|
| 0.70–1.00 | Strong match — career well-aligned |
| 0.40–0.69 | Moderate match — overlapping skills |
| 0.00–0.39 | Weak match — significant skill gap |

### 6 · Cold Start Bypass

When a user's skills produce a zero-magnitude vector, cosine similarity returns 0.0 for every role — making ranking meaningless. TechMatch detects this and switches to a **trending popularity fallback**, while clearly notifying the user.

---

## Running the Tests

```bash
# With stdlib unittest (no dependencies needed)
python -m unittest discover -s tests -v

# With pytest (if installed)
python -m pytest tests/ -v
```

**39 tests** covering:
- `normalise()` — casing, whitespace, edge cases
- `build_user_profile()` — frequency mapping, validation errors
- `compute_idf()` — IDF values, dampening, empty corpus guard
- `build_tfidf_vector()` — weighting, OOV terms, empty input
- `cosine_similarity()` — identical/orthogonal vectors, symmetry, unit range
- `is_cold_start()` — zero magnitude detection
- `recommend()` — full integration, ranking correctness, error propagation

---

## Configuration

All tuneable constants are in `techmatch/config.py`. Common adjustments:

| Constant | Default | Purpose |
|----------|---------|---------|
| `TOP_N` | `3` | Number of results to return |
| `MIN_SKILLS` | `3` | Minimum skills before engine runs |
| `MAX_SKILLS` | `20` | Maximum skills accepted |
| `IDF_SMOOTHING` | `1.0` | Log dampening constant |
| `SCORE_HIGH_THRESHOLD` | `0.70` | Green match threshold |
| `SCORE_MID_THRESHOLD` | `0.40` | Yellow match threshold |

---

## Extending the Project

**Add job roles** — append a dict to `JOB_CORPUS` in `techmatch/data.py`:
```python
{
    "id": "my-new-role",
    "title": "My New Role",
    "tags": ["skill1", "skill2", "skill3"],
    "description": "What this role does.",
    "popularity": 0.85,
}
```

**Add skills to the picker** — add strings to any category in `SKILL_CATEGORIES`.

**Change result count** — update `TOP_N` in `config.py`.

---

## Algorithms Reference

| Concept | Formula | File |
|---------|---------|------|
| Term Frequency | `TF(t,d) = count(t,d) / \|d\|` | `engine.py` |
| Inverse Document Frequency | `IDF(t) = log((N+1)/(df+1))+1` | `engine.py` |
| TF-IDF Weight | `TF-IDF = TF × IDF` | `engine.py` |
| Cosine Similarity | `cos(θ) = A·B / (‖A‖‖B‖)` | `engine.py` |



<div align="center">
  <sub>Built by Muhammad Ayas</sub>
</div>
