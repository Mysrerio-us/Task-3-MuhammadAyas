<div align="center">

# TechMatch

### AI Tech Stack Recommender

**Content-Based Filtering | TF-IDF Vector Weighting | Cosine Similarity**


---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-39%20passing-brightgreen)
![Dependencies](https://img.shields.io/badge/Dependencies-zero-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-421%20real%20postings-orange)

</div>

---

## What It Does

TechMatch maps your skills to the most relevant tech career paths using a real dataset of **421 job postings** from Dice.com (`raw_skills.csv`). Enter 3 or more skills and get a ranked list of roles sorted by TF-IDF cosine similarity — the same algorithmic logic powering Netflix, Spotify, and Amazon.

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

Make sure raw_skills.csv is in the project root (else use the githublink below to download)
link: dataset: https://github.com/mikeasilva/data-scientist-skills/blob/master/raw_skills.csv?plain=1

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
├── main.py                  # Entry point — CLI loop & user interaction
├── README.md
└── setup.py                 # Package setup

```

---

## How It Works

TechMatch implements every technique from the DecodeLabs Project 3 curriculum:

### The Full v2 Pipeline

```
raw_skills.csv (421 rows)
        │
        ▼
┌──────────────────┐
│  CSV PARSER      │  ast.literal_eval each row → clean tag lists
│  (data.py)       │  discard noise tokens, normalise to lowercase
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ROLE CLASSIFIER │  keyword-set matching → assigns each posting
│  (data.py)       │  to one of 12 career-path clusters
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CORPUS BUILDER  │  aggregate tags per cluster → top-40 frequency
│  (data.py)       │  tags per role; derive popularity from cluster size
└────────┬─────────┘
         │
    JOB_CORPUS (12 enriched role dicts)
         │
         ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  INGESTION  │ → │   SCORING   │ → │   SORTING   │ → │  FILTERING  │
│  user input │   │  TF-IDF +   │   │  descending │   │   Top-N     │
│  validated  │   │  Cosine Sim │   │  by score   │   │  returned   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

### 1 · Data Bridge — old corpus → CSV-powered corpus

The original `data.py` hardcoded 12 job roles by hand. v2 replaces this with a three-stage bridge:

| Stage | What happens |
|-------|-------------|
| **Parse** | Every row in `raw_skills.csv` is parsed with `ast.literal_eval`, cleaned, and noise-filtered |
| **Classify** | Each posting is assigned to a named role cluster via primary-keyword matching (NLP/LLM → AI Engineer; Hadoop/Spark/ETL → Data Engineer; etc.) |
| **Aggregate** | Tags within each cluster are frequency-counted; the top-40 become that role's representative tag list, giving 421× more signal than handcrafted tags |

This means recommendations are now backed by real market data, not intuition.

### 2 · TF-IDF Weighting — beyond binary overlap

```
TF(t, d)     = count(t in d) / total_tags(d)
IDF(t)       = log( (N + s) / (df(t) + s) ) + s     ← logarithm = dampening
TF-IDF(t, d) = TF × IDF
```

`python` appears in 254 of 421 postings — it gets a **low IDF** and contributes little to differentiation. `cuda` appears in 2 — it gets a **high IDF** and powerfully distinguishes ML from Cloud roles.

### 3 · Cosine Similarity

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

Magnitude-invariant — a role cluster built from 200 postings is compared fairly against one built from 10.

### 4 · Cold Start Bypass

If no user skills appear in the corpus vocabulary, the engine detects a zero-magnitude vector and switches to a **trending popularity fallback** (roles ranked by their posting count).

---

## Running the Tests

```bash
# With stdlib unittest
python -m unittest discover -s tests -v

# With pytest
python -m pytest tests/ -v
```

**54 tests** across 6 test classes:

| Class | What it covers |
|-------|---------------|
| `TestNormalise` | casing, whitespace, edge cases |
| `TestBuildUserProfile` | frequency mapping, validation errors |
| `TestComputeIDF` | IDF values, dampening, empty-corpus guard |
| `TestBuildTFIDFVector` | weighting, OOV handling, empty input |
| `TestCosineSimilarity` | identical/orthogonal vectors, symmetry, unit range |
| `TestIsColdStart` | zero-magnitude detection |
| `TestRecommend` | full mini-corpus integration, ranking, error propagation |
| `TestColdStartEdgeCases` | OOV vector non-zero, zero-weight detection |
| `TestCSVCorpus` | live CSV corpus structure + recommendation accuracy |
| `TestSkillCategories` | category completeness, no duplicates |

---

## Configuration

All tuneable constants in `techmatch/config.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `CSV_PATH` | auto-resolved | Path to `raw_skills.csv` |
| `TOP_N` | `3` | Results returned |
| `MIN_SKILLS` | `3` | Minimum skills before engine runs |
| `MAX_SKILLS` | `20` | Maximum skills accepted |
| `IDF_SMOOTHING` | `1.0` | Log dampening constant |
| `OOV_IDF_DEFAULT` | `1.0` | Weight for unknown skills |
| `SCORE_HIGH_THRESHOLD` | `0.70` | Green match label |
| `SCORE_MID_THRESHOLD` | `0.40` | Yellow match label |

---

## Extending the Project

**Add more real postings** — append rows to `raw_skills.csv` in the same format; `data.py` picks them up automatically at next run.

**Add a new role cluster** — append a dict to `_ROLE_CLASSIFIERS` in `data.py`:
```python
{
    "id":          "my-new-role",
    "title":       "My New Role",
    "description": "What this role does.",
    "primary":     {"solidity", "smart contracts", "ethereum"},
    "secondary":   {"python", "javascript", "security"},
}
```

**Change result count** — set `TOP_N` in `config.py`.

---

## Algorithms Reference

| Concept | Formula | File |
|---------|---------|------|
| Term Frequency | `TF(t,d) = count(t,d) / \|d\|` | `engine.py` |
| Inverse Document Frequency | `IDF(t) = log((N+s)/(df+s)) + s` | `engine.py` |
| TF-IDF Weight | `TF-IDF = TF × IDF` | `engine.py` |
| Cosine Similarity | `cos(θ) = A·B / (‖A‖‖B‖)` | `engine.py` |
| Cold Start Detection | `‖user_vec‖ < threshold` | `engine.py` |



<div align="center">
  <sub>Built by Muhammad Ayas</sub>
</div>
