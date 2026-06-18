from __future__ import annotations

import math
import logging
from collections import Counter
from typing import Any

from techmatch import config
from techmatch.exceptions import (
    ColdStartError,
    EmptyCorpusError,
    EmptyInputError,
    InsufficientSkillsError,
)

logger = logging.getLogger(__name__)


Vector = dict[str, float]
RecommendationResult = dict[str, Any]

#  Ingestion


def normalise(tag: str) -> str:
    return tag.strip().lower()




def build_user_profile(selected_tags: list[str]) -> dict[str, int]:
    if not selected_tags:
        raise EmptyInputError()

    # Deduplicate at the normalised level before length check
    unique = {normalise(t) for t in selected_tags}
    if len(unique) < config.MIN_SKILLS:
        raise InsufficientSkillsError(
            provided=len(unique),
            required=config.MIN_SKILLS,
        )

    freq: dict[str, int] = Counter(normalise(t) for t in selected_tags)
    logger.debug("User profile built: %d unique term(s).", len(freq))
    return dict(freq)




#  TF-IDF Feature Extraction
#  Upgrades binary 1/0 overlap to weighted vectors.
#  Penalises generic terms (appear in many roles),
#  rewards specific terms (appear in few roles).

def compute_idf(corpus: list[dict]) -> Vector:
    if not corpus:
        raise EmptyCorpusError()

    N = len(corpus)
    s = config.IDF_SMOOTHING
    doc_freq: dict[str, int] = {}

    for role in corpus:
        unique_terms = {normalise(t) for t in role["tags"]}
        for term in unique_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    idf: Vector = {
        term: math.log((N + s) / (df + s)) + s
        for term, df in doc_freq.items()
    }

    logger.debug("IDF computed for %d unique corpus terms.", len(idf))
    return idf





def build_tfidf_vector(tags: list[str], idf: Vector) -> Vector:
    total = len(tags)
    if total == 0:
        return {}

    tf_counts: dict[str, int] = Counter(normalise(t) for t in tags)
    vector: Vector = {}

    for term, count in tf_counts.items():
        tf = count / total
        idf_score = idf.get(term, config.OOV_IDF_DEFAULT)
        vector[term] = tf * idf_score

    return vector



# Scoring

def _dot_product(vec_a: Vector, vec_b: Vector) -> float:
    return sum(
        vec_a[term] * vec_b[term]
        for term in vec_a
        if term in vec_b
    )




def _magnitude(vec: Vector) -> float:
    return math.sqrt(sum(w * w for w in vec.values()))




def cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    mag_a = _magnitude(vec_a)
    mag_b = _magnitude(vec_b)

    if mag_a < config.COLD_START_THRESHOLD or mag_b < config.COLD_START_THRESHOLD:
        return 0.0

    return _dot_product(vec_a, vec_b) / (mag_a * mag_b)




def is_cold_start(user_vec: Vector) -> bool:
    return _magnitude(user_vec) < config.COLD_START_THRESHOLD




def recommend(
    selected_tags: list[str],
    corpus: list[dict],
    top_n: int | None = None,
) -> RecommendationResult:
    
    top_n = top_n if top_n is not None else config.TOP_N

    if not corpus:
        raise EmptyCorpusError()

    # Ingestion
    user_freq = build_user_profile(selected_tags)

    user_tag_list: list[str] = [
        tag for tag, cnt in user_freq.items() for _ in range(cnt)
    ]

    idf = compute_idf(corpus)

    user_vec = build_tfidf_vector(user_tag_list, idf)
    logger.debug("User TF-IDF vector: %d dimensions.", len(user_vec))

    # Cold start check
    if is_cold_start(user_vec):
        logger.warning("Cold Start detected — switching to trending fallback.")
        fallback = sorted(
            corpus,
            key=lambda r: r.get("popularity", 0.0),
            reverse=True,
        )
        return {
            "results": fallback[:top_n],
            "user_vec": {},
            "idf": idf,
            "cold_start": True,
        }
        
        

    # Scoring
    scored_roles: list[dict] = []
    user_terms = set(user_vec.keys())

    for role in corpus:
        role_vec = build_tfidf_vector(role["tags"], idf)
        score = cosine_similarity(user_vec, role_vec)

        role_terms = {normalise(t) for t in role["tags"]}
        matched_tags = sorted(user_terms & role_terms)
        gap_tags = sorted(role_terms - user_terms)

        scored_roles.append({
            **role,
            "score":        score,
            "matched_tags": matched_tags,
            "gap_tags":     gap_tags,
        })

    # Sorting
    scored_roles.sort(key=lambda r: r["score"], reverse=True)

    # Filtering (Top-N) 
    results = scored_roles[:top_n]

    logger.info(
        "Pipeline complete. Top result: '%s' (%.2f%%)",
        results[0]["title"],
        results[0]["score"] * 100,
    )

    return {
        "results":    results,
        "user_vec":   user_vec,
        "idf":        idf,
        "cold_start": False,
    }
