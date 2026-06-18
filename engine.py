# engine.py - Ai Recomendation engine with python only
# Implementing TF-IDF weighting and Cosine similarity
# Pipeline - Ingestion → Scoring → Sorting → Filtering

import math
from collections import Counter



#  Step 1 - Ingestion
#  Get user state as a normalised tag list


def normalise(tag: str) -> str:
    return tag.strip().lower()


def build_user_profile(selected_tags: list[str]) -> dict[str, int]:
    freq = Counter(normalise(t) for t in selected_tags)
    return dict(freq)


#  TF-IDF feature extraction
#  Penalises generic terms, rewards specific ones.


def compute_idf(corpus: list[dict]) -> dict[str, float]:
    
    N = len(corpus)
    doc_freq: dict[str, int] = {}

    for role in corpus:
        # Count each term once per document (set removes duplicates)
        unique_terms = set(normalise(t) for t in role["tags"])
        for term in unique_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    idf: dict[str, float] = {}
    for term, df in doc_freq.items():
        idf[term] = math.log((N + 1) / (df + 1)) + 1

    return idf


def build_tfidf_vector(tags: list[str], idf: dict[str, float]) -> dict[str, float]:
    
    total = len(tags)
    if total == 0:
        return {}

    tf_counts = Counter(normalise(t) for t in tags)

    vector: dict[str, float] = {}
    for term, count in tf_counts.items():
        tf = count / total
        idf_score = idf.get(term, 1.0)   # default 1.0 for out-of-vocab terms
        vector[term] = tf * idf_score

    return vector


#  Step 2 - Scoring — Cosine similarity engine


def dot_product(vec_a: dict, vec_b: dict) -> float:
    return sum(
        vec_a[term] * vec_b[term]
        for term in vec_a
        if term in vec_b
    )


def magnitude(vec: dict) -> float:
    return math.sqrt(sum(w ** 2 for w in vec.values()))


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:

    mag_a = magnitude(vec_a)
    mag_b = magnitude(vec_b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product(vec_a, vec_b) / (mag_a * mag_b)



#  Cold start detection
#  A new user with no history → zero vector →
#  every cosine score = 0. We detect and bypass.


def is_cold_start(user_vec: dict) -> bool:
    return len(user_vec) == 0 or magnitude(user_vec) == 0



#  FULL PIPELINE
#  Ingestion → Scoring → Sorting → Filtering


def recommend(
    selected_tags: list[str],
    corpus: list[dict],
    top_n: int = 3
) -> dict:
    
    # Ingestion
    user_freq = build_user_profile(selected_tags)
    # Reconstruct flat tag list for vectoriser (respects duplicates)
    user_tag_list = [tag for tag, cnt in user_freq.items() for _ in range(cnt)]

    # ── IDF computation
    idf = compute_idf(corpus)

    # Vector Mapping
    user_vec = build_tfidf_vector(user_tag_list, idf)

    # Cold start check
    cold_start = is_cold_start(user_vec)

    if cold_start:
        # Bypass sort by global popularity (trending fallback)
        scored = sorted(corpus, key=lambda r: r.get("popularity", 0), reverse=True)
        for role in scored:
            role = role.copy()
        results = scored[:top_n]
        return {"results": results, "user_vec": {}, "idf": idf, "cold_start": True}

    # Scoring
    scored_roles = []
    for role in corpus:
        role_vec = build_tfidf_vector(role["tags"], idf)
        score = cosine_similarity(user_vec, role_vec)

        # Track which of the user's skills matched this role
        user_terms = set(user_vec.keys())
        role_terms = set(normalise(t) for t in role["tags"])
        matched_tags = sorted(user_terms & role_terms)
        gap_tags = sorted(role_terms - user_terms)

        scored_roles.append({
            **role,
            "score": score,
            "matched_tags": matched_tags,
            "gap_tags": gap_tags,
        })

    # Sorting
    scored_roles.sort(key=lambda r: r["score"], reverse=True)

    # Filtering
    results = scored_roles[:top_n]

    return {
        "results": results,
        "user_vec": user_vec,
        "idf": idf,
        "cold_start": False,
    }
