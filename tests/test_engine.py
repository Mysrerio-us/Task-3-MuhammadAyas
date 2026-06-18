import math
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from techmatch.engine import (
    normalise,
    build_user_profile,
    compute_idf,
    build_tfidf_vector,
    cosine_similarity,
    is_cold_start,
    recommend,
)
from techmatch.exceptions import (
    EmptyInputError,
    EmptyCorpusError,
    InsufficientSkillsError,
)




MINI_CORPUS = [
    {
        "id": "role-a", "title": "Role A",
        "tags": ["python", "sql", "data analysis"],
        "description": "Role A.", "popularity": 0.9,
    },
    {
        "id": "role-b", "title": "Role B",
        "tags": ["javascript", "react", "css"],
        "description": "Role B.", "popularity": 0.7,
    },
    {
        "id": "role-c", "title": "Role C",
        "tags": ["python", "docker", "kubernetes"],
        "description": "Role C.", "popularity": 0.5,
    },
]



class TestNormalise(unittest.TestCase):

    def test_lowercases(self):
        self.assertEqual(normalise("Python"), "python")

    def test_strips_whitespace(self):
        self.assertEqual(normalise("  SQL  "), "sql")

    def test_empty_string(self):
        self.assertEqual(normalise(""), "")

    def test_already_lowercase(self):
        self.assertEqual(normalise("docker"), "docker")

    def test_mixed_case(self):
        self.assertEqual(normalise("TensorFlow"), "tensorflow")



class TestBuildUserProfile(unittest.TestCase):

    def test_basic_keys_present(self):
        profile = build_user_profile(["Python", "SQL", "Docker"])
        self.assertIn("python", profile)
        self.assertIn("sql", profile)
        self.assertIn("docker", profile)

    def test_frequency_counts(self):
        profile = build_user_profile(["Python", "Python", "SQL", "Docker"])
        self.assertEqual(profile["python"], 2)
        self.assertEqual(profile["sql"], 1)

    def test_empty_raises_empty_input_error(self):
        with self.assertRaises(EmptyInputError):
            build_user_profile([])

    def test_insufficient_raises_error(self):
        with self.assertRaises(InsufficientSkillsError) as ctx:
            build_user_profile(["Python", "SQL"])
        self.assertEqual(ctx.exception.provided, 2)

    def test_normalises_input(self):
        profile = build_user_profile(["PYTHON", "sql", "  Docker  "])
        self.assertIn("python", profile)
        self.assertIn("docker", profile)



class TestComputeIDF(unittest.TestCase):

    def test_returns_non_empty_dict(self):
        idf = compute_idf(MINI_CORPUS)
        self.assertIsInstance(idf, dict)
        self.assertGreater(len(idf), 0)

    def test_all_corpus_terms_present(self):
        idf = compute_idf(MINI_CORPUS)
        self.assertIn("python", idf)
        self.assertIn("javascript", idf)

    def test_shared_terms_have_lower_idf(self):
        # 'python' in 2 docs; 'react' in 1 → python idf < react idf
        idf = compute_idf(MINI_CORPUS)
        self.assertLess(idf["python"], idf["react"])

    def test_empty_corpus_raises(self):
        with self.assertRaises(EmptyCorpusError):
            compute_idf([])

    def test_all_values_positive(self):
        idf = compute_idf(MINI_CORPUS)
        self.assertTrue(all(v > 0 for v in idf.values()))




class TestBuildTFIDFVector(unittest.TestCase):

    def setUp(self):
        self.idf = compute_idf(MINI_CORPUS)

    def test_returns_dict(self):
        vec = build_tfidf_vector(["python", "sql"], self.idf)
        self.assertIsInstance(vec, dict)

    def test_empty_tags_returns_empty_dict(self):
        self.assertEqual(build_tfidf_vector([], self.idf), {})

    def test_all_values_positive(self):
        vec = build_tfidf_vector(["python", "docker"], self.idf)
        self.assertTrue(all(v > 0 for v in vec.values()))

    def test_oov_term_gets_default_weight(self):
        vec = build_tfidf_vector(["completely_unknown_skill"], self.idf)
        self.assertIn("completely_unknown_skill", vec)
        self.assertGreater(vec["completely_unknown_skill"], 0)




class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors_score_one(self):
        vec = {"python": 0.5, "sql": 0.3}
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=9)

    def test_orthogonal_vectors_score_zero(self):
        a = {"python": 1.0}
        b = {"javascript": 1.0}
        self.assertEqual(cosine_similarity(a, b), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(cosine_similarity({}, {"python": 1.0}), 0.0)

    def test_score_in_unit_range(self):
        idf = compute_idf(MINI_CORPUS)
        a = build_tfidf_vector(["python", "sql"], idf)
        b = build_tfidf_vector(["python", "docker"], idf)
        score = cosine_similarity(a, b)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_symmetry(self):
        idf = compute_idf(MINI_CORPUS)
        a = build_tfidf_vector(["python", "sql"], idf)
        b = build_tfidf_vector(["python", "docker"], idf)
        self.assertAlmostEqual(
            cosine_similarity(a, b), cosine_similarity(b, a), places=12
        )




class TestIsColdStart(unittest.TestCase):

    def test_empty_dict_is_cold_start(self):
        self.assertTrue(is_cold_start({}))

    def test_non_empty_vec_not_cold_start(self):
        self.assertFalse(is_cold_start({"python": 0.5}))

    def test_zero_weight_is_cold_start(self):
        self.assertTrue(is_cold_start({"python": 0.0}))




class TestRecommend(unittest.TestCase):

    def test_returns_all_expected_keys(self):
        out = recommend(["python", "sql", "data analysis"], MINI_CORPUS)
        for key in ("results", "user_vec", "idf", "cold_start"):
            self.assertIn(key, out)

    def test_top_n_respected(self):
        out = recommend(["python", "sql", "data analysis"], MINI_CORPUS, top_n=2)
        self.assertEqual(len(out["results"]), 2)

    def test_results_sorted_descending(self):
        out = recommend(["python", "sql", "data analysis"], MINI_CORPUS)
        scores = [r["score"] for r in out["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_most_relevant_role_ranked_first(self):
        out = recommend(["python", "sql", "data analysis"], MINI_CORPUS)
        self.assertEqual(out["results"][0]["id"], "role-a")

    def test_cold_start_false_for_oov_due_to_default_weight(self):
        out = recommend(["assembler", "cobol", "fortran"], MINI_CORPUS)
        self.assertFalse(out["cold_start"])

    def test_cold_start_false_on_normal_input(self):
        out = recommend(["python", "sql", "data analysis"], MINI_CORPUS)
        self.assertFalse(out["cold_start"])

    def test_matched_tags_are_subset_of_role_tags(self):
        out = recommend(["python", "sql", "data analysis"], MINI_CORPUS)
        for role in out["results"]:
            role_tag_set = {t.lower() for t in role["tags"]}
            for matched in role.get("matched_tags", []):
                self.assertIn(matched, role_tag_set)

    def test_empty_corpus_raises(self):
        with self.assertRaises(EmptyCorpusError):
            recommend(["python", "sql", "data analysis"], [])

    def test_insufficient_skills_raises(self):
        with self.assertRaises(InsufficientSkillsError):
            recommend(["python"], MINI_CORPUS)

    def test_empty_skills_raises(self):
        with self.assertRaises(EmptyInputError):
            recommend([], MINI_CORPUS)


if __name__ == "__main__":
    unittest.main(verbosity=2)

class TestColdStartEdgeCases(unittest.TestCase):

    def test_is_cold_start_on_zero_magnitude(self):
        self.assertTrue(is_cold_start({"python": 0.0, "sql": 0.0}))

    def test_oov_skills_produce_non_zero_vector(self):
        idf = compute_idf(MINI_CORPUS)
        vec = build_tfidf_vector(["totally_made_up_skill"], idf)
        self.assertFalse(is_cold_start(vec))
