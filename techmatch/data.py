from __future__ import annotations

import ast
import csv
import logging
import math
import os
from collections import Counter, defaultdict
from typing import Any

from techmatch import config
from techmatch.exceptions import EmptyCorpusError, MissingRoleFieldError

logger = logging.getLogger(__name__)

_REQUIRED_ROLE_FIELDS: tuple[str, ...] = (
    "id", "title", "tags", "description", "popularity",
)

# Noise tokens 
_STOPWORDS: frozenset[str] = frozenset({
    "", "n/a", "na", "and", "or", "the", "a", "of", "in", "to", "for",
    "with", "at", "by", "an", "as", "is", "it", "on", "up", "its", "etc",
    "able", "good", "strong", "experience", "years", "year",
    "team", "teams", "work", "working", "role", "roles",
})

# Role definitions:
_ROLE_CLASSIFIERS: list[dict[str, Any]] = [
    {
        "id":          "ai-engineer",
        "title":       "AI Engineer",
        "description": (
            "Builds intelligent systems using modern AI/ML. Specialises in LLMs, "
            "generative models, NLP pipelines, and production AI services."
        ),
        "primary":     {"artificial intelligence", "nlp", "natural language processing",
                        "transformers", "llm", "prompt engineering", "langchain",
                        "rag", "fine-tuning", "generative"},
        "secondary":   {"python", "tensorflow", "pytorch", "deep learning",
                        "machine learning"},
    },
    {
        "id":          "data-scientist",
        "title":       "Data Scientist",
        "description": (
            "Extracts insight from complex datasets using statistical modelling and "
            "machine learning. Turns raw data into business decisions."
        ),
        "primary":     {"data scientist", "data science", "machine learning",
                        "statistical analysis", "predictive analytics",
                        "predictive modeling", "regression", "classification",
                        "clustering", "feature engineering"},
        "secondary":   {"python", "r", "sql", "statistics", "data analysis",
                        "pandas", "numpy", "scikit-learn", "tensorflow",
                        "deep learning", "data mining"},
    },
    {
        "id":          "ml-engineer",
        "title":       "Machine Learning Engineer",
        "description": (
            "Bridges research and production. Trains and deploys ML models at scale, "
            "combining software engineering rigour with deep learning expertise."
        ),
        "primary":     {"mlops", "model deployment", "pytorch", "tensorflow",
                        "deep learning", "neural networks", "cuda",
                        "machine learning algorithms"},
        "secondary":   {"python", "docker", "kubernetes", "ci/cd", "aws",
                        "scikit-learn", "spark"},
    },
    {
        "id":          "data-engineer",
        "title":       "Data Engineer",
        "description": (
            "Builds and maintains data infrastructure. Designs ETL pipelines that "
            "process millions of records reliably and feed downstream analytics."
        ),
        "primary":     {"data pipelines", "etl", "hadoop", "spark", "apache spark",
                        "kafka", "airflow", "hive", "data warehouse", "data warehousing",
                        "pyspark", "bigquery", "snowflake", "redshift", "dbt",
                        "large data sets", "big data"},
        "secondary":   {"python", "sql", "aws", "linux", "mongodb",
                        "postgresql", "bash scripting"},
    },
    {
        "id":          "backend-developer",
        "title":       "Backend Developer",
        "description": (
            "Designs and builds server-side logic, databases, and APIs. Focuses on "
            "performance, security, and scalability."
        ),
        "primary":     {"rest api", "api development", "api", "microservices",
                        "node.js", "graphql", "authentication",
                        "system design", "backend"},
        "secondary":   {"python", "java", "sql", "databases", "postgresql",
                        "mongodb", "redis", "docker", "aws", "linux"},
    },
    {
        "id":          "frontend-developer",
        "title":       "Frontend Developer",
        "description": (
            "Crafts interactive, accessible user interfaces. Turns designs into "
            "pixel-perfect, performant web experiences."
        ),
        "primary":     {"javascript", "react", "html", "css", "typescript",
                        "vue.js", "next.js", "responsive design", "ui/ux",
                        "webpack", "figma", "tailwind css", "animation",
                        "accessibility"},
        "secondary":   {"git", "rest api", "graphql", "testing"},
    },
    {
        "id":          "devops-engineer",
        "title":       "DevOps Engineer",
        "description": (
            "Automates infrastructure, deployment pipelines, and monitoring. "
            "Keeps systems reliable, scalable, and secure in production."
        ),
        "primary":     {"docker", "kubernetes", "ci/cd", "terraform", "ansible",
                        "jenkins", "monitoring", "infrastructure as code",
                        "helm", "prometheus", "grafana", "reliability engineering",
                        "devops"},
        "secondary":   {"linux", "aws", "azure", "gcp", "bash scripting",
                        "networking", "security", "git"},
    },
    {
        "id":          "cloud-architect",
        "title":       "Cloud Architect",
        "description": (
            "Designs scalable, cost-effective cloud infrastructure. Responsible for "
            "architecture decisions that affect entire organisations."
        ),
        "primary":     {"cloud", "cloud computing", "aws", "azure", "gcp",
                        "serverless", "architecture"},
        "secondary":   {"terraform", "kubernetes", "microservices", "system design",
                        "networking", "security", "linux"},
    },
    {
        "id":          "cybersecurity-analyst",
        "title":       "Cybersecurity Analyst",
        "description": (
            "Defends systems and networks from threats. Monitors for anomalies, "
            "responds to incidents, and hardens infrastructure against attacks."
        ),
        "primary":     {"security", "ethical hacking", "penetration testing",
                        "firewalls", "cryptography", "siem",
                        "incident response", "vulnerability assessment",
                        "compliance", "threat modelling"},
        "secondary":   {"linux", "python", "networking", "bash scripting", "aws"},
    },
    {
        "id":          "business-analyst",
        "title":       "Business / Data Analyst",
        "description": (
            "Bridges business and data. Translates stakeholder requirements into "
            "data queries, dashboards, and actionable insights."
        ),
        "primary":     {"business intelligence", "tableau", "excel", "business analyst",
                        "reporting", "bi", "business objects", "dashboard",
                        "business requirements", "analytics", "data analytics",
                        "visualization"},
        "secondary":   {"sql", "python", "r", "data analysis", "management"},
    },
    {
        "id":          "mobile-developer",
        "title":       "Mobile Developer",
        "description": (
            "Builds cross-platform or native mobile applications with smooth "
            "performance and tight integration with device APIs."
        ),
        "primary":     {"react native", "flutter", "swift", "kotlin",
                        "android", "ios", "mobile"},
        "secondary":   {"javascript", "typescript", "firebase", "rest api",
                        "ui/ux", "testing"},
    },
    {
        "id":          "fullstack-developer",
        "title":       "Full Stack Developer",
        "description": (
            "Works across the full application stack — UI, server logic, and databases. "
            "A generalist who can ship features end-to-end."
        ),
        "primary":     {"fullstack", "full stack", "full-stack"},
        "secondary":   {"javascript", "python", "react", "node.js", "sql",
                        "mongodb", "rest api", "html", "css", "docker"},
    },
]


# Parse csv
def _parse_csv(path: str) -> list[dict[str, Any]]:
    _PLACEHOLDER = {"", "['']", "['n/a']", "n/a"}
    postings: list[dict[str, Any]] = []

    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("raw_skills", "").strip()
            if raw.lower() in _PLACEHOLDER:
                continue

            try:
                tokens = ast.literal_eval(raw)
                if not isinstance(tokens, list):
                    tokens = [raw]
            except (ValueError, SyntaxError):
                tokens = [raw]

            tags = [
                t.strip().lower()
                for t in tokens
                if t.strip().lower() not in _STOPWORDS and len(t.strip()) > 1
            ]
            if tags:
                postings.append({"tags": tags, "url": row.get("url", "")})

    logger.debug("CSV parsed: %d usable postings.", len(postings))
    return postings


# Role classify
def _classify_posting(tags: list[str]) -> str | None:
    tag_set = set(tags)
    for role in _ROLE_CLASSIFIERS:
        if tag_set & role["primary"]:
            return role["id"]

    for role in _ROLE_CLASSIFIERS:
        if len(tag_set & role["secondary"]) >= 3:
            return role["id"]

    return None  


# corpus builder
def _build_corpus(postings: list[dict]) -> list[dict]:
    clusters: dict[str, list[list[str]]] = defaultdict(list)
    unclassified = 0

    for posting in postings:
        role_id = _classify_posting(posting["tags"])
        if role_id is None:
            role_id = "data-scientist"   
            unclassified += 1
        clusters[role_id].append(posting["tags"])

    logger.debug(
        "Classification complete. %d roles populated. %d postings defaulted to data-scientist.",
        len(clusters), unclassified,
    )

    total = len(postings)
    corpus: list[dict] = []

    role_meta: dict[str, dict] = {r["id"]: r for r in _ROLE_CLASSIFIERS}

    for role_id, tag_lists in clusters.items():
        meta = role_meta[role_id]

        all_tags: list[str] = [tag for tags in tag_lists for tag in tags]
        freq = Counter(all_tags)

        top_tags = [tag for tag, _ in freq.most_common(40)]

        raw_pop = len(tag_lists) / total
        popularity = 0.5 + 0.5 * min(raw_pop / 0.5, 1.0)  # cap at 1.0

        corpus.append({
            "id":          role_id,
            "title":       meta["title"],
            "tags":        top_tags,
            "description": meta["description"],
            "popularity":  round(popularity, 3),
            "posting_count": len(tag_lists),
        })

    corpus.sort(key=lambda r: r["posting_count"], reverse=True)
    logger.info("Corpus built: %d roles from %d postings.", len(corpus), total)
    return corpus


# Picker menu
SKILL_CATEGORIES: dict[str, list[str]] = {
    "Languages & Core": [
        "Python", "SQL", "R", "Java", "JavaScript", "TypeScript",
        "Scala", "C++", "C#", "Julia", "Perl", "Ruby", "Go",
        "Swift", "Kotlin", "Bash Scripting", "Shell Scripting", "SAS",
        "MATLAB", "Solidity",
    ],
    "ML / AI / Data Science": [
        "Machine Learning", "Deep Learning", "NLP",
        "Natural Language Processing", "Statistics",
        "Statistical Analysis", "Data Mining", "Data Science",
        "Predictive Analytics", "Predictive Modeling",
        "Feature Engineering", "Regression", "Classification",
        "Clustering", "Algorithms", "Scikit-Learn",
        "TensorFlow", "PyTorch", "NumPy", "Pandas", "SciPy",
        "Artificial Intelligence", "Neural Networks", "LLM",
        "Transformers", "RAG", "Fine-Tuning",
    ],
    "Data Engineering & Big Data": [
        "Hadoop", "Spark", "Apache Spark", "Hive", "Kafka",
        "Airflow", "ETL", "Data Pipelines", "Data Warehouse",
        "Data Warehousing", "BigQuery", "Snowflake", "Redshift",
        "PySpark", "NoSQL", "MongoDB", "PostgreSQL", "MySQL",
        "Oracle", "SQL Server", "Teradata", "Redis",
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes",
        "CI/CD", "Terraform", "Ansible", "Jenkins", "Linux",
        "Cloud Computing", "Infrastructure as Code", "Monitoring",
        "Prometheus", "Grafana", "Helm", "Serverless",
    ],
    "Web & APIs": [
        "REST API", "API Development", "Node.js", "React",
        "HTML", "CSS", "GraphQL", "Next.js", "Vue.js",
        "TypeScript", "Microservices", "Authentication",
        "Responsive Design", "Figma", "Tailwind CSS",
    ],
    "Domains & Soft Skills": [
        "Data Analysis", "Business Intelligence", "Tableau",
        "Data Visualisation", "Excel", "Analytics",
        "System Design", "Security", "Networking",
        "Research", "Architecture", "Agile", "Scrum",
        "Business Requirements", "Data Modeling",
        "Supply Chain", "Simulation", "Optimisation",
    ],
}


# Module level corpus
def _load() -> list[dict]:
    """Load and build the corpus from CSV. Called once at module import."""
    try:
        postings = _parse_csv(config.CSV_PATH)
        return _build_corpus(postings)
    except FileNotFoundError:
        logger.error("raw_skills.csv not found at %s", config.CSV_PATH)
        return []
    except Exception as exc:
        logger.error("Failed to load CSV corpus: %s", exc)
        return []


JOB_CORPUS: list[dict] = _load()


# Validation
def validate_corpus(corpus: list[dict] | None = None) -> None:

    corpus = corpus if corpus is not None else JOB_CORPUS

    if not corpus:
        raise EmptyCorpusError()

    for role in corpus:
        role_id = role.get("id", "<unknown>")

        for field in _REQUIRED_ROLE_FIELDS:
            if field not in role:
                raise MissingRoleFieldError(role_id, field)

        if not isinstance(role["tags"], list) or len(role["tags"]) == 0:
            raise MissingRoleFieldError(role_id, "tags (must be a non-empty list)")

        pop = role["popularity"]
        if not isinstance(pop, (int, float)) or not (0.0 <= float(pop) <= 1.0):
            raise ValueError(
                f"Role '{role_id}': popularity must be a float in [0.0, 1.0], got {pop!r}."
            )
