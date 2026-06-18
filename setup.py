from setuptools import setup, find_packages

setup(
    name="techmatch",
    version="2.0.0",
    author="Muhammad Ayas",
    author_email="bahardeenayas8@gmail.com",
    description="AI Tech Stack Recommender — TF-IDF + Cosine Similarity",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mysrerio-us/Task-3-MuhammadAyas.git",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Not yet",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "techmatch=main:main",
        ],
    },
)
