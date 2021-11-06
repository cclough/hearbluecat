#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hearbluecat",
    description="Entry for the HEAR 2021 at NeurIPS.",
    author="Christian Clough",
    author_email="christian.clough@gmail.com",
    url="https://github.com/cclough/hearbluecat",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/cclough/hearbluecat/issues",
        "Source Code": "https://github.com/cclough/hearbluecat",
    },
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "librosa",
        # otherwise librosa breaks
        "numba==0.48",
        # tf 2.6.0
        "numpy==1.19.2",
        "tensorflow>=2.0",
        "torch",
        # For wav2vec2 model
        "speechbrain",
        "transformers==4.4.0",
        "torchcrepe",
        "torchopenl3",
        # otherwise librosa breaks
        "numba==0.48",
        # "numba>=0.49.0", # not directly required, pinned by Snyk to avoid a vulnerability
        "scikit-learn>=0.24.2",  # not directly required, pinned by Snyk to avoid a vulnerability
    ],
)
