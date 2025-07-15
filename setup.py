from pathlib import Path

from setuptools import find_packages, setup

# Project metadata ----------------------------------------------------
NAME        = "iso_lora"
DESCRIPTION = "IsoLoRA: Isomorphic Expansion + Low‑Rank Adaptation"
VERSION     = "0.1.0"
AUTHOR      = "Noah Schliesman"
EMAIL       = "nschliesman@sandiego.edu"
PY_VERS     = ">=3.10"

# Read long description from README if available
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8") \
    if (this_dir / "README.md").exists() else DESCRIPTION

# Setup ----------------------------------------------------------------
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=PY_VERS,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[  # runtime deps duplicated in requirements/base.txt
        "torch>=2.3",
        "hydra-core>=1.3.2",
        "numpy>=1.28",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2",
            "black>=24.3",
            "ruff>=0.4",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)