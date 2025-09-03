"""Setup script for Lobster Cloud Client."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README.md from parent directory
this_directory = Path(__file__).parent.parent
long_description = (this_directory / "README.md").read_text()

# Version for cloud package
version = "0.1.0"

# Minimal requirements for cloud client
install_requires = [
    # Core dependency
    "lobster-core>=0.1.0",
    
    # HTTP client
    "requests>=2.28.0",
    
    # Utilities
    "typing-extensions>=4.0.0",
]

setup(
    name="lobster-cloud",
    version=version,
    author="Homara AI",
    author_email="support@homara.ai",
    description="Cloud client for Lobster AI bioinformatics system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/homara-ai/lobster-ai",
    project_urls={
        "Bug Tracker": "https://github.com/homara-ai/lobster-ai/issues",
        "Documentation": "https://docs.lobster-ai.com",
        "Source Code": "https://github.com/homara-ai/lobster-ai",
        "Discord": "https://discord.gg/homaraai",
    },
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
    ],
    keywords="bioinformatics, cloud, API, client, AI, machine-learning",
)
