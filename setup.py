"""Setup script for Lobster AI."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read version from version.py
version_file = this_directory / "lobster" / "version.py"
version_dict = {}
with open(version_file) as f:
    exec(f.read(), version_dict)
version = version_dict["__version__"]

# Core requirements
install_requires = [
    # Core
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    "plotly>=5.0.0",
    "rich>=12.0.0",
    "typer>=0.7.0",
    "python-dotenv>=1.0.0",
    
    # LangChain and AI
    "langchain>=0.1.0",
    "langchain-community>=0.0.10",
    "langchain-openai>=0.0.5",
    "langchain-aws>=0.1.0",
    "langgraph>=0.0.20",
    "langgraph-supervisor>=0.0.1",
    "openai>=1.0.0",
    
    # Bioinformatics
    "scanpy>=1.9.3",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "anndata>=0.9.0",
    "biopython>=1.81",
    "leidenalg>=0.9.0",
    "igraph>=0.10.4",
    "scrublet>=0.2.3",
    "h5py>=3.9.0",
    "tables>=3.8.0",
    
    # Visualization
    "seaborn>=0.12.0",
    "matplotlib>=3.7.0",
    "kaleido>=0.2.0",
    
    # Data I/O
    "openpyxl>=3.1.0",
    "pyarrow>=12.0.0",
    
    # HTTP and utilities
    "requests>=2.31.0",
    "aiofiles>=23.0.0",
    "xmltodict>=0.13.0",
    
    # API support
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
]

# Development requirements
dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "pytest-asyncio>=0.20.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "pylint>=2.17.0",
    "mypy>=1.0.0",
    "bandit>=1.7.0",
    "pre-commit>=3.0.0",
    "bumpversion>=0.6.0",
    "twine>=4.0.0",
    "build>=0.10.0",
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "langfuse>=2.0.0",
    'tabulate>=0.9.0',
]

setup(
    name="lobster-ai",
    version=version,
    author="Omics-OS",
    author_email="info@omics-os.com",
    description="Multi-Agent Bioinformatics Analysis System powered by LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/the-omics-os/lobster-ai",
    project_urls={
        "Bug Tracker": "https://github.com/the-omics-os/lobster-ai/issues",
        "Documentation": "https://docs.lobster-ai.com",
        "Source Code": "https://github.com/the-omics-os/lobster-ai",
        "Discord": "https://discord.gg/HDTRbWJ8omicsos",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "all": install_requires + dev_requires,
    },
    entry_points={
        "console_scripts": [
            "lobster=lobster.cli:app",
        ],
    },
    python_requires=">=3.12",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Framework :: FastAPI",
    ],
    keywords="bioinformatics, RNA-seq, single-cell, AI, machine-learning, data-analysis, genomics",
)
