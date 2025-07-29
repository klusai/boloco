from setuptools import setup, find_packages

setup(
    name="boloco",
    version="2.0.0",
    description="A modern Boolean logic expression dataset generator with enhanced features and HuggingFace integration.",
    author="Mihai Nadăș",
    author_email="mihai.nadas@klusai.com",
    url="https://github.com/klusai/boloco",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "enhanced": [
            "datasets>=2.0.0",
            "rich>=10.0.0",
        ],
        "full": [
            "datasets>=2.0.0", 
            "transformers>=4.0.0",
            "rich>=10.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
            "datasets>=2.0.0",
            "rich>=10.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "boloco=boloco.boloco:main",  # Legacy CLI
            "boloco-enhanced=boloco.cli:main",  # Enhanced CLI
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    package_data={
        "boloco": ["data/**/*.txt", "*.md"],
    },
    include_package_data=True,
    keywords="boolean logic, dataset generation, machine learning, logical reasoning, AI",
    long_description=open("README.md", "r", encoding="utf-8").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
)
