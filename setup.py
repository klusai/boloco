from setuptools import setup

setup(
    name="boloco",  # Name of your package
    version="0.1.0",  # Version of your package
    description="A script for generating Boolean logic expressions and datasets.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/boloco",  # Your GitHub repo
    py_modules=["boloco"],  # Name of your script without `.py`
    install_requires=[
        "numpy",  # Add dependencies here if your script uses them
    ],
    entry_points={
        "console_scripts": [
            "boloco=boloco:main",  # Command-line tool entry point
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Specify the Python version requirement
)
