from setuptools import setup, find_packages

setup(
    name="boloco",
    version="0.1.0",
    description="A script for generating Boolean logic expressions and datasets.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/boloco",
    packages=find_packages(),
    py_modules=["boloco"],
    install_requires=["numpy"],
    entry_points={
        "console_scripts": [
            "boloco=boloco:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    package_data={
        "boloco": [
            "data/**/*.txt"
        ],  # Include all .txt files in `data` and its subdirectories
    },
    include_package_data=True,
)
