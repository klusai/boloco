from setuptools import setup, find_packages

setup(
    name="boloco",
    version="0.1.0",
    description="A script for generating Boolean logic expressions and datasets.",
    author="Mihai Nadăș",
    author_email="mihai.nadas@klusai.com",
    url="https://github.com/klusai/boloco",
    packages=find_packages(),  # Automatically find packages in the project
    install_requires=["numpy"],
    entry_points={
        "console_scripts": [
            "boloco=boloco.boloco:main",  # Adjust entry point to the module inside the package
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    package_data={
        "boloco": ["data/**/*.txt"],  # Include all .txt files in the `data/` folder
    },
    include_package_data=True,
)
