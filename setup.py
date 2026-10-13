from setuptools import setup, find_packages

setup(
    name="superzip",
    version="1.0.0",
    author="Ibrahim Malek",
    author_email="ibrahim.malek.dev@gmail.com",  # Placeholder contact
    description="Single-file archive tool with a shared Zstandard dictionary and lossless media recompression.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ibrahim-malek/SuperZIP-LLM",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: System :: Archiving :: Compression",
    ],
    python_requires=">=3.8",
    install_requires=[
        "zstandard>=0.15.0",
        "pillow",
    ],
    entry_points={
        "console_scripts": [
            "smartzip=superzip.smartzip:main",
        ],
    },
)
