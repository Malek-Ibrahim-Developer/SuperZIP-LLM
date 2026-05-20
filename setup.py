from setuptools import setup, find_packages

setup(
    name="superzip-llm",
    version="1.0.0",
    author="Ibrahim Malek",
    author_email="ibrahim.malek.dev@gmail.com",  # Placeholder contact
    description="3x faster LLM inference on low-VRAM GPUs – safe 4-bit compression, no quality loss. Run 70B models on a 4GB GPU.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ibrahim-malek/SuperZIP-LLM",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "zstandard>=0.15.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "airllm>=2.0.0",
        "pillow",
        "numpy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "smartzip=superzip.smartzip:main",
        ],
    },
)
