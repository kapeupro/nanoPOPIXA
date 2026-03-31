from setuptools import setup, find_packages

setup(
    name="nanopopixa",
    version="0.1.0",
    description="nanoPOPIXA — Un LLM minimaliste from scratch",
    author="Dimitri",
    py_modules=[
        "model",
        "train",
        "generate",
        "chat",
        "data_prep",
        "scrape",
        "popixa_cli",
        "splash",
        "monitor",
        "session_cache",
    ],
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "tiktoken",
    ],
    entry_points={
        "console_scripts": [
            "popixa=popixa_cli:main",
        ],
    },
    python_requires=">=3.9",
)
