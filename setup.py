"""Setup script for RAG Email Assistant"""
from setuptools import setup, find_packages

setup(
    name="rag-email-assistant",
    version="0.1.0",
    description="Retrieval-Augmented Generation system for email search and QA",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "openai>=1.0.0",
        "gradio>=4.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "ipywidgets",
            "matplotlib",
        ],
        "finetune": [
            "peft>=0.5.0",
            "datasets>=2.14.0",
        ],
    },
)
