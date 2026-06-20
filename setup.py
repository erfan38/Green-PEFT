from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="energypeft",
    version="0.1.0",
    author="Fatemeh Erfan",
    author_email="fatemeh.erfan@polymtl.ca",
    description="Energy-Aware Parameter-Efficient Fine-Tuning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erfan38/energypeft",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0", 
        "peft>=0.4.0",
        "datasets>=2.10.0",
        "nvidia-ml-py>=12.535.133", # Replaces pynvml
        "psutil>=5.9.0",
        # "codecarbon", # Removed: using internal EnergyMonitor instead
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "accelerate>=0.21.0", # Often needed for HF Trainer
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
    ],
    keywords=[
        "machine-learning",
        "deep-learning",
        "energy-efficiency", 
        "parameter-efficient-fine-tuning",
        "peft",
        "sustainable-ai",
        "llm",
        "transformers",
        "energy-monitoring",
    ],
    include_package_data=True,
)