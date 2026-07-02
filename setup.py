"""Setup configuration for Plant Disease Classifier"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plant-disease-classifier",
    version="1.0.0",
    author="ML Engineer",
    description="Production-ready plant disease classification system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/plant-disease-classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.13.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pydantic>=2.4.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "plant-classifier-train=src.train:main",
            "plant-classifier-api=src.api:main",
        ],
    },
)
