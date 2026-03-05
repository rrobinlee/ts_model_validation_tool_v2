"""
Setup script for model_validation package.
"""

from setuptools import setup, find_packages

setup(
    name="model_validation",
    version="2.0.0",
    author="Robin Lee",
    description="Python framework for automated model diagnostics and validation",
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "sktime": ["sktime>=0.13.0"],
        "viz": ["seaborn>=0.11.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "model-validate=model_validation.cli:main",
        ],
    },
)