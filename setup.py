from setuptools import setup, find_packages

setup(
    name="smart_city_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.3",
        "scikit-learn>=1.3.2",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.2",
        "geopy>=2.4.1",
        "joblib>=1.3.2",
    ],
    author="Smart City Project Team",
    author_email="example@example.com",
    description="A Smart City system for sustainability analysis and optimization",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)