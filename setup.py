import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reina",
    version="0.0.1",
    author="Qyu.ai Inc.",
    author_email="soumil@qyu.ai",
    description="A Causal Inference library for Big Data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Qyu-ai/Reina/",
    project_urls={
        "Bug Tracker": "https://github.com/Qyu-ai/Reina/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "reina"},
    packages=setuptools.find_packages(where="reina"),
    python_requires=">=3.6",
    install_requires=['py4j==0.10.9.2', 'pyspark'],
    extras_require={
        'ml': ['numpy>=1.7'],
        'mllib': ['numpy>=1.7'],
        'sql': [
            'pandas>=0.23.2',
            'pyarrow>=1.0.0',
        ],
        'pandas_on_spark': [
            'pandas>=0.23.2',
            'pyarrow>=1.0.0',
            'numpy>=1.14',
        ],
    }
)
