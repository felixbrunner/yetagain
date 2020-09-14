import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="carrots-erbferninlux", # Replace with your own username
    version="0.0.1",
    author="erbferninlux",
    author_email="",
    description="Package for Hidden Markov Models",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/path/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)