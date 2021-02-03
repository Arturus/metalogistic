import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metalogistic",
    version="0.0.4",
    author="Thomas Adamczewski",
    author_email="tmkadamcz@gmail.com",
    description="A Python package for the metalogistic distribution. The metalogistic or metalog distribution is a highly flexible probability distribution that can be used to model data without traditional parameters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tadamcz/metalogistic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        'numpy ~=1.19.3',
        'scipy ~=1.6.0',
        'matplotlib ~=3.3.3'
    ],
    python_requires='>=3.8',
)