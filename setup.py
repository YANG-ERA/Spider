
import io
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Package meta-data.
NAME = 'st-spider'
DESCRIPTION = 'A tools to simulate spatial transcriptomics data.'
EMAIL = '599568651@qq.com'
URL="https://github.com/YANG-ERA/Spider/tree/main"
AUTHOR ='Jiyuan Yang'
VERSION = '1.0.0'

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
	license='MIT',
    description=DESCRIPTION,
	url=URL,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "anndata",
        "matplotlib",
        "numba",
        "numpy",
        "pandas",
        "scanpy",
        "scikit-learn",
        "scipy",
        "seaborn",
        "squidpy"

    ]
)
