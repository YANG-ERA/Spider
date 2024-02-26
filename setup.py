
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
VERSION = '0.2.5'

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
        "anndata==0.10.5.post1",
        "matplotlib==3.8.3",
        "numba==0.59.0",
        "numpy==1.23.4",
        "pandas==2.2.1",
        "scanpy==1.9.8",
        "scikit-learn==1.4.1.post1",
        "scipy==1.12.0",
        "seaborn==0.13.2",
        "squidpy==1.4.1",
        "torch==2.2.1",
    ]
)
