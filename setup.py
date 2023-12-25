from setuptools import find_packages, setup
import re

_VERSION_FILE = "pylambertw/_version.py"
verstrline = open(_VERSION_FILE, "rt").read()
_VERSION = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(_VERSION, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (_VERSION_FILE,))

pkg_descr = """
Python implementation of the Lambert W x F framework for analyzing skewed, heavy-tailed distribution
with an sklearn interface and torch based maximum likelihood estimation (MLE).
"""

setup(
    name="pylambertw",
    version=verstr,
    url="https://github.com/gmgeorg/pylambertw.git",
    author="Georg M. Goerg",
    author_email="im@gmge.org",
    description=pkg_descr,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.0.1",
        "scipy~=1.6.0",
        "pytest>=6.1.1",
        "pandas>=1.0.0",
        "matplotlib>=3.3.0",
        "statsmodels>=0.12.0",
        "seaborn>=0.11.1",
        "tqdm>=4.46.1",
        "dataclasses>=0.6",
        "scikit-learn>=1.0.1",
        "torchlambertw @ git+ssh://git@github.com/gmgeorg/torchlambertw.git#egg=torchlambertw-0.0.3",
    ],
)
