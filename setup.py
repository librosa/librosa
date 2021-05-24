import sys
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages


version = SourceFileLoader('librosa.version',
                           'librosa/version.py').load_module()

if __name__ == '__main__':
    setup(version=version.version)
