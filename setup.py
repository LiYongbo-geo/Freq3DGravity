import sys
from os.path import join
import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("freqinv._prism",
              ["freqinv/_prism.pyx"],
              include_dirs=[numpy.get_include()]
    ),
]

exts = cythonize(extensions)

install_requires = ["matplotlib","scipy","Cython","appdirs","numpy","pandas"]

setuptools.setup(
    name="freqinv",
    author="IGP-GRAVITY",
    author_email="gravity_igp@sina.com",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    ext_modules=exts,
    include_package_data=True,
    install_requires=install_requires,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
