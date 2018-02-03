#!/usr/bin/env python

from distutils.core import setup

setup(name='mnist_model',
        version='0.0.1',
        author='Benjamin Wilhelm',
        author_email='benjamin@b-wilhelm.de',
        description='Library for learning MNIST with a Keras model.',
        license='BSD 2-clause',
        packages=['mnist_model'],
        install_requires=[
            'keras',
            'numpy',
            'pandas',
            'matplotlib',
            'h5py'
            ],
        )
