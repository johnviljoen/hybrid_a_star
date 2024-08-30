from setuptools import setup, find_packages

setup(
    name='hybrid_a_star',
    version='0.0.1',
    author='John Viljoen',
    author_email='johnviljoen2@gmail.com',
    install_requires=[
        'matplotlib',   # plotting...
        'tqdm',         # just for pretty loops in a couple places
        'HeapDict',
        'scipy'
    ],
    packages=find_packages(include=[]),
)

