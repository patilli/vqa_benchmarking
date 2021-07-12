import setuptools
from setuptools import setup

setup(
    name='vqabenchmarking',
    url='https://github.tik.uni-stuttgart.de/dirkvaeth/vqa-benchmarking-backend',
    author='Dirk Vaeth ; Pascal Tilli',
    author_email='dirk.vaeth@ims.uni-stuttgart.de ; pascal.tilli@ims.uni-stuttgart.de',
    packages=setuptools.find_packages(include='vqa_benchmarking_backend.*'),#,#setuptools.find_packages(),
    install_requires=['numpy', 'torch'],#, 'nltk', 'spacy'
    version='0.1',
    license='MIT',
    description='Python package to evaluate VQA models with different metrics and datasets.',
    long_description=open('vqa_benchmarking_backend/README.md').read(),
)
