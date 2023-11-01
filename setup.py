from setuptools import setup, find_packages

setup(name='dsir',
      version='0.0.1',
      description='Data Selection with Importance Resampling',
      url='https://github.com/p-lambda/dsir',
      author='Sang Michael Xie',
      author_email='xie@cs.stanford.edu',
      packages=find_packages('.'),
      install_requires=[
        'numpy>=1.21.6',
        'datasets>=2.13.2',
        'tqdm>=4.62.3',
        'joblib>=1.1.0',
        'nltk>=3.7',
      ]
)
