from setuptools import setup, find_packages

from pathlib import Path
if __name__ == "__main__":
    # parse version from pyproject.toml
    curr_dir = Path(__file__).parent
    with open(curr_dir / 'pyproject.toml', 'r') as f:
        for line in f:
            if line.startswith('version'):
                version = line.split('=')[1].strip().strip('"')
                break

    setup(name='data-selection',
          version=version,
          description='Data Selection with Importance Resampling',
          url='https://github.com/p-lambda/dsir',
          author='Sang Michael Xie',
          author_email='xie@cs.stanford.edu',
          packages=['data_selection'],
          install_requires=[
            'numpy>=1.21.6',
            'tqdm>=4.62.3',
            'joblib>=1.1.0',
            'nltk>=3.8.1',
          ]
    )
