try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version('data-selection')

from .base import DSIR
from .hashed_ngram_dsir import HashedNgramDSIR
