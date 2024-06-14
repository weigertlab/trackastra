import os

from ._version import __version__, __version_tuple__

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
