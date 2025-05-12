import os

from ._version import __version__, __version_tuple__

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Workaround for OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
