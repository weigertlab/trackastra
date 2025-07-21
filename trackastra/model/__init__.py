# ruff: noqa: F401

import os

from .model import TrackingTransformer
from .model_api import Trackastra

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
