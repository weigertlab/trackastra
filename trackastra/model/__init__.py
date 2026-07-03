import os

from .model import ModelConfig, TrackingTransformer
from .model_api import INFERENCE_CONFIG_KEYS, Trackastra

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
