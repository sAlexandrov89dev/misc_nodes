"""
from .depth_to_normal import DepthToNormal
from .save_image_16bit import SaveImage16Bit

__all__ = ["DepthToNormal", "SaveImage16Bit"]
"""
import os
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Get directory of this file
nodes_dir = os.path.dirname(__file__)

# Scan all .py files except __init__.py
for filename in os.listdir(nodes_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]
        module = importlib.import_module(f'.{module_name}', package=__name__)

        # Check if module defines its own mappings
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)