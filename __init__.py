"""ComfyUI Archon Nodes.

Combined node suite for:
- wellkept resolution/prompt nodes
- booru roulette node
"""

import importlib.util
import os


def _load_local_module(module_filename, module_name_hint):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, module_filename)
    spec = importlib.util.spec_from_file_location(module_name_hint, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_wk_mod = _load_local_module("resolution.py", "archon_resolution")
_booru_mod = _load_local_module("booru_roulette.py", "archon_booru_roulette")


def _merge_mappings(*mapping_dicts):
    merged = {}
    for mapping in mapping_dicts:
        for key, value in mapping.items():
            if key in merged:
                raise RuntimeError(f"Duplicate node id detected while merging mappings: {key}")
            merged[key] = value
    return merged


NODE_CLASS_MAPPINGS = _merge_mappings(
    _wk_mod.NODE_CLASS_MAPPINGS,
    _booru_mod.NODE_CLASS_MAPPINGS,
)

NODE_DISPLAY_NAME_MAPPINGS = _merge_mappings(
    _wk_mod.NODE_DISPLAY_NAME_MAPPINGS,
    _booru_mod.NODE_DISPLAY_NAME_MAPPINGS,
)
