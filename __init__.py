"""ComfyUI outfit captioning node pack: NVIDIA NIM and Gemini providers."""

from __future__ import annotations

from .nodes import gemini, nvidia_nim

NODE_CLASS_MAPPINGS = {
    **nvidia_nim.NODE_CLASS_MAPPINGS,
    **gemini.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **nvidia_nim.NODE_DISPLAY_NAME_MAPPINGS,
    **gemini.NODE_DISPLAY_NAME_MAPPINGS,
}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
