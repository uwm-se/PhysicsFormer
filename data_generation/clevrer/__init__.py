"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
CLEVRER Data Conversion Module

Scripts for converting CLEVRER benchmark data into physics state tensors.
"""

from .scene_converter import (
    load_clevrer_scene,
    clevrer_scene_to_state_tensor,
    SHAPE_MAP,
    COLOR_MAP,
    MATERIAL_MAP,
    CLEVRER_FPS
)

__all__ = [
    'load_clevrer_scene',
    'clevrer_scene_to_state_tensor',
    'SHAPE_MAP',
    'COLOR_MAP',
    'MATERIAL_MAP',
    'CLEVRER_FPS'
]
