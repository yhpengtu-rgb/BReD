"""BReD 第二阶段的包内路径解析。

训练、评测均只从当前 BReD 根目录读取 vendor/assets，并写入 results。
"""
import sys
from pathlib import Path

BRED_ROOT = Path(__file__).resolve().parents[1]
RASTERIZER = BRED_ROOT / "vendor" / "gaussian_rasterization"
if str(RASTERIZER) not in sys.path:
    sys.path.insert(0, str(RASTERIZER))

_PATH_KEYS = {
    "output", "base_dir", "image_dir", "road_gt_dir", "start_checkpoint",
    "v10a1_elevation_dir", "v10a1_cache_dir", "v11a2_rgb_cache_dir",
    "v11a4_geometry_export_dir", "v11a4_geometry_cache_dir",
}


def resolve_config_paths(node):
    """递归把配置中的包内相对路径转为绝对路径，不改变数值/训练协议。"""
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key in _PATH_KEYS and isinstance(value, str) and value:
                node[key] = str(BRED_ROOT / value) if not Path(value).is_absolute() else value
            else:
                resolve_config_paths(value)
    elif isinstance(node, list):
        for value in node:
            resolve_config_paths(value)
    return node
