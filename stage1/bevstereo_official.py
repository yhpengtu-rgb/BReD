"""BReD 内部 BEVDepth 实验类与数据 collate 入口。

仅保留 v8-a17-ms 训练、评测和冻结导出实际调用的接口；原 smoke 中的
可视化、单样本演示和文件哈希函数不属于两阶段流程，未纳入。
"""
import importlib.util
import sys
from pathlib import Path

BRED_ROOT = Path(__file__).resolve().parents[1]
REPO = BRED_ROOT / "vendor" / "BEVDepth"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bevdepth.datasets.nusc_det_dataset import collate_fn

CONFIG = REPO / "bevdepth/exps/nuscenes/mv/bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema.py"


def load_experiment_class():
    """加载包内官方 EMA+CBGS BEVStereo Lightning 实验类。"""
    spec = importlib.util.spec_from_file_location("bred_bevstereo_ema_cbgs", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BEVDepthLightningModel
