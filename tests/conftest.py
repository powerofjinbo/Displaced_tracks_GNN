import os

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _data_available() -> bool:
    root = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
    return os.path.isdir(os.path.join(root, "unpacked", "WmuHNL10GeV100mm_nopileup_10K"))


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device (run through Slurm on an A30)")
    skip_data = pytest.mark.skip(reason="needs the unpacked npz files on /ceph")
    has_gpu, has_data = _cuda_available(), _data_available()
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "data" in item.keywords and not has_data:
            item.add_marker(skip_data)
