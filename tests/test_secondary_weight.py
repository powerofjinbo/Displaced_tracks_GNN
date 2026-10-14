"""Secondary-particle upweighting (hnl.models.secondary_weight): weights in training events only, config unchanged
from the Canto classifier apart from the model line and the two weighting keys."""

import yaml

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("acorn")
from torch_geometric.data import Data  # noqa: E402

from hnl.models import secondary_weight as SW  # noqa: E402
from tests.test_variants import _hparams  # noqa: E402

CFGS = ("configs/stage1/triplet_symregion_trunc_secw18.yaml",)
REF = "configs/stage1/triplet_symregion_trunc.yaml"


class FakeLookup:
    def decorate(self, ev):
        ev.hit_absd0 = torch.tensor([0.5, 0.5, 50.0, 50.0, float("nan")])
        ev.hit_cat = torch.zeros(5, dtype=torch.int8)
        return ev


def _event():
    return Data(edge_index=torch.tensor([[0, 2, 0, 4], [1, 3, 2, 3]]),
                edge_y=torch.tensor([True, True, False, True]), hit_id=torch.arange(5))


@pytest.mark.parametrize("cfg", CFGS)
def test_weights_train_only(monkeypatch, cfg):
    from hnl.models import vertex_gnn as VG
    monkeypatch.setattr(VG, "_LOOKUP", FakeLookup())
    ds = SW.TruncSecondaryWeightGraphDataset.__new__(SW.TruncSecondaryWeightGraphDataset)
    ds.hparams = _hparams("hnl.models.secondary_weight.TruncSecWeightSymmetricTripletIGNN", cfg)
    w = float(ds.hparams["secw_weight"])
    ds.data_name = "trainset"
    out = ds.construct_weighting(_event())
    assert out.edge_weights.tolist() == [1.0, w, 1.0, 1.0]       # true and |d0| >= cut; fake; NaN |d0| -> 1
    assert "hit_absd0" not in out and "hit_cat" not in out
    ds.data_name = "valset"
    assert ds.construct_weighting(_event()).edge_weights.tolist() == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("cfg", CFGS)
def test_config_differs_only_in_weighting(cfg):
    a, b = yaml.safe_load(open(REF)), yaml.safe_load(open(cfg))
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"model", "secw_d0_mm", "secw_weight"}


def test_model_uses_the_weighted_dataset():
    m = SW.TruncSecWeightSymmetricTripletIGNN(
        _hparams("hnl.models.secondary_weight.TruncSecWeightSymmetricTripletIGNN", CFGS[0]))
    assert m.hparams["dataset_class"] == "TruncSecondaryWeightGraphDataset"
    assert m.dataset_resolver.lookup("TruncSecondaryWeightGraphDataset") is SW.TruncSecondaryWeightGraphDataset
    assert sum(p.numel() for p in m.parameters()) == 347777
