"""p11: readers split events randomly without a seed (patch 0006 fixes TrackMLReader only).

Pin, acorn/stages/data_reading/models/trackml_reader.py:31-46: raw_events in glob order, then
`random_split(raw_events, [int(.8n), int(.1n), rest])` with the global (unseeded) torch RNG.
Patched, same file:16-103: complete events sorted by id; `split_lists` (id files) or a generator seeded by
`split_seed` (default 0); listed-but-missing ids, overlaps, unknown keys and duplicate ids raise.

Still random / order dependent at the pin AND after the patches (characterised below):
  athena_reader.py:38-52          torch.manual_seed(42) (global!) then random_split over glob order
  athena_cluster_reader.py:17-29  unseeded random_split over glob order
  acts_reader.py:55-65            unseeded random_split over raw_events[:sum(data_split)], glob order
  colliderml_reader.py:36-42, 92-157  random_split, seeded only with `random_seed` (which reseeds torch, numpy and
                                  random globally); a rerun with a different split raises (dataset_split.csv)
  edge_classifier/models/utils.py:28-63, 124-148  load_dataset / LargeDataset: unseeded random.shuffle subset
                                  (not referenced anywhere in acorn)
Deterministic: athena_root_reader.py:45-122 (event lists); every stage dataset (edge_classifier GraphDataset
edge_classifier_stage.py:642-645 (pin 644-647), track_building GraphDataset track_building_stage.py:414-417,
PartialGraphDataset ml_track_building.py:192-195 via loading_utils.load_datafiles_in_dir:111-129 (pin 96-114), and
graph_construction EventDataset graph_construction_stage.py:319-373) takes the first N in lexicographic order.
"""

import glob as globmod
import os
import random
import re

import pandas as pd
import pytest
import torch

import acorn.stages.data_reading.data_reading_stage as drs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_GLOB = globmod.glob
TRACKML_TERMS = ["truth", "hits", "particles", "cells"]
DET_COLS = ["volume_id", "layer_id", "module_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw", "rot_yu",
            "rot_yv", "rot_yw", "rot_zu", "rot_zv", "rot_zw", "module_t", "module_minhu", "module_maxhu",
            "module_hv", "pitch_u", "pitch_v"]


def touch(path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    open(path, "w").close()


def write_detector(path):
    pd.DataFrame([[7, 2, 1, 0.0, 0.0, 0.0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.15, 8.4, 8.4, 36.0, 0.05, 0.05]],
                 columns=DET_COLS).to_csv(path, index=False)
    return str(path)


def write_lookup(path):
    with open(path, "w") as f:
        f.write("0 0 0 0 0 0.0 0.0 0.0 1 0\n")
    return str(path)


def ids(subset):
    return [int(e["event_id"]) if isinstance(e, dict) else int(e) for e in subset]


def split_ids(reader):
    return tuple(ids(s) for s in (reader.trainset, reader.valset, reader.testset))


@pytest.fixture(autouse=True)
def restore_global_rngs():
    """AthenaReader / ColliderMLReader(random_seed) reseed the process-wide RNGs; do not leak that to other tests."""
    import numpy as np

    states = torch.get_rng_state(), np.random.get_state(), random.getstate()
    yield
    torch.set_rng_state(states[0])
    np.random.set_state(states[1])
    random.setstate(states[2])


def use_glob_order(monkeypatch, descending):
    """data_reading_stage.get_file_names uses glob.glob, whose order is whatever the filesystem returns."""
    monkeypatch.setattr(drs.glob, "glob", lambda p, *a, **k: sorted(REAL_GLOB(p, *a, **k), reverse=descending))


@pytest.fixture
def trackml(tmp_path):
    d = tmp_path / "trackml"
    for i in range(1000, 1010):
        for term in TRACKML_TERMS:
            touch(d / f"event{i:09d}-{term}.csv")
    cfg = dict(input_dir=str(d), detector_path=write_detector(tmp_path / "detector.csv"),
               module_columns=["volume_id", "layer_id", "module_id"], stage_dir=str(tmp_path / "stage"))
    return d, cfg


# --------------------------------------------------------------------------------------------------------------
# TrackMLReader (patch 0006): tests of the fixed behaviour, expected to fail at the pin
# --------------------------------------------------------------------------------------------------------------


def test_trackml_default_split_ignores_global_rng_and_glob_order(trackml, monkeypatch):
    from acorn.stages.data_reading.models.trackml_reader import TrackMLReader

    _, cfg = trackml
    results = []
    for seed, descending in [(1, False), (2, False), (3, True)]:
        use_glob_order(monkeypatch, descending)
        torch.manual_seed(seed)
        results.append(split_ids(TrackMLReader(dict(cfg))))
    assert results[0] == results[1] == results[2]
    train, val, test = results[0]
    assert (len(train), len(val), len(test)) == (8, 1, 1)
    assert sorted(train + val + test) == list(range(1000, 1010))
    # exactly: the id-sorted complete events, random_split with Generator().manual_seed(split_seed=0)
    ref = torch.utils.data.random_split(list(range(1000, 1010)), [8, 1, 1], generator=torch.Generator().manual_seed(0))
    assert results[0] == tuple(list(s) for s in ref)
    # the split changes only through split_seed
    other = split_ids(TrackMLReader(dict(cfg, split_seed=1)))
    assert other != results[0] and sorted(sum(other, [])) == list(range(1000, 1010))


def test_trackml_split_lists_select_exactly_the_listed_events(trackml, tmp_path):
    from acorn.stages.data_reading.models.trackml_reader import TrackMLReader

    _, cfg = trackml
    (tmp_path / "train.txt").write_text("# frozen list\n1003\n1001\n1007  # trailing comment\n\n")
    (tmp_path / "val.txt").write_text("1000\n")
    torch.manual_seed(123)
    reader = TrackMLReader(dict(cfg, split_lists={"trainset": str(tmp_path / "train.txt"),
                                                  "valset": str(tmp_path / "val.txt")}))
    assert split_ids(reader) == ([1003, 1001, 1007], [1000], [])  # file order kept, omitted set empty


def test_trackml_split_lists_are_validated(trackml, tmp_path):
    from acorn.stages.data_reading.models.trackml_reader import TrackMLReader

    _, cfg = trackml
    (tmp_path / "a.txt").write_text("1001\n1002\n")
    (tmp_path / "b.txt").write_text("1002\n")
    (tmp_path / "missing.txt").write_text("1001\n4242\n")
    (tmp_path / "dup.txt").write_text("1001\n1001\n")
    cases = [
        ({"trainset": "a.txt", "testset": "b.txt"}, ValueError, "overlaps"),
        ({"trainset": "missing.txt"}, FileNotFoundError, "lack input files"),
        ({"trainset": "dup.txt"}, ValueError, "Duplicate"),
        ({"trainset": "a.txt", "validset": "b.txt"}, ValueError, "keys must be"),
    ]
    for lists, exc, msg in cases:
        with pytest.raises(exc, match=msg):
            TrackMLReader(dict(cfg, split_lists={k: str(tmp_path / v) for k, v in lists.items()}))


def test_trackml_events_with_missing_input_files_are_never_split(trackml, tmp_path):
    from acorn.stages.data_reading.models.trackml_reader import TrackMLReader

    d, cfg = trackml
    os.remove(d / "event000001009-cells.csv")
    with pytest.warns(UserWarning, match="Dropping 1 events"):
        train, val, test = split_ids(TrackMLReader(dict(cfg)))
    assert 1009 not in train + val + test and len(train + val + test) == 9
    (tmp_path / "t.txt").write_text("1009\n")
    with pytest.raises(FileNotFoundError):
        TrackMLReader(dict(cfg, split_lists={"testset": str(tmp_path / "t.txt")}))


def test_repo_split_files_are_valid_split_lists():
    """Our splits/*.txt (brief 5c) can be passed as split_lists: one id per line, '#' comments, no duplicates."""
    from acorn.stages.data_reading.models.trackml_reader import read_event_id_list

    files = sorted(globmod.glob(os.path.join(REPO, "splits", "*.txt")))
    if not files:
        pytest.skip("no splits/*.txt in this checkout")
    by_sample = {}
    for f in files:
        with open(f) as fh:
            n_header = int(re.search(r":\s*(\d+) events", fh.readline()).group(1))
        lst = read_event_id_list(f)
        assert len(lst) == n_header, f
        sample, split = os.path.basename(f)[: -len(".txt")].rsplit("_", 1)
        by_sample.setdefault(sample, {})[split] = set(lst)
    for sample, splits in by_sample.items():
        names = sorted(splits)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not splits[a] & splits[b], (sample, a, b)


# --------------------------------------------------------------------------------------------------------------
# readers the patches do not touch: characterisation (same at the pin and patched)
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def athena(tmp_path):
    d = tmp_path / "athena"
    for i in range(10):
        for term in ["clusters", "particles", "spacepoints"]:
            touch(d / f"{term}_evt{i:06d}.txt")
    return dict(input_dir=str(d), module_lookup_path=write_lookup(tmp_path / "lookup.txt"), stage_dir=str(tmp_path))


def test_athena_reader_split_depends_on_glob_order_and_reseeds_global_rng(athena, monkeypatch, tmp_path):
    from acorn.stages.data_reading.models.athena_reader import AthenaReader

    use_glob_order(monkeypatch, descending=False)
    torch.manual_seed(1)
    asc = split_ids(AthenaReader(dict(athena)))
    assert torch.initial_seed() == 42  # the reader called torch.manual_seed(42) for the whole process
    torch.manual_seed(2)
    assert split_ids(AthenaReader(dict(athena))) == asc  # fixed seed: same split for the same file order ...
    use_glob_order(monkeypatch, descending=True)
    desc = split_ids(AthenaReader(dict(athena)))
    assert set(desc[0]) != set(asc[0])  # ... but another directory listing order gives another split
    assert [len(s) for s in asc] == [8, 1, 1]
    # int(.8n)+int(.1n)+int(.1n) != n for n = 13: the reader cannot be built at all
    for i in range(10, 13):
        for term in ["clusters", "particles", "spacepoints"]:
            touch(tmp_path / "athena" / f"{term}_evt{i:06d}.txt")
    with pytest.raises(ValueError, match="Sum of input lengths"):
        AthenaReader(dict(athena))


def test_athena_cluster_reader_split_is_unseeded(athena, monkeypatch):
    from acorn.stages.data_reading.models.athena_cluster_reader import AthenaClusterReader

    use_glob_order(monkeypatch, descending=False)
    splits = []
    for seed in (1, 2, 1):
        torch.manual_seed(seed)
        splits.append(split_ids(AthenaClusterReader(dict(athena))))
    assert splits[0] == splits[2]  # driven only by the global torch RNG state
    assert splits[0] != splits[1]


def test_acts_reader_split_is_unseeded_and_its_subset_follows_glob_order(tmp_path, monkeypatch):
    from acorn.stages.data_reading.models.acts_reader import ActsReader

    d = tmp_path / "acts"
    for i in range(10):
        for term in ["truth", "measurements", "particles", "measurement-simhit-map"]:
            touch(d / f"event{i:09d}-{term}.csv")
    cfg = dict(input_dir=str(d), detector_path=write_detector(tmp_path / "detector.csv"), data_split=[4, 2, 2],
               feature_sets={"hit_features": ["hit_x"]}, stage_dir=str(tmp_path))
    use_glob_order(monkeypatch, descending=False)
    splits = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        splits.append(split_ids(ActsReader(dict(cfg))))
    assert splits[0] != splits[1]
    assert sorted(sum(splits[0], [])) == list(range(8))  # the first 8 files in listing order
    use_glob_order(monkeypatch, descending=True)
    assert sorted(sum(split_ids(ActsReader(dict(cfg))), [])) == list(range(2, 10))


def test_colliderml_reader_unseeded_by_default_but_guarded_by_split_csv(tmp_path):
    from acorn.stages.data_reading.models.colliderml_reader import ColliderMLReader

    hits, parts = tmp_path / "hits", tmp_path / "parts"
    os.makedirs(hits)
    os.makedirs(parts)

    def build(stage, seed, **extra):
        os.makedirs(stage, exist_ok=True)
        pd.DataFrame({"event_id": range(10), "tracker_hits": ["h.parquet"] * 10,
                      "particles": ["p.parquet"] * 10}).to_csv(stage / "map.csv", index=False)
        torch.manual_seed(seed)
        return split_ids(ColliderMLReader(dict(stage_dir=str(stage), tracker_hits_dir=str(hits),
                                               particles_dir=str(parts), event_mapping="map.csv", **extra)))

    a, b = build(tmp_path / "s1", 1), build(tmp_path / "s2", 2)
    assert a != b and [len(s) for s in a] == [8, 1, 1]
    with pytest.raises(RuntimeError, match="Split has changed"):
        build(tmp_path / "s2", 3)  # rerun in the same stage_dir with another RNG state
    assert build(tmp_path / "s3", 1, random_seed=5) == build(tmp_path / "s4", 2, random_seed=5)
    assert torch.initial_seed() == 5  # random_seed reseeds the global torch RNG (and numpy, random)


def test_athena_root_reader_takes_sets_from_event_lists(tmp_path):
    from acorn.stages.data_reading.models.athena_root_reader import AthenaRootReader

    lists = {"train": [5, 3, 9], "valid": [1], "test": [7]}
    for name, evs in lists.items():
        (tmp_path / f"{name}.txt").write_text("".join(f"0 {e} {j} f.root\n" for j, e in enumerate(evs)))
    cfg = dict(input_sets={n: str(tmp_path / f"{n}.txt") for n in lists}, stage_dir=str(tmp_path))
    torch.manual_seed(1)
    reader = AthenaRootReader(dict(cfg))
    assert (reader.trainset, reader.valset, reader.testset) == ([5, 3, 9], [1], [7])
    # an event listed twice is silently moved to the later list (the independence check cannot fire)
    (tmp_path / "test.txt").write_text("0 7 0 f.root\n0 3 1 f.root\n")
    reader = AthenaRootReader(dict(cfg))
    assert (reader.trainset, reader.valset, reader.testset) == ([5, 9], [1], [3, 7])


def test_stage_datasets_take_the_first_n_files_in_lexicographic_order(tmp_path):
    from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset as ECDataset
    from acorn.stages.graph_construction.graph_construction_stage import EventDataset
    from acorn.stages.track_building.track_building_stage import GraphDataset as TBDataset

    padded, plain = tmp_path / "padded", tmp_path / "plain"
    order = [10, 3, 1, 2, 30]
    random.Random(0).shuffle(order)  # creation order does not matter
    for i in order:
        touch(padded / "testset" / f"event{i:09d}.pyg.gz")
        touch(plain / "testset" / f"event{i}.pyg.gz")
    for seed in (1, 2):
        torch.manual_seed(seed)
        random.seed(seed)
        for cls in (ECDataset, TBDataset):
            ds = cls(str(padded), "testset", 3, "fit", {})
            assert [os.path.basename(p) for p in ds.input_paths] == [f"event{i:09d}.pyg" for i in (1, 2, 3)]
            ds = cls(str(plain), "testset", 3, "fit", {})  # not zero-padded: lexicographic, not numeric
            assert [os.path.basename(p) for p in ds.input_paths] == ["event1.pyg", "event10.pyg", "event2.pyg"]
        assert EventDataset(str(padded), "testset", 3, hparams={}).evt_ids == [f"{i:09d}" for i in (1, 2, 3)]
        assert EventDataset(str(plain), "testset", 3, hparams={}).evt_ids == ["1", "10", "2"]
    with pytest.raises(AssertionError):
        ECDataset(str(padded), "testset", 6, "fit", {})  # asks for more events than exist


def test_unused_large_dataset_takes_an_unseeded_random_subset(tmp_path):
    from acorn.stages.edge_classifier.models.utils import LargeDataset

    for i in range(10):
        touch(tmp_path / "train" / f"event{i:09d}.pyg")
    subsets = []
    for seed in (1, 2, 1):
        random.seed(seed)
        subsets.append(sorted(os.path.basename(p) for p in LargeDataset(str(tmp_path), "train", 3, {}).input_paths))
    assert subsets[0] == subsets[2] and subsets[0] != subsets[1]


def test_source_audit_of_random_split_and_shuffle_sites():
    """Static audit of the imported acorn tree (ast): random_split calls without generator=, random/np shuffles and
    DataLoader(shuffle=True). At the pin trackml_reader.py is a fifth unseeded random_split."""
    import ast
    import warnings
    from collections import Counter

    import acorn

    base = os.path.dirname(os.path.dirname(os.path.abspath(acorn.__file__)))
    unseeded_split, shuffles, loader_shuffle = Counter(), Counter(), Counter()
    for path in globmod.glob(os.path.join(base, "acorn", "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, base)
        with open(path) as f, warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            keywords = {k.arg: k.value for k in node.keywords}
            if name == "random_split" and "generator" not in keywords:
                unseeded_split[rel] += 1
            if name == "shuffle" and isinstance(node.func, ast.Attribute):
                shuffles[rel] += 1
            if name == "DataLoader" and getattr(keywords.get("shuffle"), "value", None) is True:
                loader_shuffle[rel] += 1
    models = "acorn/stages/data_reading/models/"
    assert dict(unseeded_split) == {models + f: 1 for f in
                                    ("acts_reader.py", "athena_cluster_reader.py", "athena_reader.py",
                                     "colliderml_reader.py")}
    assert dict(shuffles) == {"acorn/stages/edge_classifier/models/utils.py": 2,  # load_dataset, LargeDataset
                              "acorn/utils/plotting_utils.py": 1}  # which tracks to draw in a plot
    assert dict(loader_shuffle) == {"acorn/stages/track_building/models/ml_track_building.py": 1}
