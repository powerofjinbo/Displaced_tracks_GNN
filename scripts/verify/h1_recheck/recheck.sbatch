#!/bin/bash
#SBATCH -J h1_recheck
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit00,submit01,submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh > /dev/null || exit 1
cd /tmp
W=$(mktemp -d)
cat > $W/recheck.py <<'PYEOF'
"""Independent recheck of claim H1 (addendum H) on H-test, from the stored grids and stored truth."""
import json, sys, glob, time
import numpy as np
import polars as pl

t0 = time.time()
REPO = "/work/submit/jinboz1/hnl/hnl-displaced"
RUNS = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/overlay_h_v1"
ROOT = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/derived/overlay_v1"
CONDS = ["full", "1gev", "hs"]
WPS = ["WP-A", "WP-S", "WP-R-P3"]
GUN_LO, GUN_ISO_HI, GUN_HI = 1001, 1008, 1010
SHIFT = 2 ** 52

# ---- events
split = json.load(open(f"{REPO}/configs/trackml/split_h_v1.json"))
events = sorted(52_000_000 + int(h) for h in split["htest"])
assert len(events) == 600 == len(set(events))
for c in CONDS:
    g = json.load(open(f"{RUNS}/grid/htest_{c}/grid.json"))
    assert sorted(g["events"]) == events, c
    names = json.load(open(f"{RUNS}/grid/htest_{c}/config_names.json"))
    assert names == WPS, names
    assert g["condition"] == c
print("events ok", len(events))

# ---- truth: gun particles
vf = (pl.col("particle_id") // SHIFT)
parts = (pl.scan_parquet(f"{ROOT}/particles/htest/*.parquet")
         .filter(pl.col("event").is_in(events))
         .with_columns(vf.alias("vf"))
         .filter((pl.col("vf") >= GUN_LO) & (pl.col("vf") <= GUN_HI))
         .select(["event", "particle_id", "vf", "pt", "eta", "d0", "rv", "n_hits", "is_target", "r_first",
                  "vx", "vy", "px", "py", "q"])
         .collect())
hits = (pl.scan_parquet(f"{ROOT}/hits/htest/*.parquet")
        .select(["event", "pid", "z"])
        .filter(pl.col("event").is_in(events))
        .with_columns((pl.col("pid") // SHIFT).alias("vf"))
        .filter((pl.col("vf") >= GUN_LO) & (pl.col("vf") <= GUN_HI))
        .collect())
print("truth read", parts.shape, hits.shape, f"{time.time()-t0:.0f}s")
hstat = hits.group_by(["event", "pid"]).agg(pl.len().alias("nh_own"), (pl.col("z").abs() > 1000).any().alias("fwd"))
g = parts.join(hstat.rename({"pid": "particle_id"}), on=["event", "particle_id"], how="left").with_columns(
    pl.col("nh_own").fill_null(0), pl.col("fwd").fill_null(False))
print("n_hits stored == counted from hits:", (g["n_hits"] == g["nh_own"]).all())
g = g.with_columns(((pl.col("pt") > 1) & (pl.col("eta").abs() < 4) & (pl.col("nh_own") >= 5)).alias("tgt"))
print("own target == stored is_target:", (g["tgt"] == g["is_target"]).all(), "n gun targets", g["tgt"].sum())
# independent |d0| about z axis, |B| = 2 T
R = g["pt"].to_numpy().astype(np.float64) / (0.299792458 * 2.0) * 1000.0  # mm
px, py = g["px"].to_numpy().astype(float), g["py"].to_numpy().astype(float)
vx, vy, q = g["vx"].to_numpy().astype(float), g["vy"].to_numpy().astype(float), g["q"].to_numpy().astype(float)
pt = np.hypot(px, py)
for s in (+1, -1):
    xc = vx + s * q * R * py / pt
    yc = vy - s * q * R * px / pt
    d0m = np.abs(np.hypot(xc, yc) - R)
    m = g["tgt"].to_numpy()
    print(f"own |d0| (sign {s:+d}) vs stored |d0| on targets: max abs diff",
          np.nanmax(np.abs(d0m[m] - np.abs(g['d0'].to_numpy()[m]))))
rvm = np.hypot(vx, vy)
print("own r_v vs stored: max abs diff", np.nanmax(np.abs(rvm - g["rv"].to_numpy())))
g = g.with_columns(
    (pl.col("vf") <= GUN_ISO_HI).alias("iso"),
    ((pl.col("d0").abs() >= 10) | (pl.col("rv") >= 100)).alias("disp"),
    (pl.col("d0").abs() / pl.col("r_first")).alias("ratio"),
    pl.col("d0").abs().alias("abs_d0"),
)
g = g.with_columns((pl.col("tgt") & pl.col("iso") & pl.col("disp")).alias("pp"),
                   (pl.col("tgt") & ~pl.col("iso") & pl.col("disp")).alias("pd"))
# soft siblings: another gun particle with the same vertex field in the event, pT <= 1, >= 1 hit
g = g.with_columns(((pl.col("pt") <= 1) & (pl.col("nh_own") >= 1)).cast(pl.Int32).alias("soft"))
g = g.with_columns((pl.col("soft").sum().over(["event", "vf"]) - pl.col("soft")).alias("n_soft_other"))
g = g.with_columns((pl.col("n_soft_other") > 0).alias("soft_sib"))
print("P_P", g["pp"].sum(), "P_D", g["pd"].sum(), "P_P with soft sibling", (g["pp"] & g["soft_sib"]).sum())
pp = g.filter(pl.col("pp"))
print("P_P by region: central", (~pp["fwd"]).sum(), "fwd+", (pp["fwd"] & (pp["eta"] > 0)).sum(),
      "fwd-", (pp["fwd"] & (pp["eta"] < 0)).sum())

# ---- bootstrap index table
n = len(events)
IDX = np.random.default_rng(12345).integers(0, n, size=(500, n))
ev_pos = {e: i for i, e in enumerate(events)}


def per_event_counts(df, col):
    """found and total per event (in event order) for targets in df with found flag column col."""
    a = df.group_by("event").agg(pl.col(col).sum().alias("f"), pl.len().alias("n"))
    F = np.zeros(n); N = np.zeros(n)
    for e, f, k in a.iter_rows():
        F[ev_pos[e]] = f; N[ev_pos[e]] = k
    return F, N


def boot_delta(df, ca, cb):
    Fa, N = per_event_counts(df, ca)
    Fb, N2 = per_event_counts(df, cb)
    assert (N == N2).all()
    ea, eb = 100 * Fa.sum() / N.sum(), 100 * Fb.sum() / N.sum()
    sa, sb, sn = Fa[IDX].sum(1), Fb[IDX].sum(1), N[IDX].sum(1)
    ok = sn > 0
    d = 100 * (sa[ok] - sb[ok]) / sn[ok]
    return dict(n=int(N.sum()), eps_a=ea, eps_b=eb, delta=ea - eb, sigma=float(np.std(d, ddof=1)),
                found_a=int(Fa.sum()), found_b=int(Fb.sum()), skipped=int((~ok).sum()))


def boot_eps(df, c):
    F, N = per_event_counts(df, c)
    return 100 * F.sum() / N.sum(), float(np.std(100 * F[IDX].sum(1) / N[IDX].sum(1), ddof=1)), int(F.sum())


BINS = {"|d0| [mm]": ("abs_d0", [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]),
        "r_v [mm]": ("rv", [0, 10, 30, 100, 200, 300, 450, 650, np.inf]),
        "first-hit radius [mm]": ("r_first", [0, 50, 100, 200, 300, 500, 700, np.inf]),
        "|d0| / r_first": ("ratio", [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.01]),
        "pT [GeV]": ("pt", [1, 2, 5, 10, 20, np.inf])}

# ---- lost share (generator level P_P-like among isolated gun particles)
gt = pl.read_parquet(f"{ROOT}/gun_truth_htest.parquet").filter(pl.col("event").is_in(events))
gvf = gt["particle_id"].to_numpy() // SHIFT
gpx, gpy, gpz = gt["px"].to_numpy(), gt["py"].to_numpy(), gt["pz"].to_numpy()
gpt = np.hypot(gpx, gpy); geta = np.arcsinh(gpz / gpt)
gR = gpt / (0.299792458 * 2.0) * 1000.0
gq = gt["q"].to_numpy().astype(float)
gvx, gvy = gt["vx"].to_numpy(), gt["vy"].to_numpy()
gd0 = np.abs(np.hypot(gvx + gq * gR * gpy / gpt, gvy - gq * gR * gpx / gpt) - gR)
grv = np.hypot(gvx, gvy)
like = (gvf >= GUN_LO) & (gvf <= GUN_ISO_HI) & (gpt > 1) & (np.abs(geta) < 4) & ((gd0 >= 10) | (grv >= 100))
lost = gt["sim_status"].to_numpy() == "failed"
lost_share = lost[like].sum() / like.sum()
print(f"gun particles {len(gt)}, lost {lost.sum()}, P_P-like {like.sum()}, lost among them {lost[like].sum()}, "
      f"share {lost_share:.6f}")

# ---- per working point
res = {}
for wi, wp in enumerate(WPS):
    t = g.select(["event", "particle_id", "pp", "pd", "tgt", "soft_sib", "abs_d0", "rv", "r_first", "ratio", "pt"]
                 ).filter(pl.col("tgt"))
    for c in CONDS:
        o = (pl.read_parquet(f"{RUNS}/grid/htest_{c}/per_target.parquet")
             .filter(pl.col("config") == wi).select(["event", "particle_id", "dm"])
             .with_columns(pl.col("event").cast(pl.Int64)))
        assert o.select(["event", "particle_id"]).is_duplicated().sum() == 0
        t = t.join(o.rename({"dm": f"dm_{c}"}), on=["event", "particle_id"], how="left")
        miss = t.filter(pl.col(f"dm_{c}").is_null())
        print(f"{wp} {c}: gun targets without an output row: {len(miss)} (P_P {miss['pp'].sum()})")
        t = t.with_columns(pl.col(f"dm_{c}").fill_null(False))
    P = t.filter(pl.col("pp"))
    h = boot_delta(P, "dm_full", "dm_1gev")
    h["lower"] = h["delta"] - 2 * h["sigma"]
    h["outcome"] = "holds" if h["lower"] > -1 else ("resolved below" if h["delta"] + 2 * h["sigma"] < -1 else "inconclusive")
    ns = boot_delta(P.filter(~pl.col("soft_sib")), "dm_full", "dm_1gev")
    eff = {}
    for name, sub in (("pp", P), ("pd", t.filter(pl.col("pd"))), ("target", t)):
        eff[name] = {c: boot_eps(sub, f"dm_{c}") for c in CONDS}
        eff[name]["n"] = len(sub)
    # bins and nav bound
    devs, used, below = [], 0, []
    for var, (col, edges) in BINS.items():
        x = P[col].to_numpy()
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            k = int(m.sum())
            if k == 0:
                continue
            sub = P.filter(pl.Series(m))
            db = 100 * (sub["dm_full"].sum() - sub["dm_1gev"].sum()) / k
            if k >= 20:
                devs.append(abs(db - h["delta"])); used += 1
            else:
                below.append((var, lo, hi, k, db))
    bound = lost_share * max(devs)
    # fake rates
    fr = {}
    for c in CONDS:
        pe = pl.read_parquet(f"{RUNS}/grid/htest_{c}/per_event.parquet").filter(pl.col("config") == wi)
        assert pe.height == 600 and sorted(pe["event"].to_list()) == events
        fr[c] = (int(pe["n_fake_ge5"].sum()), int(pe["n_tracks_ge5"].sum()),
                 100 * pe["n_fake_ge5"].sum() / pe["n_tracks_ge5"].sum())
    res[wp] = dict(h1=h, no_soft=ns, eff=eff, nav=dict(max_dev=max(devs), bins_used=used, bound=bound, below=below),
                   fake=fr)
    print(f"\n=== {wp}")
    print(json.dumps(res[wp], indent=1, default=str))
print(f"done {time.time()-t0:.0f}s")
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1, default=str)
PYEOF
python -I $W/recheck.py /dev/null
